"""
Clinical Agent — The Triage Nurse.

The most complex and safety-critical agent. Manages clinical assessment
through an adaptive loop (not a linear script):

  analyze_referral → generate_questions → ask_questions ←→ re-evaluate
    → collect_documents → score_risk → complete

Each cycle through the loop:
  1. Evaluate what we know (referral + answers so far)
  2. Identify gaps in clinical picture
  3. Generate the most important next question (LLM tool call)
  4. Re-evaluate after each answer — may skip ahead or circle back

Handles:
  - INTAKE_COMPLETE: Start clinical assessment from referral analysis
  - USER_MESSAGE (clinical phase): Process answers, documents, lab results
  - INTAKE_DATA_PROVIDED: Resume after backward loop
  - GP_RESPONSE: Merge GP-provided data
  - DETERIORATION_ALERT: Reassess without re-booking

Can emit:
  - NEEDS_INTAKE_DATA: backward loop for missing demographics
  - GP_QUERY: request missing clinical data from GP
  - CLINICAL_COMPLETE: hand off to Booking Agent

Uses deterministic RiskScorer — hard rules ALWAYS override LLM.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.agents.risk_scorer import RiskScorer, RiskResult
from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSubPhase,
    CommunicationPlan,
    DeteriorationAssessment,
    PatientDiary,
    Phase,
    RiskLevel,
)
from medforce.gateway.agents.llm_utils import is_response_complete, llm_generate
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.agents.clinical")

# Maximum clinical questions before forcing scoring
MAX_CLINICAL_QUESTIONS = 8

# ── LLM Prompts ──

REFERRAL_ANALYSIS_PROMPT = """\
You are a clinical triage nurse AI reviewing a referral letter. Extract key \
medical information and return a JSON object:

{{
  "chief_complaint": "main reason for referral",
  "condition_context": "identified condition (e.g. cirrhosis, MASH, hepatitis, IBS)",
  "medical_history": ["list of conditions"],
  "current_medications": ["list of medications"],
  "allergies": ["list of allergies"],
  "red_flags": ["any concerning symptoms"],
  "lab_values": {{"parameter": value}},
  "key_findings": "summary of key clinical findings"
}}

Only include fields you can confidently extract. Return ONLY valid JSON.\
"""

QUESTION_GENERATION_PROMPT = """\
You are an experienced clinical triage nurse conducting a pre-consultation \
assessment via chat. Generate the single most important question to ask next.

Patient context:
- Chief complaint: {chief_complaint}
- Condition context: {condition}
- Medical history: {history}
- Medications: {medications}
- Allergies: {allergies}
- Red flags: {red_flags}
- Already asked: {asked}
- Lifestyle data collected: {lifestyle}
- Pain assessed: {pain_assessed}

Requirements:
1. Ask ONE clear, specific, conversational question
2. Prioritise: safety-critical gaps > symptom progression > medication effects > lifestyle
3. If condition is cirrhosis/liver disease → ask about alcohol consumption if not yet asked
4. If condition is MASH/NAFLD → ask about weight, diet, exercise if not yet asked
5. If pain not assessed → ask about pain level (0-10) and location
6. Be warm and empathetic — use the patient's own words when referencing symptoms
7. Never ask about demographics (name, DOB, etc.)
8. Never repeat a question already in the "Already asked" list
9. Avoid vague questions like "What brings you in today?" if the chief complaint is known
10. For patients on multiple medications, ask about adherence or side effects specifically
11. Ask about symptom timeline (when started, getting better/worse) if not yet covered

Return ONLY the question text, nothing else.\
"""

PERSONALIZED_QUESTIONS_PROMPT = """\
You are an experienced clinical triage nurse conducting a pre-consultation \
assessment via chat. Generate the top 5 most clinically important questions \
for this patient. Rank them by clinical importance.

Patient context:
- Chief complaint: {chief_complaint}
- Condition context: {condition}
- Medical history: {history}
- Medications: {medications}
- Known information: {known}

Return a JSON array of 5 strings, ordered by clinical importance:
["most important question", "second most important", ...]

Guidelines:
1. Red-flag symptoms that could indicate emergency (jaundice, confusion, \
   haematemesis, severe pain escalation, chest pain, sudden breathlessness)
2. Symptom timeline and progression — when did it start? getting worse?
3. Medication adherence, side effects, and recent changes
4. Condition-specific lifestyle factors (alcohol for liver, diet for MASH, etc.)
5. Pain assessment with functional impact (how does it affect daily life?)

Each question should be:
- Conversational and empathetic, not robotic
- Specific to THIS patient's conditions, not generic
- Self-contained (doesn't reference previous questions)

Return ONLY the JSON array.\
"""

EXTRACTION_PROMPT = """\
You are a clinical data extraction AI. Extract structured clinical information \
from the patient's response. Return a JSON object with these possible fields:

- chief_complaint: string (main reason for visit)
- medical_history: list of strings (conditions, surgeries, etc.)
- current_medications: list of strings
- allergies: list of strings
- red_flags: list of strings (concerning symptoms like jaundice, confusion, bleeding)
- symptom_details: string (description of symptoms, onset, duration, severity)
- pain_level: integer 0-10 (if patient mentions pain scale)
- pain_location: string (where the pain is)
- lifestyle_alcohol: string (drinking habits if mentioned)
- lifestyle_weight: string (weight/BMI if mentioned)
- lifestyle_diet: string (dietary info if mentioned)
- lifestyle_exercise: string (exercise habits if mentioned)
- lifestyle_smoking: string (smoking status if mentioned)

Only include fields you can confidently extract. If the message doesn't contain \
clinical information, return an empty object {{}}.

Patient message: "{message}"

Return ONLY valid JSON, no markdown, no explanation.\
"""


class ClinicalAgent(BaseAgent):
    """
    Manages clinical assessment through an adaptive re-evaluation loop.

    Rather than following a fixed script, the agent continuously evaluates
    the clinical picture and generates the most important next question.
    """

    agent_name = "clinical"

    def __init__(self, llm_client=None, risk_scorer: RiskScorer | None = None) -> None:
        self._client = llm_client
        self._risk_scorer = risk_scorer or RiskScorer()
        self._model_name = os.getenv("CLINICAL_MODEL", "gemini-2.0-flash")

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            except Exception as exc:
                logger.error("Failed to create Gemini client: %s", exc)
        return self._client

    async def process(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Route to the appropriate handler based on event type and sub-phase."""

        if event.event_type == EventType.INTAKE_COMPLETE:
            return await self._handle_intake_complete(event, diary)

        if event.event_type == EventType.INTAKE_DATA_PROVIDED:
            return await self._handle_intake_data_provided(event, diary)

        if event.event_type == EventType.GP_RESPONSE:
            return await self._handle_gp_response(event, diary)

        if event.event_type == EventType.DETERIORATION_ALERT:
            return await self._handle_deterioration(event, diary)

        if event.event_type == EventType.USER_MESSAGE:
            return await self._handle_user_message(event, diary)

        if event.event_type == EventType.DOCUMENT_UPLOADED:
            return await self._handle_document(event, diary)

        logger.warning(
            "Clinical received unexpected event: %s", event.event_type.value
        )
        return AgentResult(updated_diary=diary)

    # ── Event Handlers ──

    async def _handle_intake_complete(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Intake is done — start clinical assessment with referral analysis."""
        channel = event.payload.get("channel", "websocket")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ANALYZING_REFERRAL)

        # Analyze referral letter if available
        if diary.intake.referral_letter_ref:
            await self._analyze_referral(diary)
            message = (
                "Thank you for your details. I've reviewed your referral letter "
                "and have some context about your case. I'll now ask some "
                "follow-up questions to complete your assessment."
            )
            diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        else:
            message = (
                "Thank you for completing your registration. I'm now going to ask "
                "you some clinical questions to prepare for your consultation. "
                "Let's start — what is the main reason for your visit today?"
            )
            diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)

        # Generate personalized question plan if we have enough context
        if diary.clinical.chief_complaint or diary.clinical.condition_context:
            await self._generate_question_plan(diary)

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    async def _handle_user_message(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """
        Adaptive re-evaluation loop:
          1. Extract clinical data from message
          2. Update clinical picture
          3. Re-evaluate: enough to score? need documents? ask more?
          4. Generate the most important next question
        """
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")
        sub_phase = diary.clinical.sub_phase

        if sub_phase in (ClinicalSubPhase.NOT_STARTED, ClinicalSubPhase.ANALYZING_REFERRAL):
            diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)

        if sub_phase == ClinicalSubPhase.COMPLETE:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "Your clinical assessment is already complete. "
                    "You should hear from our booking team shortly."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # ── Extract clinical data from the message ──
        extracted = await self._extract_clinical_data(text)
        self._apply_extracted_data(diary, extracted)

        # Record the Q&A if we had a pending question
        unanswered = [q for q in diary.clinical.questions_asked if q.answer is None]
        if unanswered and text:
            q = unanswered[0]
            q.answer = text
            q.answered_by = (
                event.sender_id
                if event.sender_role != SenderRole.PATIENT
                else "patient"
            )

        # Check if we need to request missing intake data (backward loop)
        backward_event = self._check_backward_loop_needed(diary)
        if backward_event:
            return AgentResult(
                updated_diary=diary,
                emitted_events=[backward_event],
                responses=[
                    AgentResponse(
                        recipient="patient",
                        channel=channel,
                        message=(
                            "I need a bit more information from you before we continue. "
                            "Our reception team will follow up shortly."
                        ),
                        metadata={"patient_id": event.patient_id},
                    )
                ],
            )

        # Handle document collection phase responses
        if diary.clinical.sub_phase == ClinicalSubPhase.COLLECTING_DOCUMENTS:
            text_lower = text.lower().strip()
            skip_keywords = {"skip", "no", "none", "don't have", "i don't", "no documents", "nothing"}
            if any(kw in text_lower for kw in skip_keywords):
                return await self._score_and_complete(event, diary, channel)

        # ── Re-evaluate: generate question plan if we just got chief complaint ──
        if diary.clinical.chief_complaint and not diary.clinical.generated_questions:
            await self._generate_question_plan(diary)

        # ── Adaptive decision: enough to score, need docs, or ask more? ──
        if self._ready_for_scoring(diary):
            # Check if we should collect documents first
            if (
                diary.clinical.sub_phase != ClinicalSubPhase.COLLECTING_DOCUMENTS
                and not diary.clinical.documents
            ):
                diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
                return self._prompt_for_documents(event, diary, channel)
            return await self._score_and_complete(event, diary, channel)

        # Hard cap — force scoring if too many questions asked
        if len(diary.clinical.questions_asked) >= MAX_CLINICAL_QUESTIONS:
            logger.info(
                "Question cap (%d) reached for patient %s — forcing scoring",
                MAX_CLINICAL_QUESTIONS, event.patient_id,
            )
            return await self._score_and_complete(event, diary, channel)

        # Otherwise, ask the next adaptive question
        return await self._ask_next_question(event, diary, channel)

    async def _handle_document(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Process an uploaded document (lab results, imaging, etc.)."""
        channel = event.payload.get("channel", "websocket")
        file_ref = event.payload.get("file_ref", "")
        doc_type = event.payload.get("type", "unknown")
        content_hash = event.payload.get("content_hash")

        # P3: Document deduplication
        if content_hash and diary.clinical.has_document_hash(content_hash):
            logger.info(
                "Duplicate document skipped for patient %s (hash=%s)",
                event.patient_id, content_hash,
            )
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message="It looks like you've already uploaded this document. We have it on file.",
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        doc = ClinicalDocument(
            type=doc_type,
            source=event.sender_id,
            file_ref=file_ref,
            processed=False,
            content_hash=content_hash,
        )

        lab_values = event.payload.get("extracted_values", {})
        if lab_values:
            doc.extracted_values = lab_values
            doc.processed = True

        diary.clinical.documents.append(doc)

        if diary.clinical.sub_phase != ClinicalSubPhase.COMPLETE:
            diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)

        message = "Thank you for uploading that document. I've added it to your file."

        if lab_values and self._ready_for_scoring(diary):
            return await self._score_and_complete(event, diary, channel)

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id},
        )
        return AgentResult(updated_diary=diary, responses=[response])

    async def _handle_gp_response(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """GP has responded to a query — merge their data."""
        channel = event.payload.get("channel", "websocket")
        lab_results = event.payload.get("lab_results", {})
        attachments = event.payload.get("attachments", [])

        for q in diary.gp_channel.queries:
            if q.status == "pending":
                q.status = "responded"
                q.attachments_received = attachments
                break

        if lab_results:
            doc = ClinicalDocument(
                type="lab_results",
                source=f"gp:{diary.gp_channel.gp_name or 'unknown'}",
                file_ref="",
                processed=True,
                extracted_values=lab_results,
            )
            diary.clinical.documents.append(doc)

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "We've received additional information from your GP. "
                "This will be included in your clinical assessment."
            ),
            metadata={"patient_id": event.patient_id},
        )

        if self._ready_for_scoring(diary):
            return await self._score_and_complete(event, diary, channel)

        return AgentResult(updated_diary=diary, responses=[response])

    async def _handle_intake_data_provided(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Intake has provided the data we requested — resume clinical."""
        channel = event.payload.get("channel", "websocket")

        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)

        return await self._ask_next_question(event, diary, channel)

    async def _handle_deterioration(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Monitoring detected deterioration — reassess risk using assessment data."""
        channel = event.payload.get("channel", "websocket")
        new_values = event.payload.get("new_values", {})
        reason = event.payload.get("reason", "")
        assessment_data = event.payload.get("assessment", {})
        source = event.payload.get("source", "")

        if new_values:
            doc = ClinicalDocument(
                type="deterioration_labs",
                source="monitoring",
                processed=True,
                extracted_values=new_values,
            )
            diary.clinical.documents.append(doc)

        # Re-score risk with new data
        lab_values: dict[str, Any] = {}
        for doc in diary.clinical.documents:
            if doc.extracted_values:
                lab_values.update(doc.extracted_values)

        # If we have assessment data from monitoring's interactive assessment,
        # incorporate it into the risk evaluation
        if assessment_data:
            severity = assessment_data.get("severity", "")
            if severity == "emergency":
                diary.header.risk_level = RiskLevel.CRITICAL
                diary.clinical.risk_level = RiskLevel.CRITICAL
                diary.clinical.risk_reasoning = f"Emergency deterioration: {assessment_data.get('reasoning', reason)}"
            elif severity == "severe":
                diary.header.risk_level = RiskLevel.HIGH
                diary.clinical.risk_level = RiskLevel.HIGH
                diary.clinical.risk_reasoning = f"Severe deterioration: {assessment_data.get('reasoning', reason)}"
            elif severity == "moderate":
                # At least MEDIUM, but could be higher from lab rules
                risk_result = self._risk_scorer.score(diary.clinical, lab_values)
                if self._risk_scorer._risk_rank(risk_result.risk_level) < self._risk_scorer._risk_rank(RiskLevel.MEDIUM):
                    diary.header.risk_level = RiskLevel.MEDIUM
                    diary.clinical.risk_level = RiskLevel.MEDIUM
                else:
                    diary.header.risk_level = risk_result.risk_level
                    diary.clinical.risk_level = risk_result.risk_level
                diary.clinical.risk_reasoning = f"Moderate deterioration: {assessment_data.get('reasoning', reason)}"
            else:
                risk_result = self._risk_scorer.score(diary.clinical, lab_values)
                diary.clinical.risk_level = risk_result.risk_level
                diary.clinical.risk_reasoning = risk_result.reasoning
                diary.header.risk_level = risk_result.risk_level
        else:
            risk_result = self._risk_scorer.score(diary.clinical, lab_values)
            diary.clinical.risk_level = risk_result.risk_level
            diary.clinical.risk_reasoning = risk_result.reasoning
            diary.header.risk_level = risk_result.risk_level

        # If patient already has a confirmed appointment, decide based on assessment
        if diary.booking.confirmed:
            # If assessment says to bring forward, trigger rebooking
            if source in ("deterioration_assessment", "emergency_escalation") and assessment_data:
                recommendation = assessment_data.get("recommendation", "")
                severity = assessment_data.get("severity", "mild")

                if severity == "emergency" or recommendation == "emergency":
                    # Emergency — patient should go to A&E, NOT get appointment slots
                    diary.header.risk_level = RiskLevel.CRITICAL
                    diary.clinical.risk_level = RiskLevel.CRITICAL
                    diary.clinical.risk_reasoning = (
                        f"Emergency deterioration: {assessment_data.get('reasoning', reason)}"
                    )
                    logger.info(
                        "Emergency escalation for patient %s — no rebooking, A&E advised",
                        event.patient_id,
                    )
                    # Return silently — monitoring already told patient to call 999
                    return AgentResult(updated_diary=diary)

                if severity in ("moderate", "severe") or recommendation in ("bring_forward", "urgent_referral"):
                    # Clear confirmed booking so booking agent will offer new slots
                    diary.booking.confirmed = False
                    diary.booking.slots_offered = []
                    diary.booking.slot_selected = None
                    # Reset monitoring state for re-flow (keep entries + alerts for audit)
                    diary.monitoring.communication_plan = CommunicationPlan()
                    diary.monitoring.deterioration_assessment = DeteriorationAssessment()
                    diary.monitoring.monitoring_active = False
                    diary.monitoring.next_scheduled_check = None
                    diary.header.current_phase = Phase.BOOKING

                    logger.info(
                        "Rebooking triggered for patient %s — severity=%s, recommendation=%s",
                        event.patient_id, severity, recommendation,
                    )

                    # Emit CLINICAL_COMPLETE to trigger booking agent
                    # Include a transition message explaining the appointment is being brought forward
                    transition_response = AgentResponse(
                        recipient="patient",
                        channel=channel,
                        message=(
                            "Based on your recent assessment, our clinical team has decided "
                            "to bring your appointment forward. We're now arranging a new, "
                            "earlier appointment for you."
                        ),
                        metadata={"patient_id": event.patient_id},
                    )
                    rebooking_event = EventEnvelope.handoff(
                        event_type=EventType.CLINICAL_COMPLETE,
                        patient_id=event.patient_id,
                        source_agent="clinical",
                        payload={
                            "risk_level": diary.clinical.risk_level.value,
                            "risk_method": "deterioration_reassessment",
                            "risk_reasoning": diary.clinical.risk_reasoning,
                            "condition_context": diary.clinical.condition_context,
                            "rebooking": True,
                            "rebooking_reason": f"Deterioration assessment: {severity}",
                            "channel": channel,
                        },
                        correlation_id=event.correlation_id,
                    )
                    return AgentResult(
                        updated_diary=diary,
                        emitted_events=[rebooking_event],
                        responses=[transition_response],
                    )

                # Mild severity — just acknowledge
                response_msg = self._assessment_based_guidance(assessment_data, diary)
                response = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=response_msg,
                    metadata={
                        "patient_id": event.patient_id,
                        "risk_level": diary.clinical.risk_level.value,
                    },
                )
                return AgentResult(updated_diary=diary, responses=[response])

            # Non-assessment deterioration (e.g., lab values) — existing behavior
            risk_result_for_guidance = RiskResult(
                risk_level=diary.clinical.risk_level,
                method="deterioration_reassessment",
                reasoning=diary.clinical.risk_reasoning or "",
            )
            response_msg = self._deterioration_guidance(risk_result_for_guidance, diary, reason)
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=response_msg,
                metadata={
                    "patient_id": event.patient_id,
                    "risk_level": diary.clinical.risk_level.value,
                },
            )
            return AgentResult(updated_diary=diary, responses=[response])

        return await self._score_and_complete(event, diary, channel)

    def _deterioration_guidance(
        self, risk_result: RiskResult, diary: PatientDiary, reason: str
    ) -> str:
        """Generate risk-stratified guidance for deterioration when booking is confirmed."""
        level = risk_result.risk_level
        appt_info = ""
        if diary.booking.slot_selected:
            appt_info = f" Your appointment on {diary.booking.slot_selected.date} at {diary.booking.slot_selected.time} remains in place."

        if level == RiskLevel.CRITICAL:
            return (
                "Based on your latest results, this requires immediate attention. "
                "Please attend A&E immediately or call 999. "
                "Our clinical team has been alerted."
            )
        if level == RiskLevel.HIGH:
            return (
                "Thank you for this update. Your case now requires urgent clinical review. "
                "Please contact NHS 111 or visit your nearest urgent care centre. "
                "Our clinical team has been notified and will prioritise your case."
                + appt_info
            )
        if level == RiskLevel.MEDIUM:
            return (
                "Thank you — we've noted the changes and our clinical team is reviewing your case. "
                "We may need to bring your appointment forward."
                + appt_info
            )
        return (
            "Thank you for letting us know. We've noted the changes in your file. "
            "Your current appointment remains on track."
            + appt_info
        )

    def _assessment_based_guidance(
        self, assessment_data: dict[str, Any], diary: PatientDiary
    ) -> str:
        """Generate guidance based on monitoring agent's interactive assessment."""
        severity = assessment_data.get("severity", "mild")
        recommendation = assessment_data.get("recommendation", "continue_monitoring")
        reasoning = assessment_data.get("reasoning", "")

        appt_info = ""
        if diary.booking.slot_selected:
            appt_info = (
                f" Your current appointment is on {diary.booking.slot_selected.date} "
                f"at {diary.booking.slot_selected.time}."
            )

        if severity == "emergency":
            return (
                "Our clinical team has reviewed the assessment and confirms this "
                "requires immediate emergency attention. Please call 999 or attend "
                "A&E now. Do not wait."
            )

        if severity == "severe":
            return (
                "Our clinical team has reviewed your assessment and is prioritising "
                "your case. We are working to bring your appointment forward urgently."
                + appt_info
                + " Please contact NHS 111 if your symptoms worsen before we're in touch."
            )

        if severity == "moderate":
            if recommendation == "bring_forward":
                return (
                    "Our clinical team has reviewed your assessment. "
                    "We are arranging to bring your appointment forward so we "
                    "can see you sooner."
                    + appt_info
                    + " We'll be in touch with the new date shortly."
                )
            return (
                "Our clinical team has reviewed your assessment and noted the changes. "
                "We'll keep monitoring closely."
                + appt_info
            )

        # mild — shouldn't normally reach clinical agent, but handle gracefully
        return (
            "Our clinical team has noted your update. Your current monitoring "
            "plan will continue."
            + appt_info
        )

    # ── Referral Analysis ──

    async def _analyze_referral(self, diary: PatientDiary) -> None:
        """LLM-powered referral letter analysis to pre-populate clinical data."""
        try:
            if self.client is None:
                return

            # In production, fetch referral content from GCS.
            # The LLM extracts structured clinical data from the letter.
            raw_response = await llm_generate(
                self.client,
                self._model_name,
                REFERRAL_ANALYSIS_PROMPT
                + f"\n\nReferral reference: {diary.intake.referral_letter_ref}"
                + f"\nGP: {diary.intake.gp_name or 'unknown'}",
            )
            if raw_response is None:
                return

            raw = raw_response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            analysis = json.loads(raw)
            diary.clinical.referral_analysis = analysis
            self._apply_referral_data(diary, analysis)

            logger.info(
                "Referral analysis complete for patient %s: condition=%s",
                diary.header.patient_id,
                analysis.get("condition_context", "unknown"),
            )

        except Exception as exc:
            logger.warning("Referral analysis failed: %s — continuing without", exc)

    def _apply_referral_data(self, diary: PatientDiary, analysis: dict) -> None:
        """Apply extracted referral data to the diary."""
        if analysis.get("chief_complaint"):
            diary.clinical.chief_complaint = analysis["chief_complaint"]
        if analysis.get("condition_context"):
            diary.clinical.condition_context = analysis["condition_context"]
        if analysis.get("medical_history"):
            for item in analysis["medical_history"]:
                if item and item not in diary.clinical.medical_history:
                    diary.clinical.medical_history.append(item)
        if analysis.get("current_medications"):
            for med in analysis["current_medications"]:
                if med and med not in diary.clinical.current_medications:
                    diary.clinical.current_medications.append(med)
        if analysis.get("allergies"):
            for allergy in analysis["allergies"]:
                if allergy and allergy not in diary.clinical.allergies:
                    diary.clinical.allergies.append(allergy)
        if analysis.get("red_flags"):
            for flag in analysis["red_flags"]:
                if flag and flag not in diary.clinical.red_flags:
                    diary.clinical.red_flags.append(flag)

    # ── Personalized Question Generation ──

    async def _generate_question_plan(self, diary: PatientDiary) -> None:
        """Generate top-5 ranked personalized questions using LLM."""
        try:
            if self.client is None:
                diary.clinical.generated_questions = self._fallback_question_plan(diary)
                return

            prompt = PERSONALIZED_QUESTIONS_PROMPT.format(
                chief_complaint=diary.clinical.chief_complaint or "not yet known",
                condition=diary.clinical.condition_context or "not yet identified",
                history=", ".join(diary.clinical.medical_history) or "none reported",
                medications=", ".join(diary.clinical.current_medications) or "none reported",
                known=self._summarize_known_info(diary),
            )

            raw_response = await llm_generate(
                self.client, self._model_name, prompt,
            )
            if raw_response is None:
                diary.clinical.generated_questions = self._fallback_question_plan(diary)
                return

            raw = raw_response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            questions = json.loads(raw)
            if isinstance(questions, list):
                diary.clinical.generated_questions = questions[:5]
                logger.info(
                    "Generated %d personalized questions for patient %s",
                    len(diary.clinical.generated_questions),
                    diary.header.patient_id,
                )

        except Exception as exc:
            logger.warning("Question plan generation failed: %s — using fallback", exc)
            diary.clinical.generated_questions = self._fallback_question_plan(diary)

    def _fallback_question_plan(self, diary: PatientDiary) -> list[str]:
        """Condition-aware fallback questions when LLM is unavailable."""
        questions = []
        condition = (diary.clinical.condition_context or "").lower()

        # Always start with symptom questions if chief complaint unknown
        if not diary.clinical.chief_complaint:
            questions.append(
                "Could you tell me about the main reason for your visit today? "
                "What symptoms or concerns brought you here?"
            )

        # Condition-specific lifestyle questions
        if any(kw in condition for kw in ["cirrhosis", "liver", "hepat"]):
            questions.extend([
                "Can you tell me about your alcohol consumption? How often and how much do you drink?",
                "Have you noticed any yellowing of your skin or eyes, or any swelling in your abdomen?",
            ])
        elif any(kw in condition for kw in ["mash", "nafld", "nash", "fatty liver"]):
            questions.extend([
                "Could you tell me about your current weight and any recent weight changes?",
                "How would you describe your typical diet and exercise routine?",
            ])

        # Universal clinical questions
        if not diary.clinical.medical_history:
            questions.append(
                "Do you have any existing medical conditions or have you had "
                "any significant illnesses or surgeries in the past?"
            )
        if not diary.clinical.current_medications:
            questions.append(
                "Are you currently taking any medications, including "
                "over-the-counter medicines or supplements?"
            )
        if not diary.clinical.allergies:
            questions.append(
                "Do you have any known allergies to medications, foods, or other substances?"
            )
        if diary.clinical.pain_level is None:
            questions.append(
                "On a scale of 0 to 10, where 0 is no pain and 10 is the worst "
                "pain imaginable, how would you rate your current pain level? "
                "And where exactly is the pain located?"
            )

        questions.append(
            "Is there anything else about your health that you think is "
            "important for us to know?"
        )

        return questions[:5]

    # ── Adaptive Question Loop ──

    async def _ask_next_question(
        self, event: EventEnvelope, diary: PatientDiary, channel: str
    ) -> AgentResult:
        """
        Adaptive question selection:
          1. Use pre-generated question plan if available
          2. Otherwise generate contextual question via LLM
          3. Fallback to pattern-based questions
        """
        asked_texts = {q.question for q in diary.clinical.questions_asked}

        # Try to use generated question plan
        if diary.clinical.generated_questions:
            for q in diary.clinical.generated_questions:
                if q not in asked_texts:
                    question_text = q
                    break
            else:
                # All planned questions asked — generate a new one
                question_text = await self._generate_contextual_question(diary)
        else:
            # No plan yet — determine what's missing and generate
            missing_info = self._identify_gaps(diary)
            if not missing_info and len(diary.clinical.questions_asked) >= 3:
                # Check if we should collect documents before scoring
                if (
                    diary.clinical.sub_phase != ClinicalSubPhase.COLLECTING_DOCUMENTS
                    and not diary.clinical.documents
                ):
                    diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
                    return self._prompt_for_documents(event, diary, channel)
                return await self._score_and_complete(event, diary, channel)

            question_text = await self._generate_contextual_question(diary)

        # Record the question
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question=question_text)
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=question_text,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    def _identify_gaps(self, diary: PatientDiary) -> list[str]:
        """Identify what clinical information is still missing."""
        gaps = []
        if not diary.clinical.chief_complaint:
            gaps.append("chief_complaint")
        if not diary.clinical.medical_history:
            gaps.append("medical_history")
        if not diary.clinical.current_medications:
            gaps.append("current_medications")
        if not diary.clinical.allergies:
            gaps.append("allergies")
        if diary.clinical.pain_level is None:
            gaps.append("pain_assessment")
        # Condition-specific gaps
        condition = (diary.clinical.condition_context or "").lower()
        if any(kw in condition for kw in ["cirrhosis", "liver", "hepat"]):
            if "alcohol" not in diary.clinical.lifestyle_factors:
                gaps.append("lifestyle_alcohol")
        if any(kw in condition for kw in ["mash", "nafld", "nash", "fatty"]):
            if "weight" not in diary.clinical.lifestyle_factors:
                gaps.append("lifestyle_weight")
        return gaps

    async def _generate_contextual_question(self, diary: PatientDiary) -> str:
        """Generate the most important next question based on current clinical picture."""
        try:
            if self.client is None:
                return self._fallback_question(diary)

            asked = [q.question[:60] for q in diary.clinical.questions_asked]

            prompt = QUESTION_GENERATION_PROMPT.format(
                chief_complaint=diary.clinical.chief_complaint or "not yet known",
                condition=diary.clinical.condition_context or "not identified",
                history=", ".join(diary.clinical.medical_history) or "none",
                medications=", ".join(diary.clinical.current_medications) or "none",
                allergies=", ".join(diary.clinical.allergies) or "none",
                red_flags=", ".join(diary.clinical.red_flags) or "none",
                asked="; ".join(asked) if asked else "none yet",
                lifestyle=json.dumps(diary.clinical.lifestyle_factors) if diary.clinical.lifestyle_factors else "none",
                pain_assessed="yes" if diary.clinical.pain_level is not None else "no",
            )

            raw = await llm_generate(self.client, self._model_name, prompt)
            if raw and is_response_complete(raw.strip()):
                return raw.strip()
            if raw:
                logger.warning("LLM question response appears truncated — using fallback")

        except Exception as exc:
            logger.warning("LLM question generation failed: %s", exc)

        return self._fallback_question(diary)

    def _fallback_question(self, diary: PatientDiary) -> str:
        """Pattern-based question generation when LLM is unavailable."""
        gaps = self._identify_gaps(diary)

        templates = {
            "chief_complaint": (
                "Could you tell me about the main reason for your visit today? "
                "What symptoms or concerns brought you here?"
            ),
            "medical_history": (
                "Do you have any existing medical conditions or have you had "
                "any significant illnesses or surgeries in the past?"
            ),
            "current_medications": (
                "Are you currently taking any medications, including "
                "over-the-counter medicines or supplements?"
            ),
            "allergies": (
                "Do you have any known allergies to medications, foods, or "
                "other substances?"
            ),
            "pain_assessment": (
                "On a scale of 0 to 10, how would you rate your current pain level? "
                "And where exactly is the pain located?"
            ),
            "lifestyle_alcohol": (
                "Could you tell me about your alcohol consumption? "
                "How often and how much do you typically drink?"
            ),
            "lifestyle_weight": (
                "Could you tell me about your current weight and any recent "
                "weight changes? How would you describe your typical diet?"
            ),
        }

        if gaps:
            return templates.get(
                gaps[0],
                "Is there anything else about your health that you think is important for us to know?",
            )

        return "Is there anything else about your health that you think is important for us to know?"

    def _prompt_for_documents(
        self, event: EventEnvelope, diary: PatientDiary, channel: str
    ) -> AgentResult:
        """Ask the patient for supporting documents."""
        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "Do you have any recent lab results, radiology reports, referral "
                "letters, or NHS app screenshots to share? These help us assess "
                "your case more accurately. You can upload them now, or type "
                "'skip' if you don't have any."
            ),
            metadata={"patient_id": event.patient_id},
        )
        return AgentResult(updated_diary=diary, responses=[response])

    # ── Scoring & Completion ──

    async def _score_and_complete(
        self, event: EventEnvelope, diary: PatientDiary, channel: str
    ) -> AgentResult:
        """Run risk scoring and complete clinical assessment."""
        diary.clinical.advance_sub_phase(ClinicalSubPhase.SCORING_RISK)

        lab_values: dict[str, Any] = {}
        for doc in diary.clinical.documents:
            if doc.extracted_values:
                lab_values.update(doc.extracted_values)

        risk_result = self._risk_scorer.score(diary.clinical, lab_values)

        diary.clinical.risk_level = risk_result.risk_level
        diary.clinical.risk_reasoning = risk_result.reasoning
        diary.clinical.risk_method = risk_result.method
        diary.header.risk_level = risk_result.risk_level

        diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)
        diary.header.current_phase = Phase.BOOKING

        logger.info(
            "Clinical complete for patient %s — risk: %s (method: %s)",
            event.patient_id,
            risk_result.risk_level.value,
            risk_result.method,
        )

        risk_label = risk_result.risk_level.value.upper()
        message = (
            f"Your clinical assessment is now complete. "
            f"Based on the information gathered, your case has been assessed as "
            f"{risk_label} priority. Our booking team will now arrange your "
            f"consultation appointment."
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id, "risk_level": risk_label},
        )

        handoff = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id=event.patient_id,
            source_agent="clinical",
            payload={
                "risk_level": risk_result.risk_level.value,
                "risk_method": risk_result.method,
                "risk_reasoning": risk_result.reasoning,
                "condition_context": diary.clinical.condition_context,
                "lifestyle_factors": diary.clinical.lifestyle_factors,
                "channel": channel,
            },
            correlation_id=event.correlation_id,
        )

        return AgentResult(
            updated_diary=diary,
            emitted_events=[handoff],
            responses=[response],
        )

    def _ready_for_scoring(self, diary: PatientDiary) -> bool:
        """Check if we have enough data to score risk."""
        has_complaint = diary.clinical.chief_complaint is not None
        has_answered = sum(
            1 for q in diary.clinical.questions_asked if q.answer is not None
        )
        has_labs = any(
            doc.extracted_values for doc in diary.clinical.documents
        )

        # Ready if: complaint + enough answered questions, or has lab data
        return (has_complaint and has_answered >= 2) or has_labs

    # ── Backward Loop ──

    def _check_backward_loop_needed(
        self, diary: PatientDiary
    ) -> EventEnvelope | None:
        """Check if we need to send patient back to intake for missing data."""
        if diary.clinical.backward_loop_count >= 3:
            return None

        missing = []
        if not diary.intake.name:
            missing.append("name")
        if not diary.intake.dob:
            missing.append("dob")
        if not diary.intake.nhs_number:
            missing.append("nhs_number")
        if not diary.intake.phone:
            missing.append("phone")

        if not missing:
            return None

        return EventEnvelope.handoff(
            event_type=EventType.NEEDS_INTAKE_DATA,
            patient_id=diary.header.patient_id,
            source_agent="clinical",
            payload={
                "missing_fields": missing,
                "reason": "Clinical verification requires contact information",
                "channel": "websocket",
            },
        )

    # ── Data Extraction ──

    def _apply_extracted_data(
        self, diary: PatientDiary, extracted: dict[str, Any]
    ) -> None:
        """Apply extracted clinical data to the diary."""
        if "chief_complaint" in extracted and extracted["chief_complaint"]:
            diary.clinical.chief_complaint = extracted["chief_complaint"]

        if "medical_history" in extracted:
            for item in extracted["medical_history"]:
                if item and item not in diary.clinical.medical_history:
                    diary.clinical.medical_history.append(item)

        if "current_medications" in extracted:
            for med in extracted["current_medications"]:
                if med and med not in diary.clinical.current_medications:
                    diary.clinical.current_medications.append(med)

        if "allergies" in extracted:
            new_allergies = [a for a in extracted["allergies"] if a]
            has_specific = any(a != "NKDA" for a in new_allergies)
            # If patient now reports a specific allergy, clear the NKDA placeholder
            if has_specific and "NKDA" in diary.clinical.allergies:
                diary.clinical.allergies.remove("NKDA")
            for allergy in new_allergies:
                # Adding a specific allergy supersedes NKDA
                if allergy == "NKDA" and diary.clinical.allergies:
                    continue
                if allergy not in diary.clinical.allergies:
                    diary.clinical.allergies.append(allergy)

        if "red_flags" in extracted:
            for flag in extracted["red_flags"]:
                if flag and flag not in diary.clinical.red_flags:
                    diary.clinical.red_flags.append(flag)

        # Pain assessment
        if "pain_level" in extracted:
            try:
                diary.clinical.pain_level = int(extracted["pain_level"])
            except (ValueError, TypeError):
                pass
        if "pain_location" in extracted:
            diary.clinical.pain_location = extracted["pain_location"]

        # Lifestyle factors
        lifestyle_keys = [
            "lifestyle_alcohol", "lifestyle_weight", "lifestyle_diet",
            "lifestyle_exercise", "lifestyle_smoking",
        ]
        for key in lifestyle_keys:
            if key in extracted and extracted[key]:
                factor = key.replace("lifestyle_", "")
                diary.clinical.lifestyle_factors[factor] = extracted[key]

        # Detect condition context from clinical data if not set
        if not diary.clinical.condition_context:
            diary.clinical.condition_context = self._detect_condition(diary)

    def _detect_condition(self, diary: PatientDiary) -> str | None:
        """Detect the primary condition context from clinical data."""
        all_text = " ".join([
            diary.clinical.chief_complaint or "",
            " ".join(diary.clinical.medical_history),
            " ".join(diary.clinical.red_flags),
        ]).lower()

        conditions = {
            "cirrhosis": ["cirrhosis", "liver cirrhosis", "decompensated"],
            "hepatitis": ["hepatitis", "hep b", "hep c", "hepatitis b", "hepatitis c"],
            "MASH": ["mash", "nafld", "nash", "fatty liver", "non-alcoholic"],
            "liver_disease": ["liver disease", "liver failure", "hepatic"],
            "ibs": ["ibs", "irritable bowel"],
        }

        for condition, keywords in conditions.items():
            if any(kw in all_text for kw in keywords):
                return condition
        return None

    def _summarize_known_info(self, diary: PatientDiary) -> str:
        """Build a summary of what we already know about the patient."""
        parts = []
        if diary.clinical.chief_complaint:
            parts.append(f"Complaint: {diary.clinical.chief_complaint}")
        if diary.clinical.medical_history:
            parts.append(f"History: {', '.join(diary.clinical.medical_history)}")
        if diary.clinical.current_medications:
            parts.append(f"Meds: {', '.join(diary.clinical.current_medications)}")
        if diary.clinical.allergies:
            parts.append(f"Allergies: {', '.join(diary.clinical.allergies)}")
        if diary.clinical.pain_level is not None:
            parts.append(f"Pain: {diary.clinical.pain_level}/10")
        if diary.clinical.lifestyle_factors:
            parts.append(f"Lifestyle: {json.dumps(diary.clinical.lifestyle_factors)}")
        return "; ".join(parts) if parts else "minimal data"

    # ── LLM Integration ──

    async def _extract_clinical_data(self, text: str) -> dict[str, Any]:
        """Extract clinical data from free text using LLM with fallback."""
        if not text:
            return {}

        try:
            if self.client is None:
                return self._fallback_extraction(text)

            prompt = EXTRACTION_PROMPT.format(message=text)
            raw_response = await llm_generate(self.client, self._model_name, prompt)
            if raw_response is None:
                return self._fallback_extraction(text)

            raw = raw_response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            return json.loads(raw)

        except Exception as exc:
            logger.warning("LLM clinical extraction failed: %s — using fallback", exc)
            return self._fallback_extraction(text)

    def _fallback_extraction(self, text: str) -> dict[str, Any]:
        """Pattern-based clinical extraction."""
        extracted: dict[str, Any] = {}
        text_lower = text.lower()

        # Red flags
        red_flag_keywords = [
            "jaundice", "confusion", "bleeding", "ascites",
            "encephalopathy", "hematemesis", "melena",
        ]
        flags = [kw for kw in red_flag_keywords if kw in text_lower]
        if flags:
            extracted["red_flags"] = flags

        # Chief complaint
        complaint_phrases = [
            "i have", "i've been", "suffering from", "experiencing",
            "my problem is", "referred for", "reason for visit",
        ]
        for phrase in complaint_phrases:
            if phrase in text_lower and "chief_complaint" not in extracted:
                idx = text_lower.index(phrase)
                remainder = text[idx + len(phrase):].strip()
                if remainder:
                    extracted["chief_complaint"] = remainder[:200]
                break

        # Pain level
        import re
        # Match "pain/level/scale ... N" or bare "N out of 10" / "N/10"
        pain_match = re.search(r'(?:pain|level|scale).*?(\d+)\s*(?:/\s*10|out of 10)?', text_lower)
        if not pain_match:
            pain_match = re.search(r'(\d+)\s*(?:/\s*10|out of 10)', text_lower)
        if pain_match:
            level = int(pain_match.group(1))
            if 0 <= level <= 10:
                extracted["pain_level"] = level

        # Lifestyle: alcohol
        alcohol_phrases = ["drink", "alcohol", "beer", "wine", "spirits", "units"]
        if any(phrase in text_lower for phrase in alcohol_phrases):
            extracted["lifestyle_alcohol"] = text[:200]

        # Lifestyle: weight/diet
        weight_phrases = ["weight", "kg", "stone", "bmi", "diet", "eat"]
        if any(phrase in text_lower for phrase in weight_phrases):
            extracted["lifestyle_weight"] = text[:200]

        # Allergies
        allergy_phrases = ["allerg", "allergic to", "no known allerg", "nkda"]
        if any(phrase in text_lower for phrase in allergy_phrases):
            if "no known" in text_lower or "nkda" in text_lower or "no allerg" in text_lower:
                extracted["allergies"] = ["NKDA"]
            else:
                idx = text_lower.find("allerg")
                if idx >= 0:
                    extracted["allergies"] = [text[idx:idx+100].strip()]

        # Medications
        med_keywords = ["take", "taking", "prescribed", "medication", "medicine", "mg", "daily"]
        if any(kw in text_lower for kw in med_keywords):
            # Simple extraction: capture medication-like patterns
            med_matches = re.findall(r'(\w+\s+\d+\s*mg)', text, re.IGNORECASE)
            if med_matches:
                extracted["current_medications"] = med_matches

        # Medical history
        history_phrases = ["diagnosed", "surgery", "operation", "condition", "disease"]
        if any(phrase in text_lower for phrase in history_phrases):
            extracted["medical_history"] = [text[:200]]

        return extracted

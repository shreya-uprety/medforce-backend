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
    CrossPhaseState,
    DeteriorationAssessment,
    PatientDiary,
    Phase,
    RiskLevel,
)
from medforce.gateway.agents.llm_utils import is_response_complete, llm_generate
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.agents.clinical")

# Maximum clinical questions before forcing scoring (safety cap, not target).
# With referral-first intake + adaptive follow-ups: 5 plan + up to 5 follow-ups + 2 safety Qs.
MAX_CLINICAL_QUESTIONS = 12

# Maximum number of adaptive question regeneration cycles
MAX_ADAPTIVE_REGENERATIONS = 5

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
You are a {specialty} in a pre-consultation chat. Generate ONE \
follow-up question that builds on the patient's latest answer while covering \
a NEW aspect of their condition.

CLINICAL SUMMARY (do NOT re-ask):
{clinical_summary}

Conversation so far:
{qa_history}

Rules:
1. Reference what the patient just said using their own words
2. Then pivot to a DIFFERENT finding from the clinical summary that hasn't \
   been discussed yet in the conversation
3. Your question MUST name a SPECIFIC finding from the clinical summary — \
   a symptom name (e.g. "the pain in your right side", "the weight loss"), \
   a named test (e.g. "your liver scan", "your blood tests"), or a general \
   reference (e.g. "some changes on your blood tests")
4. SAFETY — NEVER quote exact measurements, lab values, tumour sizes, or \
   diagnostic terms (mass, lesion, carcinoma) to the patient. \
   Say "your liver scan" not "the 4.2 cm mass on ultrasound". \
   Say "your blood tests showed some raised markers" not "AFP 485 kU/L"
5. PLAIN ENGLISH — NEVER use medical terms, abbreviations, or jargon. \
   BAD: "RUQ", "right upper quadrant", "scapula", "palpable" \
   GOOD: "right side of your tummy", "shoulder blade" \
   Translate: quadrant→side, scapula→shoulder blade, abdomen→tummy
6. If the patient said something is "worse", briefly acknowledge, then ask \
   about a different symptom or concern from the summary
7. NEVER ask vague questions like "have you noticed any changes in your \
   appetite" or "how are you feeling" — instead tie it to a specific finding: \
   "since the weight loss your GP noted, has your appetite changed?"
8. NEVER re-ask medications, allergies, or anything already covered
9. Each follow-up should cover NEW ground — do not keep drilling into the \
   same symptom for multiple turns

Return ONLY the question text.\
"""

PERSONALIZED_QUESTIONS_PROMPT = """\
You are a {specialty} starting a pre-consultation chat with a \
GP-referred patient. Using the clinical summary below, generate exactly 5 \
targeted questions. Each question MUST reference a specific data point from \
the summary (a number, a lab value, an imaging finding, a named symptom).

CLINICAL SUMMARY FROM REFERRAL (all of this is already known — NEVER re-ask):
{clinical_summary}

DO NOT ask about: medications, allergies, documents, demographics, or anything \
stated in the summary above.

Generate 5 questions that ONLY a conversation with the patient can answer. \
Each question must name a SPECIFIC finding from the summary and ask about \
something that has happened SINCE the GP visit:

1. Symptom progression — reference a SPECIFIC symptom the patient knows \
   about and ask whether it has changed since the GP visit. \
   GOOD: "The weight loss your GP mentioned — has it continued or stabilised?" \
   BAD: "Have you had any weight changes recently?" (no anchor)
2. New symptom screening — anchor the screening question to a SPECIFIC \
   finding from the referral that makes this symptom clinically relevant. \
   GOOD: "Your blood tests showed some markers were a bit raised — \
   have you noticed any yellowing of your skin or eyes since then?" \
   BAD: "Have you noticed any yellowing of your skin or eyes?" (not anchored)
3. Symptom detail — pick a DIFFERENT symptom from Q1 (e.g. if Q1 asked \
   about pain, Q3 must ask about weight loss, fatigue, or another symptom). \
   Ask for more detail (timing, triggers, severity). \
   GOOD: "The fatigue your GP noted — is it worse at certain times of day, \
   or does anything seem to make it better or worse?" \
   BAD: asking about the same symptom as Q1 (even with different wording)
4. Functional impact — tie a SPECIFIC symptom from the summary to daily life. \
   GOOD: "With the weight loss and the pain in your right side, how are you \
   managing with everyday things like cooking, walking, and self-care?" \
   BAD: "How are your symptoms affecting your daily activities?" (generic)
5. Patient concerns — ask what the patient is most worried about or what \
   questions they have ahead of their consultation. This is the only \
   question that does not need to reference a specific finding.

Return a JSON array of exactly 5 strings:
["question1", "question2", "question3", "question4", "question5"]

CRITICAL RULES:
- Questions 1-4 MUST anchor to a specific data point from the summary — \
  a named symptom (e.g. "the pain in your right side", "the weight loss"), \
  a named test (e.g. "your liver scan", "your blood tests"), or a general \
  reference to a finding (e.g. "some changes on your blood tests").
  If a question could be asked to ANY patient with the same condition, \
  it is too generic — rewrite it to reference THIS patient's data.
- SAFETY — NEVER quote exact measurements, lab values, tumour sizes, or \
  diagnostic terms (mass, lesion, carcinoma, malignancy) to the patient. \
  These are for the consultant to discuss. \
  GOOD: "Your GP arranged a scan of your liver — have you been told the results?" \
  BAD: "Your ultrasound showed a 4.2 cm mass in your liver" \
  GOOD: "Your blood tests showed some markers were a bit raised" \
  BAD: "Your AFP was 485 kU/L" \
  You may reference symptoms the patient already knows about (pain, weight \
  loss, fatigue, yellowing) but not test numbers or diagnostic findings.
- PLAIN ENGLISH ONLY — NEVER use medical terms, abbreviations, or jargon. \
  Write as if speaking to someone with no medical background. \
  BAD: "RUQ", "right upper quadrant", "scapula", "palpable", "hepatomegaly" \
  GOOD: "right side of your tummy", "shoulder blade", "tummy area" \
  Always translate: quadrant→side, scapula→shoulder blade, abdomen→tummy, \
  palpable→that your doctor felt, hepatomegaly→liver swelling, \
  radiating→spreading, bilateral→both sides, oedema→swelling
- NO DUPLICATE TOPICS: each question must be about a DIFFERENT symptom or \
  finding. If Q1 asks about pain, no other question may ask about pain \
  (even with different wording like "discomfort", "ache", "soreness"). \
  Spread questions across: pain, weight loss, fatigue, appetite, skin \
  changes, scan results, blood test results, etc.
- Each question must ask about ONE thing only — no compound questions
- Keep each question to 1-2 sentences maximum
- NEVER start with "How are you feeling" or "Tell me about your symptoms"
- Be warm and conversational — like a nurse who has READ the referral letter

Return ONLY the JSON array.\
"""

ADAPTIVE_QUESTIONS_PROMPT = """\
You are a {specialty} in a pre-consultation chat. Based on the \
patient's latest answer, generate 2-3 follow-up questions that cover NEW \
aspects of their condition.

CLINICAL SUMMARY (do NOT re-ask):
{clinical_summary}

Conversation so far:
{qa_history}

Patient's latest answer: "{latest_answer}"

Rules:
1. Each question MUST name a SPECIFIC finding from the clinical summary — \
   a symptom, a named test, or a general reference to a finding that has NOT \
   been discussed yet in the conversation above. \
   GOOD: "The weight loss your GP noted — has it continued, or has your \
   weight stabilised?" \
   GOOD: "Your blood tests showed some raised markers — have you been \
   feeling more tired or noticed any bruising since then?" \
   BAD: "Have you noticed any changes in your appetite?" (too generic) \
   BAD: "Have you had any abdominal swelling?" (not tied to a finding)
2. SAFETY — NEVER quote exact measurements, lab values, tumour sizes, or \
   diagnostic terms (mass, lesion, carcinoma) to the patient. \
   Say "the scan of your liver" not "the 4.2 cm hypoechoic lesion". \
   Say "your blood markers" not "AFP 485 kU/L". \
   You may reference symptoms the patient knows about (pain, weight loss, \
   fatigue) but not test numbers or diagnostic findings.
3. PLAIN ENGLISH — NEVER use medical terms, abbreviations, or jargon. \
   BAD: "RUQ", "right upper quadrant", "scapula", "palpable" \
   GOOD: "right side of your tummy", "shoulder blade" \
   Translate: quadrant→side, scapula→shoulder blade, abdomen→tummy
4. NEVER re-ask anything from the conversation history above
5. Cover new ground with each question — different symptoms, different findings
6. Be warm, empathetic, and conversational — like a nurse who has read the \
   referral and is checking in on specific things it mentioned

Return a JSON array of 2-3 question strings:
["question 1", "question 2"]

Return ONLY the JSON array.\
"""

FOLLOWUP_EVALUATION_PROMPT = """\
You are a {specialty} in a pre-consultation chat. The patient just \
answered a clinical question. Decide whether ONE brief follow-up is warranted \
before moving on to the next planned question.

CLINICAL SUMMARY (for context only):
{clinical_summary}

Question that was asked: "{plan_question}"
Patient's answer: "{patient_answer}"

A follow-up is warranted ONLY when the answer reveals:
- A clear change in symptoms (worsened, new onset, sudden change)
- An emergency-adjacent concern (confusion, vomiting blood, collapse)
- A new symptom not previously mentioned
- Significant functional impact (unable to eat, sleep, or work)
- Severe pain (7/10 or higher)

A follow-up is NOT warranted when:
- The patient gives a brief, stable answer ("no", "same as before", "fine")
- The answer is already detailed enough
- The topic has been sufficiently explored

If a follow-up IS warranted, return:
{{"followup": true, "question": "your single follow-up question here"}}

If NOT warranted, return:
{{"followup": false}}

Rules for the follow-up question:
1. Ask only ONE thing — no compound questions
2. Keep it to one sentence
3. NEVER quote exact lab values, measurements, or diagnostic terms
4. PLAIN ENGLISH only — no medical jargon or abbreviations
5. Be warm and conversational

Return ONLY valid JSON.\
"""

BRIDGE_RESPONSE_PROMPT = """\
You are a friendly clinical triage nurse chatting with a patient. The patient \
just answered a question and you need to briefly acknowledge their answer \
before asking the next question.

Patient said: "{patient_message}"
Next question to ask: "{next_question}"

Rules:
1. Write 1-2 sentences that acknowledge what they said naturally
2. Then transition smoothly to the next question
3. NEVER start with "Thank you for sharing" or "Thank you for that"
4. Sound like a caring, attentive nurse
5. If they said something emotional or off-topic, acknowledge it warmly
6. Keep the bridge SHORT — the question is the important part

Return ONLY the bridge text followed by the next question, separated by a blank line.\
"""

CLINICAL_WELCOME_PROMPT = """\
You are a friendly clinical triage nurse starting a patient's clinical \
assessment via chat. The patient has just completed registration.

Patient name: {patient_name}
Has referral letter: {has_referral}
Known condition: {condition}

Rules:
1. Welcome them warmly to the clinical assessment
2. If they have a referral, mention you've reviewed it
3. Explain you'll ask a few questions to prepare for their consultation
4. Sound natural and reassuring, 2-3 sentences
5. NEVER use bullet points or numbered lists

Return ONLY your message text.\
"""

EXTRACTION_PROMPT = """\
You are a clinical data extraction AI. Extract structured clinical information \
from the patient's response. Return a JSON object with these possible fields:

- chief_complaint: string (main reason for visit — ONLY set this if the patient \
  is describing why they are seeking care. Do NOT extract allergy reaction \
  symptoms like "rash", "swelling", "hives" as chief_complaint when the patient \
  is describing an allergic reaction to a substance.)
- medical_history: list of strings (conditions, surgeries, etc.)
- current_medications: list of strings
- allergies: list of strings (include the substance AND reaction type together, \
  e.g. "Penicillin - rash and swelling")
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

IMPORTANT: When a patient says something like "I'm allergic to penicillin, it \
causes rash and swelling", the rash and swelling are the ALLERGY REACTION, not \
a chief complaint. Extract "Penicillin - rash and swelling" under allergies only.

Patient message: "{message}"

Return ONLY valid JSON, no markdown, no explanation.\
"""

DOCUMENT_REQUESTS_PROMPT = """\
You are a clinical triage nurse preparing a patient's file for a consultant. \
Based on the clinical summary below, decide which 2-3 types of medical document \
would be most useful for the consultant to review before the appointment.

CLINICAL SUMMARY:
{clinical_summary}

Rules:
1. Pick exactly 2-3 document types that are RELEVANT to this specific patient's \
   condition, investigations, and referral reason
2. Even if the referral mentions a test result, still request it — the patient \
   may have a newer copy
3. Be specific: say "CT or MRI scan reports" not just "imaging"; say \
   "viral load test results" not just "blood tests"
4. Do NOT request referral letters, GP summaries, or documents the system \
   already has
5. Prioritise documents that would change the consultant's management plan

Return a JSON array of 2-3 short document type strings, e.g.:
["blood test results (including AFP/tumour markers)", "CT or MRI scan reports"]

Return ONLY the JSON array.\
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

        # Cross-phase follow-up: patient is answering a cross-phase question
        if event.payload.get("_cross_phase_followup"):
            return await self._handle_cross_phase_followup(event, diary)

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

        if event.event_type == EventType.CROSS_PHASE_DATA:
            return await self._handle_cross_phase_data(event, diary)

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

        # If intake already extracted clinical data from the referral PDF,
        # skip the redundant _analyze_referral() LLM call.
        has_referral = bool(diary.intake.referral_letter_ref)
        intake_pre_extracted = bool(diary.clinical.chief_complaint)

        if has_referral and not intake_pre_extracted:
            # Legacy path: intake didn't extract clinical data, do it now
            await self._analyze_referral(diary)
        elif intake_pre_extracted:
            logger.info(
                "Skipping _analyze_referral — intake already extracted clinical data "
                "(chief_complaint=%s) for patient %s",
                diary.clinical.chief_complaint, diary.header.patient_id,
            )

        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)

        # If referral already provided meds/allergies, mark them as addressed
        # so the clinical agent doesn't re-ask.
        if diary.clinical.current_medications:
            diary.clinical.meds_addressed = True
        if diary.clinical.allergies:
            diary.clinical.allergies_addressed = True

        # Simplified welcome — intake already sent the personalized hello,
        # so just transition smoothly to clinical questions.
        if intake_pre_extracted:
            welcome = (
                "I now have some clinical context from your referral. "
                "I'm going to ask a few questions to help us prepare "
                "for your consultation."
            )
        elif has_referral:
            welcome = (
                "Thank you for your details. I've reviewed your referral letter "
                "and have some context about your case. I'm going to ask a few "
                "questions now to help us prepare for your consultation."
            )
        else:
            welcome = (
                "Thank you for completing your registration. I'm going to ask "
                "a few questions to help us prepare for your consultation."
            )

        # Generate personalized question plan if we have enough context
        if diary.clinical.chief_complaint or diary.clinical.condition_context:
            await self._generate_initial_question_plan(diary)

        # Pick the first question: use the generated plan if available,
        # otherwise fall back to pattern-based.
        first_question = None
        if diary.clinical.generated_questions:
            first_question = diary.clinical.generated_questions.pop(0)
        if not first_question:
            first_question = self._fallback_question(diary)
        diary.clinical.questions_asked.append(ClinicalQuestion(question=first_question))
        diary.clinical.awaiting_followup = True
        message = f"{welcome}\n\n{first_question}"

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

        # Cross-phase suppression: if the gateway detected cross-phase content
        # targeting another agent (not clinical), suppress our response so the
        # cross-phase handler deals with it without a duplicate clinical reply.
        has_cross_phase = event.payload.get("_has_cross_phase_content", False)
        if has_cross_phase:
            xphase_targets = event.payload.get("_cross_phase_targets", [])
            if "clinical" not in xphase_targets:
                # Purely intake/other data — stay silent
                return AgentResult(updated_diary=diary)

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
        just_answered_q = None
        if unanswered and text:
            q = unanswered[0]
            q.answer = text
            q.answered_by = (
                event.sender_id
                if event.sender_role != SenderRole.PATIENT
                else "patient"
            )
            just_answered_q = q

        # ── Adaptive follow-up evaluation ──
        if just_answered_q and diary.clinical.awaiting_followup:
            if just_answered_q.is_followup:
                # Follow-up answer received — never chain another follow-up.
                # Clear state and fall through to next plan question.
                diary.clinical.awaiting_followup = False
            else:
                # Plan question answer received — evaluate if a follow-up is warranted
                diary.clinical.awaiting_followup = False
                followup_q = await self._evaluate_followup(
                    diary, just_answered_q.question, text,
                )
                if followup_q:
                    diary.clinical.questions_asked.append(
                        ClinicalQuestion(question=followup_q, is_followup=True)
                    )
                    response = AgentResponse(
                        recipient="patient",
                        channel=channel,
                        message=followup_q,
                        metadata={"patient_id": event.patient_id},
                    )
                    return AgentResult(updated_diary=diary, responses=[response])

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
            # "skip" or "no documents" skips ALL remaining document requests
            skip_all_keywords = {"skip", "no documents", "no more", "nothing else", "nothing"}
            # "no" / "don't have" skips just the current document, asks for next
            skip_one_keywords = {"no", "none", "don't have", "i don't"}
            if any(kw in text_lower for kw in skip_all_keywords):
                return await self._score_and_complete(event, diary, channel)
            if any(kw in text_lower for kw in skip_one_keywords):
                remaining = [
                    d for d in diary.clinical.pending_document_requests
                    if d not in diary.clinical.documents_requested
                ]
                if remaining:
                    return await self._prompt_for_documents(event, diary, channel)
                return await self._score_and_complete(event, diary, channel)

        # ── Detect "conversation concluded" signals during questioning ──
        #    If the patient signals they have nothing more to add and we have
        #    enough clinical data, fast-track to document collection / scoring.
        text_lower = text.lower().strip()
        conclude_phrases = {
            "nothing else to add", "nothing else", "that's all", "that's everything",
            "that is all", "that is everything", "no more to add", "nothing more",
            "no further information", "nothing to add", "i'm done", "im done",
        }
        if any(phrase in text_lower for phrase in conclude_phrases):
            if self._questions_sufficient(diary):
                logger.info(
                    "Patient signalled conclusion for %s — fast-tracking to documents/scoring",
                    event.patient_id,
                )
                if diary.clinical.sub_phase != ClinicalSubPhase.COLLECTING_DOCUMENTS:
                    diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
                    return await self._prompt_for_documents(event, diary, channel)
                if self._ready_for_scoring(diary):
                    return await self._score_and_complete(event, diary, channel)

        # ── Adaptive decision: enough questions? → collect docs → score ──

        # Step 1: If enough clinical questions answered, transition to doc collection
        if self._questions_sufficient(diary):
            if diary.clinical.sub_phase != ClinicalSubPhase.COLLECTING_DOCUMENTS:
                diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
                return await self._prompt_for_documents(event, diary, channel)

            # Step 2: If documents collected, proceed to scoring
            if self._ready_for_scoring(diary):
                return await self._score_and_complete(event, diary, channel)

            # Still collecting documents — prompt for next one
            return await self._prompt_for_documents(event, diary, channel)

        # Hard cap — force to document collection if too many questions asked
        if len(diary.clinical.questions_asked) >= MAX_CLINICAL_QUESTIONS:
            logger.info(
                "Question cap (%d) reached for patient %s — moving to document collection",
                MAX_CLINICAL_QUESTIONS, event.patient_id,
            )
            if diary.clinical.sub_phase != ClinicalSubPhase.COLLECTING_DOCUMENTS:
                diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
                return await self._prompt_for_documents(event, diary, channel)
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

        ack = "Thank you for uploading that document. I've added it to your file."

        if lab_values and self._ready_for_scoring(diary):
            return await self._score_and_complete(event, diary, channel)

        # After acknowledging, continue the conversation — either ask for
        # the next document or ask a clinical follow-up question, so the
        # patient doesn't have to send "ok" to move forward.
        next_part = ""
        if diary.clinical.sub_phase == ClinicalSubPhase.COLLECTING_DOCUMENTS:
            # Check if there are more documents to request
            for doc_type in diary.clinical.pending_document_requests:
                if doc_type not in diary.clinical.documents_requested:
                    diary.clinical.documents_requested.append(doc_type)
                    next_part = (
                        f"\n\nDo you have any recent {doc_type} to share? "
                        f"You can upload them now, or say 'no' if you don't have them."
                    )
                    break
            if not next_part and self._ready_for_scoring(diary):
                # All documents collected and ready — proceed to scoring
                return await self._score_and_complete(event, diary, channel)
            elif not next_part:
                # Not ready to score yet — go back to questions
                diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
                next_q = await self._generate_contextual_question(diary)
                diary.clinical.questions_asked.append(ClinicalQuestion(question=next_q))
                next_part = f"\n\n{next_q}"
        elif not self._ready_for_scoring(diary):
            # Still in questioning phase — ask the next question
            next_q = await self._generate_contextual_question(diary)
            diary.clinical.questions_asked.append(ClinicalQuestion(question=next_q))
            next_part = f"\n\n{next_q}"

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=ack + next_part,
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

                    # Emit CLINICAL_COMPLETE to trigger booking agent.
                    # No patient message here — monitoring already informed
                    # the patient that the appointment is being brought forward.
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
        """LLM-powered referral letter analysis to pre-populate clinical data.

        Skipped entirely if intake already extracted clinical data from the
        referral PDF (chief_complaint is populated).
        """
        # Skip if intake already did the extraction
        if diary.clinical.chief_complaint and diary.clinical.referral_analysis:
            logger.info(
                "Skipping _analyze_referral — already populated by intake for %s",
                diary.header.patient_id,
            )
            return

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

    def _build_clinical_summary(self, diary: PatientDiary) -> str:
        """Build a rich clinical summary from all diary data for LLM prompts.

        Primary path: use referral_narrative (preserves genotype, viral load,
        social history, referral intent — details lost in structured extraction).
        Appends any post-referral data gathered from Q&A (red flags, lifestyle, pain).

        Fallback: reconstruct from structured fields when narrative is absent.
        """
        # ── Primary path: referral narrative ──
        if diary.clinical.referral_narrative:
            parts = [diary.clinical.referral_narrative]

            # Append post-referral data gathered during Q&A
            if diary.clinical.red_flags:
                parts.append(f"Red flags identified: {', '.join(diary.clinical.red_flags)}")
            if diary.clinical.lifestyle_factors:
                lifestyle_strs = [f"{k}: {v}" for k, v in diary.clinical.lifestyle_factors.items()]
                parts.append(f"Lifestyle: {', '.join(lifestyle_strs)}")
            if diary.clinical.pain_level is not None:
                pain = f"Pain: {diary.clinical.pain_level}/10"
                if diary.clinical.pain_location:
                    pain += f" ({diary.clinical.pain_location})"
                parts.append(pain)

            return "\n".join(parts)

        # ── Fallback: structured field reconstruction ──
        parts = []

        if diary.clinical.condition_context:
            parts.append(f"Condition: {diary.clinical.condition_context}")
        if diary.clinical.chief_complaint:
            parts.append(f"Chief complaint: {diary.clinical.chief_complaint}")

        ref = diary.clinical.referral_analysis or {}

        # Symptoms — both from structured fields and referral
        symptoms = ref.get("symptoms", [])
        if isinstance(symptoms, list) and symptoms:
            parts.append(f"Symptoms: {', '.join(symptoms)}")

        # Key findings — this is the gold: mass sizes, scan results, etc.
        key_findings = ref.get("key_findings", "")
        if key_findings:
            parts.append(f"Key findings: {key_findings}")

        # Lab values with actual numbers
        lab_values = ref.get("lab_values", {})
        if isinstance(lab_values, dict) and lab_values:
            lab_strs = [f"{k}: {v}" for k, v in lab_values.items()]
            parts.append(f"Lab values: {', '.join(lab_strs)}")

        # Red flags
        if diary.clinical.red_flags:
            parts.append(f"Red flags: {', '.join(diary.clinical.red_flags)}")

        # Medical history
        if diary.clinical.medical_history:
            parts.append(f"Medical history: {', '.join(diary.clinical.medical_history)}")

        # Medications (context only, don't ask about)
        if diary.clinical.current_medications:
            parts.append(f"Medications: {', '.join(diary.clinical.current_medications)}")

        # Allergies (context only, don't ask about)
        if diary.clinical.allergies:
            parts.append(f"Allergies: {', '.join(diary.clinical.allergies)}")

        if not parts:
            return "No referral data available."

        return "\n".join(parts)

    @staticmethod
    def _derive_specialty(diary: PatientDiary) -> str:
        """Derive the clinical specialty from condition context / chief complaint.

        Used to make prompt templates specialty-agnostic instead of always
        saying "hepatology triage nurse".
        """
        context = " ".join(filter(None, [
            (diary.clinical.condition_context or "").lower(),
            (diary.clinical.chief_complaint or "").lower(),
        ]))

        if any(kw in context for kw in [
            "cancer", "carcinoma", "hcc", "tumour", "tumor",
            "malignancy", "malignant", "2ww", "2-week wait",
        ]):
            return "oncology triage nurse"
        if any(kw in context for kw in [
            "hepatitis", "hep b", "hep c", "hcv", "hbv",
            "cirrhosis", "fibrosis", "liver", "hepatic",
            "mash", "nafld", "nash", "masld", "fatty liver",
        ]):
            return "hepatology triage nurse"
        if any(kw in context for kw in [
            "ibs", "crohn", "colitis", "coeliac", "celiac",
            "bowel", "gastro", "gord", "reflux", "dyspepsia",
        ]):
            return "gastroenterology triage nurse"
        if any(kw in context for kw in [
            "pancrea", "gallstone", "cholecyst", "bile duct",
        ]):
            return "hepatobiliary triage nurse"

        return "clinical triage nurse"

    async def _generate_question_plan(self, diary: PatientDiary) -> None:
        """Generate top-5 ranked personalized questions using LLM."""
        return await self._generate_initial_question_plan(diary)

    async def _generate_initial_question_plan(self, diary: PatientDiary) -> None:
        """Generate top-5 ranked personalized questions using LLM (first-message fallback)."""
        try:
            if self.client is None:
                diary.clinical.generated_questions = self._fallback_question_plan(diary)
                return

            prompt = PERSONALIZED_QUESTIONS_PROMPT.format(
                clinical_summary=self._build_clinical_summary(diary),
                specialty=self._derive_specialty(diary),
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
        """Referral-aware fallback questions when LLM is unavailable.

        Reads actual diary data (lab values, symptoms, findings, red flags)
        from referral_analysis and builds questions that reference the patient's
        specific clinical picture. Does NOT ask about medications, allergies,
        or demographics.
        """
        ref = diary.clinical.referral_analysis or {}
        lab_values = ref.get("lab_values", {}) if isinstance(ref.get("lab_values"), dict) else {}
        key_findings = ref.get("key_findings", "") or ""
        red_flags = diary.clinical.red_flags or []
        symptoms = ref.get("symptoms", []) if isinstance(ref.get("symptoms"), list) else []
        chief = diary.clinical.chief_complaint or ""
        narrative = diary.clinical.referral_narrative or ""
        combined_text = f"{key_findings} {narrative}".lower()

        questions: list[str] = []

        # 1. Symptom progression — reference a specific symptom or finding
        if red_flags:
            flag = red_flags[0]
            questions.append(
                f"Your referral mentions {flag} — has this changed, "
                "improved, or worsened since you saw your GP?"
            )
        elif chief:
            questions.append(
                f"Since your GP visit regarding {chief}, "
                "have your symptoms changed — better, worse, or about the same?"
            )
        else:
            questions.append(
                "Could you tell me how your symptoms have been since your GP visit?"
            )

        # 2. New symptom screening — condition-specific warning signs
        condition = (diary.clinical.condition_context or "").lower()
        context = f"{condition} {(chief or '').lower()}"
        if any(kw in context for kw in ["cancer", "carcinoma", "hcc", "tumour", "tumor", "mass", "2ww"]):
            if "jaundice" not in " ".join(red_flags).lower():
                questions.append(
                    "Have you noticed any yellowing of your skin or the whites of "
                    "your eyes since your GP appointment?"
                )
            else:
                questions.append(
                    "Have you noticed any new swelling in your abdomen or legs "
                    "since your GP appointment?"
                )
        elif any(kw in context for kw in ["cirrhosis", "fibrosis", "decompensated"]):
            questions.append(
                "Have you experienced any confusion, drowsiness, or difficulty "
                "concentrating recently?"
            )
        elif any(kw in context for kw in ["hepatitis", "hep b", "hep c", "hcv", "hbv"]):
            questions.append(
                "Have you noticed any yellowing of your skin or eyes, "
                "or dark-coloured urine?"
            )
        else:
            questions.append(
                "Have you developed any new symptoms since your GP visit that "
                "you'd like to mention?"
            )

        # 3. Symptom detail — reference a specific lab value or finding
        if lab_values:
            lab_name = next(iter(lab_values))
            lab_val = lab_values[lab_name]
            questions.append(
                f"Your referral shows a {lab_name} level of {lab_val} — "
                "have you been experiencing any symptoms you think might be "
                "related to this?"
            )
        elif "weight loss" in combined_text or "weight" in " ".join(symptoms).lower():
            questions.append(
                "The weight loss mentioned in your referral — has it continued, "
                "or has your weight stabilised?"
            )
        elif any(kw in combined_text for kw in ["pain", "discomfort", "ache"]):
            questions.append(
                "Can you describe the pain or discomfort mentioned in your "
                "referral in more detail — has it changed in character, "
                "location, or severity?"
            )
        else:
            questions.append(
                "Are you experiencing any pain? If so, where and how severe "
                "on a scale of 0-10?"
            )

        # 4. Functional impact — reference specific condition
        if chief:
            questions.append(
                f"How is your {chief} affecting your daily life — things like "
                "walking, eating, sleeping, or working?"
            )
        else:
            questions.append(
                "How are your symptoms affecting your daily activities — "
                "work, exercise, social life?"
            )

        # 5. Patient concerns
        questions.append(
            "Is there anything you're particularly worried about or want to "
            "ask ahead of your consultation?"
        )

        return questions[:5]

    async def _regenerate_adaptive_questions(
        self, diary: PatientDiary, latest_answer: str
    ) -> None:
        """
        Regenerate questions adaptively after each patient answer.
        Uses the full Q&A history and latest answer to generate 2-3 targeted follow-ups.
        Capped at MAX_ADAPTIVE_REGENERATIONS cycles.
        """
        if diary.clinical.question_generation_count >= MAX_ADAPTIVE_REGENERATIONS:
            return

        diary.clinical.question_generation_count += 1

        # Build Q&A history string
        qa_pairs = []
        for q in diary.clinical.questions_asked:
            qa_pairs.append(f"Q: {q.question}")
            if q.answer:
                qa_pairs.append(f"A: {q.answer}")
        qa_history = "\n".join(qa_pairs) if qa_pairs else "No questions asked yet."

        try:
            if self.client is None:
                diary.clinical.generated_questions = self._fallback_question_plan(diary)
                return

            prompt = ADAPTIVE_QUESTIONS_PROMPT.format(
                clinical_summary=self._build_clinical_summary(diary),
                qa_history=qa_history,
                latest_answer=latest_answer[:500] if latest_answer else "initial assessment",
                specialty=self._derive_specialty(diary),
            )

            raw_response = await llm_generate(
                self.client, self._model_name, prompt,
            )
            if raw_response is None:
                if not diary.clinical.generated_questions:
                    diary.clinical.generated_questions = self._fallback_question_plan(diary)
                return

            raw = raw_response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            questions = json.loads(raw)
            if isinstance(questions, list) and questions:
                # Filter out questions semantically similar to already-asked ones
                asked_lower = {q.question.lower().strip() for q in diary.clinical.questions_asked}
                stop = {"the","a","an","is","are","do","you","your","have","any","and","or","to","of","in","for","how","what","about","with","been"}
                filtered = []
                for q_text in questions:
                    q_lower = q_text.lower().strip()
                    if q_lower in asked_lower:
                        continue
                    q_words = set(q_lower.split()) - stop
                    is_dup = False
                    for asked in asked_lower:
                        asked_words = set(asked.split()) - stop
                        if q_words and asked_words and len(q_words & asked_words) >= 3:
                            is_dup = True
                            break
                    if not is_dup:
                        filtered.append(q_text)
                diary.clinical.generated_questions = filtered[:3]
                logger.info(
                    "Adaptive regeneration #%d: %d questions for patient %s",
                    diary.clinical.question_generation_count,
                    len(diary.clinical.generated_questions),
                    diary.header.patient_id,
                )

        except Exception as exc:
            logger.warning("Adaptive question regeneration failed: %s — keeping existing", exc)
            if not diary.clinical.generated_questions:
                diary.clinical.generated_questions = self._fallback_question_plan(diary)

    # ── Cross-Phase Data Handler ──

    async def _handle_cross_phase_data(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Handle cross-phase clinical data detected by the Gateway."""
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")
        from_phase = event.payload.get("from_phase", "unknown")

        # Extract clinical data from the text
        extracted = await self._extract_clinical_data(text)
        if not extracted:
            # Nothing clinically relevant extracted — no response needed
            return AgentResult(updated_diary=diary)

        # Apply extracted data to diary
        self._apply_extracted_data(diary, extracted)

        # Build audit trail entry
        from datetime import datetime, timezone
        diary.cross_phase_extractions.append({
            "from_phase": from_phase,
            "to_agent": "clinical",
            "text_snippet": text[:100],
            "extracted_fields": list(extracted.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Build acknowledgment message
        ack_msg = self._build_cross_phase_ack(extracted)

        # Check if clinical follow-up is needed (e.g. allergy without reaction)
        follow_up = self._needs_clinical_follow_up(extracted, text)

        logger.info(
            "Cross-phase clinical extraction from %s for patient %s: %s (follow_up=%s)",
            from_phase, event.patient_id, list(extracted.keys()), follow_up is not None,
        )

        if follow_up:
            # Set cross-phase state — the gateway will route the patient's
            # next message back to this agent for the follow-up
            diary.cross_phase_state = CrossPhaseState(
                active=True,
                target_agent="clinical",
                pending_phase=from_phase,
                follow_up_question=follow_up,
                awaiting_response=True,
                original_text=text[:200],
                started=datetime.now(timezone.utc),
            )

            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=f"{ack_msg}\n\n{follow_up}",
                metadata={"patient_id": event.patient_id, "cross_phase": True},
            )
            # DO NOT emit reprompt yet — wait for the follow-up answer
            return AgentResult(updated_diary=diary, responses=[response])

        # No follow-up needed — ack and return control to the pending phase
        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=ack_msg,
            metadata={"patient_id": event.patient_id, "cross_phase": True},
        )

        reprompt = EventEnvelope.handoff(
            event_type=EventType.CROSS_PHASE_REPROMPT,
            patient_id=event.patient_id,
            source_agent="clinical",
            payload={"_pending_phase": from_phase, "channel": channel},
            correlation_id=event.correlation_id,
        )

        # DO NOT change the phase — patient stays in their current phase
        return AgentResult(
            updated_diary=diary, responses=[response], emitted_events=[reprompt],
        )

    @staticmethod
    def _clean_allergy_name(name: str) -> str:
        """Strip LLM artifacts like '- unknown reaction' from allergy names for patient-facing messages."""
        import re
        # Remove trailing " - unknown reaction", " - unknown", " - unspecified", etc.
        cleaned = re.sub(r'\s*-\s*(unknown|unspecified|no reaction|not specified)(\s+reaction)?\s*$', '', name, flags=re.IGNORECASE)
        return cleaned.strip() or name

    def _build_cross_phase_ack(self, extracted: dict[str, Any]) -> str:
        """Build acknowledgment message from extracted cross-phase clinical data."""
        ack_parts = []
        if "allergies" in extracted and extracted["allergies"]:
            allergy_names = [self._clean_allergy_name(a) for a in extracted["allergies"] if a and a != "NKDA"]
            if allergy_names:
                ack_parts.append(f"allergy to {', '.join(allergy_names)}")
        if "current_medications" in extracted and extracted["current_medications"]:
            ack_parts.append(f"medication update ({', '.join(extracted['current_medications'][:3])})")
        if "red_flags" in extracted:
            ack_parts.append("important clinical information")
        if "chief_complaint" in extracted:
            ack_parts.append("your symptoms")

        if ack_parts:
            return f"I've noted your {' and '.join(ack_parts)} and updated your clinical record."
        return "I've noted that information and updated your clinical record."

    def _merge_allergy_reaction(self, diary: PatientDiary, reaction_text: str) -> None:
        """Merge an allergy reaction description into the most recent allergen entry.

        When a patient says "I'm allergic to penicillin" followed by "it causes a rash",
        this updates the entry from "Penicillin" to "Penicillin (rash)".
        """
        # Extract the reaction keyword(s) from the patient's text
        reaction_keywords = [
            "rash", "hives", "swelling", "anaphylaxis", "breathing difficulties",
            "itching", "vomiting", "nausea", "throat swelling", "tongue swelling",
            "stomach pain", "diarrhoea", "dizziness", "fainting",
        ]
        text_lower = reaction_text.lower()
        reactions = [kw for kw in reaction_keywords if kw in text_lower]

        if not reactions:
            # Fallback: use a cleaned version of whatever they said
            clean = reaction_text.strip().rstrip(".!,").strip()
            if clean and len(clean) < 80:
                reactions = [clean]

        if not reactions or not diary.clinical.allergies:
            return

        reaction_str = ", ".join(reactions)

        # Find the most recently added allergen that doesn't already have a reaction
        for i in range(len(diary.clinical.allergies) - 1, -1, -1):
            entry = diary.clinical.allergies[i]
            if entry and entry != "NKDA" and "(" not in entry:
                diary.clinical.allergies[i] = f"{entry} ({reaction_str})"
                logger.info(
                    "Merged allergy reaction: %s → %s",
                    entry, diary.clinical.allergies[i],
                )
                return

    def _needs_clinical_follow_up(
        self, extracted: dict[str, Any], text: str
    ) -> str | None:
        """
        Check if extracted cross-phase data needs a follow-up question.
        Returns the follow-up question string, or None if no follow-up needed.
        """
        # Allergy without reaction type — ask for reaction details
        if "allergies" in extracted and extracted["allergies"]:
            allergy_names = [a for a in extracted["allergies"] if a and a != "NKDA"]
            if allergy_names:
                text_lower = text.lower()
                reaction_keywords = [
                    "rash", "hives", "swelling", "anaphyla", "breathing",
                    "itching", "vomit", "nausea", "throat", "tongue",
                    "stomach", "diarr", "dizz", "faint",
                ]
                has_reaction = any(kw in text_lower for kw in reaction_keywords)
                if not has_reaction:
                    allergy_str = self._clean_allergy_name(allergy_names[0])
                    return (
                        f"Could you tell me what kind of reaction you have to "
                        f"{allergy_str}? For example, does it cause a rash, "
                        f"swelling, breathing difficulties, or something else?"
                    )
        return None

    async def _handle_cross_phase_followup(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Handle the patient's response to a cross-phase follow-up question."""
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")
        pending_phase = event.payload.get(
            "_pending_phase", diary.cross_phase_state.pending_phase
        )

        # Check if this was an allergy-reaction follow-up
        follow_up_q = diary.cross_phase_state.follow_up_question or ""
        is_allergy_followup = "reaction" in follow_up_q.lower() and "allerg" in (
            diary.cross_phase_state.original_text or ""
        ).lower()

        if is_allergy_followup:
            # Merge the reaction into the existing allergen entry
            self._merge_allergy_reaction(diary, text)
        else:
            # Generic extraction for other follow-ups
            extracted = await self._extract_clinical_data(text)
            if extracted:
                self._apply_extracted_data(diary, extracted)

        # Clear the cross-phase state
        diary.cross_phase_state = CrossPhaseState()

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message="Thank you, I've added that to your clinical record.",
            metadata={"patient_id": event.patient_id, "cross_phase": True},
        )

        # Return control to the pending phase
        reprompt = EventEnvelope.handoff(
            event_type=EventType.CROSS_PHASE_REPROMPT,
            patient_id=event.patient_id,
            source_agent="clinical",
            payload={"_pending_phase": pending_phase, "channel": channel},
            correlation_id=event.correlation_id,
        )

        logger.info(
            "Cross-phase follow-up complete for patient %s — returning to %s",
            event.patient_id, pending_phase,
        )

        return AgentResult(
            updated_diary=diary, responses=[response], emitted_events=[reprompt],
        )

    # ── Natural Language Generation ──

    async def _generate_clinical_welcome(
        self, diary: PatientDiary, has_referral: bool
    ) -> str:
        """Generate a natural clinical welcome message via LLM. Returns empty string on failure."""
        try:
            if self.client is not None:
                prompt = CLINICAL_WELCOME_PROMPT.format(
                    patient_name=diary.intake.name or "there",
                    has_referral="yes" if has_referral else "no",
                    condition=diary.clinical.condition_context or "not yet identified",
                )
                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw:
                    return raw.strip()
        except Exception as exc:
            logger.warning("LLM clinical welcome failed: %s — using fallback", exc)
        return ""

    async def _bridge_to_question(
        self, patient_msg: str, next_question: str, diary: PatientDiary
    ) -> str:
        """
        LLM-generate a natural bridge between patient's answer and the next question.
        Falls back to just the bare question (identical to current behavior).
        """
        try:
            if self.client is not None and patient_msg.strip():
                prompt = BRIDGE_RESPONSE_PROMPT.format(
                    patient_message=patient_msg[:500],
                    next_question=next_question,
                )
                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw and is_response_complete(raw.strip()):
                    return raw.strip()
        except Exception as exc:
            logger.warning("LLM bridge generation failed: %s — using bare question", exc)
        return next_question

    # ── Adaptive Follow-Up Evaluation ──

    async def _evaluate_followup(
        self, diary: PatientDiary, question: str, answer: str
    ) -> str | None:
        """Evaluate whether a single follow-up question is warranted after a plan answer.

        Returns the follow-up question string, or None if no follow-up needed.
        Tries LLM first, then deterministic keyword fallback.
        """
        # Fast-path: trivial/short answers don't warrant follow-up
        stripped = answer.strip().lower().rstrip(".!,")
        if stripped in ("no", "yes", "same", "fine", "ok", "okay", "good", "nope", "yeah"):
            return None
        if len(stripped) < 5:
            return None

        # Try LLM evaluation
        try:
            if self.client is not None:
                prompt = FOLLOWUP_EVALUATION_PROMPT.format(
                    clinical_summary=self._build_clinical_summary(diary),
                    plan_question=question[:500],
                    patient_answer=answer[:500],
                    specialty=self._derive_specialty(diary),
                )
                raw_response = await llm_generate(
                    self.client, self._model_name, prompt,
                )
                if raw_response:
                    raw = raw_response.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()
                    result = json.loads(raw)
                    if isinstance(result, dict) and result.get("followup"):
                        followup_q = result.get("question", "").strip()
                        if followup_q:
                            return followup_q
                    return None
        except Exception as exc:
            logger.warning("LLM follow-up evaluation failed: %s — trying deterministic", exc)

        # Deterministic fallback
        return self._deterministic_followup(answer, question)

    @staticmethod
    def _deterministic_followup(answer: str, question: str) -> str | None:
        """Keyword-based follow-up evaluation when LLM is unavailable.

        Returns a follow-up question or None.
        """
        lower = answer.lower()

        # Worsening keywords
        if any(kw in lower for kw in ("worse", "worsened", "deteriorat", "getting worse", "gone downhill")):
            return "When did you first notice this change, and has it been gradual or quite sudden?"

        # Emergency-adjacent signals
        if any(kw in lower for kw in ("confused", "vomiting blood", "collapsed", "passed out", "blacked out", "fitting")):
            return "How recently did this happen, and has it occurred more than once?"

        # New symptom signals (only if answer is substantial)
        if len(lower.split()) > 15 and any(kw in lower for kw in ("started", "new", "developed", "just begun", "noticed")):
            return "How often does this happen — is it constant, or does it come and go?"

        # Functional impact
        if any(kw in lower for kw in ("can't sleep", "can't eat", "unable to", "cannot walk", "can't work", "stopped eating")):
            return "Has this come on gradually, or was there a particular moment when it started?"

        # Severe pain (look for numbers 7-10 out of 10)
        import re
        pain_match = re.search(r'\b([7-9]|10)\s*(?:out of|/)\s*10\b', lower)
        if pain_match:
            return "Is the pain constant, or does it come and go throughout the day?"

        return None

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
        asked_lower = {q.question.lower().strip() for q in diary.clinical.questions_asked}
        _stop = {"the","a","an","is","are","do","you","your","have","any","and","or","to","of","in","for","how","what","about","with","been"}

        # Clinical topic groups — synonyms that count as the same topic
        _topic_groups = [
            {"pain", "ache", "discomfort", "soreness", "hurting", "tender"},
            {"weight", "weight loss", "lost weight", "appetite", "eating"},
            {"fatigue", "tired", "tiredness", "exhausted", "energy"},
            {"yellowing", "jaundice", "yellow", "skin colour", "tinge"},
            {"swelling", "swollen", "bloating", "fluid", "abdomen size"},
            {"scan", "ultrasound", "ct", "mri", "imaging"},
            {"blood test", "blood markers", "blood results", "lab"},
        ]

        def _extract_topics(text: str) -> set[str]:
            """Extract which clinical topic groups a question touches."""
            t = text.lower()
            topics = set()
            for i, group in enumerate(_topic_groups):
                if any(kw in t for kw in group):
                    topics.add(i)
            return topics

        asked_topics = set()
        for q in diary.clinical.questions_asked:
            asked_topics |= _extract_topics(q.question)

        def _is_duplicate(candidate: str) -> bool:
            c_lower = candidate.lower().strip()
            if c_lower in asked_lower:
                return True
            # Word overlap check
            c_words = set(c_lower.split()) - _stop
            for asked in asked_lower:
                a_words = set(asked.split()) - _stop
                if c_words and a_words and len(c_words & a_words) / min(len(c_words), len(a_words)) > 0.5:
                    return True
            # Clinical topic overlap — same symptom area already covered
            c_topics = _extract_topics(candidate)
            if c_topics and c_topics.issubset(asked_topics):
                return True
            return False

        # ── Deterministic override: ensure meds/allergies are asked ──
        # LLM prompts say "MUST ask" but don't always comply. Force it
        # after the first answer so we don't score without this data.
        meds_ok = diary.clinical.meds_addressed or bool(diary.clinical.current_medications)
        allergy_ok = diary.clinical.allergies_addressed or bool(diary.clinical.allergies)
        has_answered = sum(1 for q in diary.clinical.questions_asked if q.answer is not None)

        if has_answered >= 1 and not meds_ok:
            meds_q = (
                "Before we go any further, are you currently taking any medications, "
                "including over-the-counter medications, herbal remedies, or supplements?"
            )
            if not _is_duplicate(meds_q):
                diary.clinical.questions_asked.append(ClinicalQuestion(question=meds_q))
                return AgentResult(updated_diary=diary, responses=[
                    AgentResponse(recipient="patient", channel=channel, message=meds_q,
                                  metadata={"patient_id": event.patient_id}),
                ])

        if has_answered >= 1 and not allergy_ok:
            allergy_q = (
                "Do you have any allergies to medications, food, or anything else?"
            )
            if not _is_duplicate(allergy_q):
                diary.clinical.questions_asked.append(ClinicalQuestion(question=allergy_q))
                return AgentResult(updated_diary=diary, responses=[
                    AgentResponse(recipient="patient", channel=channel, message=allergy_q,
                                  metadata={"patient_id": event.patient_id}),
                ])

        # ── Use pre-generated question plan first (specific to referral) ──
        question_text = None
        while diary.clinical.generated_questions:
            candidate = diary.clinical.generated_questions.pop(0)
            if not _is_duplicate(candidate):
                question_text = candidate
                break

        # ── Fallback: generate contextual question via LLM ──
        if not question_text:
            question_text = await self._generate_contextual_question(diary)

        # Record the question
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question=question_text)
        )

        # Mark that we're awaiting a follow-up evaluation after the patient answers
        diary.clinical.awaiting_followup = True

        # Simple acknowledgment + question (no LLM bridge — saves ~15s)
        message_text = question_text

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message_text,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    def _identify_gaps(self, diary: PatientDiary) -> list[str]:
        """Identify what clinical information is still missing.

        Respects data pre-populated from referral — only flags genuinely
        unknown information.
        """
        # Build a corpus of all previous answers for keyword scanning
        # so we don't re-ask about topics the patient already mentioned.
        answer_corpus = " ".join(
            (q.answer or "").lower() for q in diary.clinical.questions_asked
        )

        gaps = []
        if not diary.clinical.chief_complaint:
            gaps.append("chief_complaint")
        if not diary.clinical.medical_history:
            gaps.append("medical_history")
        if not diary.clinical.meds_addressed and not diary.clinical.current_medications:
            gaps.append("current_medications")
        if not diary.clinical.allergies_addressed and not diary.clinical.allergies:
            gaps.append("allergies")

        # Symptom timeline — almost never in referral, always worth asking
        timeline_mentioned = any(
            kw in answer_corpus
            for kw in ["started", "began", "worse", "better", "getting", "weeks ago", "months ago"]
        )
        if not timeline_mentioned and diary.clinical.chief_complaint:
            gaps.append("symptom_timeline")

        # Condition-specific gaps — skip if patient already discussed the topic
        condition = (diary.clinical.condition_context or "").lower()
        if any(kw in condition for kw in ["cirrhosis", "liver", "hepat"]):
            alcohol_mentioned = (
                "alcohol" in diary.clinical.lifestyle_factors
                or any(kw in answer_corpus for kw in ["drink", "alcohol", "pint", "beer", "wine", "spirit", "unit"])
            )
            if not alcohol_mentioned:
                gaps.append("lifestyle_alcohol")
        if any(kw in condition for kw in ["mash", "nafld", "nash", "fatty"]):
            weight_mentioned = (
                "weight" in diary.clinical.lifestyle_factors
                or any(kw in answer_corpus for kw in ["weight", "bmi", "diet", "eating", "stone", "kg"])
            )
            if not weight_mentioned:
                gaps.append("lifestyle_weight")
        return gaps

    async def _generate_contextual_question(self, diary: PatientDiary) -> str:
        """Generate the most important next question based on current clinical picture."""
        try:
            if self.client is None:
                return self._fallback_question(diary)

            # Build Q&A history string so LLM sees what's been asked AND answered
            qa_lines = []
            for q in diary.clinical.questions_asked:
                qa_lines.append(f"Q: {q.question}")
                if q.answer:
                    qa_lines.append(f"A: {q.answer}")
            qa_history = "\n".join(qa_lines) if qa_lines else "No questions asked yet."

            prompt = QUESTION_GENERATION_PROMPT.format(
                clinical_summary=self._build_clinical_summary(diary),
                qa_history=qa_history,
                specialty=self._derive_specialty(diary),
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
            "symptom_timeline": (
                "When did your symptoms first start, and have they been "
                "getting better, worse, or staying about the same?"
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
                "How are your symptoms affecting your daily life — work, sleep, appetite?",
            )

        return "How are your symptoms affecting your daily life — work, sleep, appetite?"

    async def _prompt_for_documents(
        self, event: EventEnvelope, diary: PatientDiary, channel: str
    ) -> AgentResult:
        """Sequential document collection — ask for ONE document at a time."""
        # Generate document request list if not already done
        if not diary.clinical.pending_document_requests:
            diary.clinical.pending_document_requests = await self._generate_document_requests(diary)

        # Find the next document to ask about
        for doc_type in diary.clinical.pending_document_requests:
            if doc_type not in diary.clinical.documents_requested:
                diary.clinical.documents_requested.append(doc_type)
                # First document prompt includes an NHS App tip + the ask as two messages
                is_first = len(diary.clinical.documents_requested) == 1
                if is_first:
                    nhs_tip = (
                        "Before we wrap up, I'd like to check if you have any "
                        "recent medical documents that could help your consultant. "
                        "If you're not sure where to find your test results, you can "
                        "view them in the NHS App under GP health record > Test results. "
                        "Most recent blood and urine tests ordered by your GP will be there."
                    )
                    ask_msg = (
                        f"Do you have any recent {doc_type} to share? "
                        f"You can upload them here, or just say 'no' if you don't have them."
                    )
                    return AgentResult(updated_diary=diary, responses=[
                        AgentResponse(
                            recipient="patient",
                            channel=channel,
                            message=nhs_tip,
                            metadata={"patient_id": event.patient_id},
                        ),
                        AgentResponse(
                            recipient="patient",
                            channel=channel,
                            message=ask_msg,
                            metadata={"patient_id": event.patient_id},
                        ),
                    ])
                else:
                    msg = (
                        f"Do you have any recent {doc_type} to share? "
                        f"You can upload them now, or say 'no' if you don't have them."
                    )
                    return AgentResult(updated_diary=diary, responses=[
                        AgentResponse(
                            recipient="patient",
                            channel=channel,
                            message=msg,
                            metadata={"patient_id": event.patient_id},
                        ),
                    ])

        # All documents asked — return empty result (caller handles scoring)
        return AgentResult(updated_diary=diary)

    async def _generate_document_requests(self, diary: PatientDiary) -> list[str]:
        """Generate 2-3 relevant document requests using LLM with deterministic fallback."""
        # ── Try LLM path ──
        try:
            if self.client is not None:
                prompt = DOCUMENT_REQUESTS_PROMPT.format(
                    clinical_summary=self._build_clinical_summary(diary),
                )
                raw_response = await llm_generate(
                    self.client, self._model_name, prompt,
                )
                if raw_response:
                    raw = raw_response.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()
                    docs = json.loads(raw)
                    if isinstance(docs, list) and 2 <= len(docs) <= 3:
                        logger.info(
                            "LLM generated %d document requests for %s",
                            len(docs), diary.header.patient_id,
                        )
                        return docs
        except Exception as exc:
            logger.warning("LLM document request generation failed: %s — using fallback", exc)

        # ── Deterministic fallback using referral_analysis ──
        return self._fallback_document_requests(diary)

    def _fallback_document_requests(self, diary: PatientDiary) -> list[str]:
        """Deterministic document requests derived from referral_analysis data."""
        requests: list[str] = []
        ref = diary.clinical.referral_analysis or {}
        condition = (diary.clinical.condition_context or "").lower()
        chief = (diary.clinical.chief_complaint or "").lower()
        context = f"{condition} {chief}"

        # Lab values present → request blood test results
        lab_values = ref.get("lab_values", {})
        if isinstance(lab_values, dict) and lab_values:
            requests.append("blood test results")

        # Imaging / scan mentions in key_findings or narrative
        findings = (ref.get("key_findings", "") or "").lower()
        narrative = (diary.clinical.referral_narrative or "").lower()
        combined_text = f"{findings} {narrative}"
        if any(kw in combined_text for kw in ["ultrasound", "imaging", "scan", "ct", "mri"]):
            requests.append("CT or MRI scan reports")

        # Condition-specific additions
        is_cancer = any(kw in context for kw in [
            "cancer", "carcinoma", "hcc", "tumour", "tumor",
            "mass", "lesion", "malignancy", "2ww", "2-week wait",
        ])
        if is_cancer and "blood test results" not in requests:
            requests.append("blood test results (including AFP/tumour markers)")
        if is_cancer and "CT or MRI scan reports" not in requests:
            requests.append("CT or MRI scan reports")

        if any(kw in context for kw in ["hepatitis", "hep b", "hep c", "hcv", "hbv"]):
            if "viral load test results" not in requests:
                requests.append("viral load test results")

        if any(kw in context for kw in ["cirrhosis", "fibrosis", "decompensated"]):
            if "fibroscan results" not in requests:
                requests.append("fibroscan results")

        if any(kw in context for kw in ["mash", "nafld", "nash", "fatty", "masld"]):
            if "blood test results" not in requests:
                requests.append("blood test results")

        # Fallback if nothing matched
        if not requests:
            requests.append("recent blood test results")

        return requests[:3]

    # ── Scoring & Completion ──

    async def _score_and_complete(
        self, event: EventEnvelope, diary: PatientDiary, channel: str
    ) -> AgentResult:
        """Run risk scoring and complete clinical assessment."""
        diary.clinical.advance_sub_phase(ClinicalSubPhase.SCORING_RISK)

        # Collect lab values from ALL sources:
        # 1. Uploaded documents (lab results, GP responses)
        # 2. Referral letter extraction (stored in referral_analysis)
        lab_values: dict[str, Any] = {}

        # Referral lab values (extracted by intake agent from the PDF)
        referral_labs = diary.clinical.referral_analysis.get("lab_values", {})
        if isinstance(referral_labs, dict):
            lab_values.update(referral_labs)

        # Uploaded document lab values (override referral if newer)
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

    def _questions_sufficient(self, diary: PatientDiary) -> bool:
        """Check if we have enough clinical question data to move on.

        This gates the transition from ASKING_QUESTIONS to COLLECTING_DOCUMENTS.
        After this returns True, document collection happens before scoring.
        """
        has_complaint = diary.clinical.chief_complaint is not None
        has_answered = sum(
            1 for q in diary.clinical.questions_asked if q.answer is not None
        )
        has_labs = any(
            doc.extracted_values for doc in diary.clinical.documents
        )

        # Lab data alone is sufficient
        if has_labs:
            return True

        meds_ok = diary.clinical.meds_addressed or bool(diary.clinical.current_medications)
        allergy_ok = diary.clinical.allergies_addressed or bool(diary.clinical.allergies)

        _safety_qs = {"currently taking any medications", "allergies to medications"}
        clinical_answered = sum(
            1 for q in diary.clinical.questions_asked
            if q.answer is not None
            and not any(kw in q.question.lower() for kw in _safety_qs)
        )

        # Referral-first path — need 4-5 questions for a thorough pre-consultation
        has_referral_data = bool(diary.clinical.referral_analysis)
        if has_referral_data:
            # Don't transition while plan questions remain queued
            if diary.clinical.generated_questions:
                return False
            # Don't transition mid follow-up evaluation
            if diary.clinical.awaiting_followup:
                return False
            if has_complaint and clinical_answered >= 4 and meds_ok and allergy_ok:
                return True
            if has_complaint and clinical_answered >= 5:
                return True

        # Legacy path
        if has_complaint and clinical_answered >= 5 and meds_ok and allergy_ok:
            return True
        if has_complaint and clinical_answered >= 7 and (meds_ok or allergy_ok):
            return True
        if has_complaint and clinical_answered >= 8:
            return True

        # Absolute safety net
        if has_answered >= 10:
            return True

        return False

    def _ready_for_scoring(self, diary: PatientDiary) -> bool:
        """Check if we have enough data AND documents have been collected.

        This gates the transition from COLLECTING_DOCUMENTS to SCORING_RISK.
        Only returns True once document collection is complete.
        """
        if not self._questions_sufficient(diary):
            return False

        # Must have completed document collection before scoring
        if diary.clinical.sub_phase == ClinicalSubPhase.COLLECTING_DOCUMENTS:
            # Done when all pending requests have been asked
            if diary.clinical.pending_document_requests:
                return all(
                    d in diary.clinical.documents_requested
                    for d in diary.clinical.pending_document_requests
                )
            # No pending requests yet — _prompt_for_documents will generate them
            return False

        # If we're past document collection (or came from a different path
        # like GP_RESPONSE / DETERIORATION), allow scoring
        if diary.clinical.sub_phase in (
            ClinicalSubPhase.SCORING_RISK, ClinicalSubPhase.COMPLETE,
        ):
            return True

        # Still in ASKING_QUESTIONS — not ready yet (need docs first)
        return False

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
            # Only set chief complaint if not already established — once set,
            # it should only be changed by explicit referral analysis, not by
            # incidental mentions of symptoms in allergy/medication context.
            if not diary.clinical.chief_complaint:
                diary.clinical.chief_complaint = extracted["chief_complaint"]

        if "medical_history" in extracted:
            for item in extracted["medical_history"]:
                if item and item not in diary.clinical.medical_history:
                    diary.clinical.medical_history.append(item)

        if "current_medications" in extracted:
            diary.clinical.meds_addressed = True
            for med in extracted["current_medications"]:
                if med and med not in diary.clinical.current_medications:
                    diary.clinical.current_medications.append(med)

        if "allergies" in extracted:
            diary.clinical.allergies_addressed = True
            new_allergies = [a for a in extracted["allergies"] if a]
            # Normalize common "no allergy" variants to NKDA
            _nkda_variants = {
                "no known allergies", "no allergies", "none known",
                "no known drug allergies", "nil known",
            }
            normalized = []
            for a in new_allergies:
                if a.strip().lower() in _nkda_variants:
                    normalized.append("NKDA")
                else:
                    normalized.append(self._clean_allergy_name(a))
            new_allergies = normalized

            has_specific = any(a != "NKDA" for a in new_allergies)
            # If patient now reports a specific allergy, clear NKDA / negation placeholders
            if has_specific:
                diary.clinical.allergies = [
                    a for a in diary.clinical.allergies
                    if a != "NKDA" and a.strip().lower() not in _nkda_variants
                ]
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
            "HCC": ["hcc", "hepatocellular carcinoma", "liver cancer", "liver mass", "liver lesion", "2ww", "2-week wait"],
            "cancer": ["cancer", "carcinoma", "tumour", "tumor", "malignancy", "malignant"],
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
        """Extract clinical data from free text using LLM with fallback.

        Always runs fallback extraction to fill gaps the LLM may have missed.
        Fallback results are used only for keys the LLM didn't return.
        """
        if not text:
            return {}

        llm_result: dict[str, Any] = {}
        try:
            if self.client is not None:
                prompt = EXTRACTION_PROMPT.format(message=text)
                raw_response = await llm_generate(self.client, self._model_name, prompt)
                if raw_response is not None:
                    raw = raw_response.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()
                    llm_result = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM clinical extraction failed: %s — merging with fallback", exc)

        # Always run fallback extraction to fill gaps the LLM missed
        fallback_result = self._fallback_extraction(text)

        # Merge: LLM takes priority, fallback fills missing keys
        merged = {**fallback_result, **{k: v for k, v in llm_result.items() if v}}
        return merged

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

        # Chief complaint — but NOT if the text is primarily about allergies
        # (e.g. "I'm allergic to penicillin, it causes rash and swelling"
        #  should not set chief complaint to "rash and swelling")
        is_allergy_context = any(
            kw in text_lower for kw in ["allerg", "allergic to", "reaction to"]
        )
        if not is_allergy_context:
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
        # Match correction phrasing first: "more like N", "actually N"
        pain_match = re.search(
            r'(?:more like|actually|it\'?s|now)\s+(?:a\s+)?(\d+)\s*(?:/\s*10|out of 10)',
            text_lower,
        )
        # Then "pain/level/scale ... N" or bare "N out of 10" / "N/10"
        if not pain_match:
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
        allergy_phrases = ["allerg", "allergic to", "no known allerg", "nkda", "no allerg"]
        if any(phrase in text_lower for phrase in allergy_phrases):
            if "no known" in text_lower or "nkda" in text_lower or "no allerg" in text_lower:
                extracted["allergies"] = ["NKDA"]
            else:
                # Try "allergic to <substance>, causes/gives <reaction>" pattern first
                full_match = re.search(
                    r"allerg(?:ic)?\s+to\s+(\w[\w\s]*?)"
                    r"(?:\s*[,.\-–—]\s*|\s+)"
                    r"(?:(?:which|it|that)\s+)?"
                    r"(?:causes?|gives?\s+(?:me)?|results?\s+in)\s+"
                    r"(.+?)(?:\s*[.!]?\s*$)",
                    text, re.IGNORECASE,
                )
                if full_match:
                    substance = full_match.group(1).strip()
                    reaction = full_match.group(2).strip().rstrip(".")
                    extracted["allergies"] = [f"{substance} ({reaction})"]
                else:
                    # Fall back to substance-only extraction
                    allerg_match = re.search(
                        r"allerg(?:ic)?\s+to\s+(\w[\w\s]*?)(?:\s*[,.\-–—]|\s+(?:which|it|that|causes?|gives?)|\s*$)",
                        text, re.IGNORECASE,
                    )
                    if allerg_match:
                        substance = allerg_match.group(1).strip()
                        extracted["allergies"] = [substance]
                    else:
                        idx = text_lower.find("allerg")
                        if idx >= 0:
                            extracted["allergies"] = [text[idx:idx+100].strip()]

        # Medications
        med_keywords = ["take", "taking", "prescribed", "medication", "medicine", "mg", "daily",
                        "no meds", "no med", "not on any"]
        no_med_phrases = ["no meds", "no med", "no medication", "not taking any",
                          "not on any", "don't take any", "dont take any"]
        if any(kw in text_lower for kw in med_keywords):
            if any(neg in text_lower for neg in no_med_phrases):
                # Patient explicitly says no medications — mark as addressed
                extracted["current_medications"] = []
            else:
                # Simple extraction: capture medication-like patterns
                med_matches = re.findall(r'(\w+\s+\d+\s*mg)', text, re.IGNORECASE)
                if med_matches:
                    extracted["current_medications"] = med_matches

        # Medical history
        history_phrases = ["diagnosed", "surgery", "operation", "condition", "disease"]
        if any(phrase in text_lower for phrase in history_phrases):
            extracted["medical_history"] = [text[:200]]

        return extracted

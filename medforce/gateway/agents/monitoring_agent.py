"""
Monitoring Agent — The Guardian.

Cares for patients post-booking through an adaptive communication plan:
  1. On BOOKING_COMPLETE: Generate risk-stratified communication plan
     - Determine message frequency based on risk level
     - Generate personalized questions based on condition + treatment
     - Rank questions by clinical importance, select top N
     - Schedule question delivery across the monitoring period
  2. On HEARTBEAT: Execute scheduled check-ins from the plan
  3. On USER_MESSAGE: Reactive risk-aware responses
  4. On DOCUMENT_UPLOADED: Compare new labs against baseline

Can emit:
  - DETERIORATION_ALERT → Clinical Agent reassessment

The monitoring plan is a loop, not a straight line:
  Each heartbeat re-evaluates the patient state and can adjust
  the remaining plan (bring forward questions, increase frequency).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import (
    CommunicationPlan,
    DeteriorationAssessment,
    DeteriorationQuestion,
    MonitoringEntry,
    PatientDiary,
    Phase,
    RiskLevel,
    ScheduledQuestion,
)
from medforce.gateway.agents.llm_utils import is_response_complete, llm_generate
from medforce.gateway.events import EventEnvelope, EventType

logger = logging.getLogger("gateway.agents.monitoring")

# Stalled assessment timeout (P0) — hours before force-completing an assessment
ASSESSMENT_TIMEOUT_HOURS = 48

# Phase staleness thresholds (P0) — hours a patient can stay in a phase
# before the heartbeat flags it for recovery
PHASE_STALE_THRESHOLDS: dict[str, int] = {
    "intake": 72,       # 3 days
    "clinical": 72,     # 3 days
    "booking": 48,      # 2 days
    # monitoring has no timeout — it's long-lived by design
}

# Risk → number of scheduled check-ins and check-in interval (days)
RISK_SCHEDULES: dict[str, dict[str, Any]] = {
    RiskLevel.CRITICAL.value: {"total_messages": 8, "check_days": [3, 7, 10, 14, 21, 30, 45, 60]},
    RiskLevel.HIGH.value: {"total_messages": 6, "check_days": [7, 14, 21, 30, 45, 60]},
    RiskLevel.MEDIUM.value: {"total_messages": 4, "check_days": [14, 30, 60, 90]},
    RiskLevel.LOW.value: {"total_messages": 3, "check_days": [14, 30, 90]},
    RiskLevel.NONE.value: {"total_messages": 2, "check_days": [30, 90]},
}

# Deterioration thresholds
DETERIORATION_THRESHOLDS: dict[str, float] = {
    "bilirubin": 50.0,
    "total_bilirubin": 50.0,
    "ALT": 100.0,
    "alt": 100.0,
    "AST": 100.0,
    "ast": 100.0,
    "INR": 30.0,
    "inr": 30.0,
    "creatinine": 50.0,
    "platelets": -30.0,
    "platelet_count": -30.0,
    "albumin": -20.0,
}

# LLM prompt for generating personalized monitoring questions
MONITORING_QUESTIONS_PROMPT = """\
You are a clinical monitoring AI. Generate personalized monitoring questions \
for a patient based on their clinical profile.

Patient context:
- Condition: {condition}
- Risk level: {risk_level}
- Chief complaint: {chief_complaint}
- Medications: {medications}
- Medical history: {history}
- Lifestyle factors: {lifestyle}
- Red flags noted: {red_flags}

Generate {num_questions} monitoring questions ranked by clinical importance.
For each question, provide:
- The question text (clear, empathetic, specific)
- A category: "symptom", "medication", "lifestyle", "labs", "general"

Return a JSON array of objects:
[{{"question": "...", "category": "...", "priority": N}}]

Priority: 1 = most important, higher = less important.

Focus on:
1. Red-flag symptom monitoring (highest priority)
2. Medication adherence and side effects
3. Condition-specific lifestyle tracking
4. Lab follow-up reminders
5. General wellbeing

Return ONLY valid JSON.\
"""

DETERIORATION_ASSESSMENT_PROMPT = """\
You are a clinical triage AI assessing a patient who has reported worsening \
symptoms during their monitoring period.

Patient context:
- Condition: {condition}
- Current risk level: {risk_level}
- Chief complaint: {chief_complaint}
- Existing red flags: {red_flags}
- Appointment date: {appointment_date}

The patient reported: "{trigger_message}"

Follow-up assessment:
{qa_pairs}

Based on this information, assess the severity and provide a recommendation.

Return a JSON object:
{{"severity": "mild" or "moderate" or "severe" or "emergency",
  "reasoning": "brief clinical explanation",
  "bring_forward_appointment": true or false,
  "urgency": "routine" or "soon" or "urgent" or "emergency",
  "additional_instructions": "specific advice for the patient"}}

Severity guide:
- mild: Symptoms are concerning but stable, continue current monitoring
- moderate: Clear worsening that needs clinical review, appointment should be brought forward
- severe: Significant deterioration requiring urgent clinical review
- emergency: Life-threatening symptoms requiring immediate A&E attendance

Return ONLY valid JSON.\
"""

CHECKIN_RESPONSE_EVALUATION_PROMPT = """\
You are a clinical triage AI. A patient answered a scheduled monitoring \
check-in question. Evaluate whether their response indicates concerning \
symptoms that need clinical assessment.

Patient context:
- Condition: {condition}
- Risk level: {risk_level}
- Red flags noted: {red_flags}

Check-in question: "{question}"
Patient answer: "{answer}"

Does this answer indicate clinically concerning symptoms that warrant \
further assessment? Consider the patient's condition context.

Return a JSON object:
{{"concerning": true or false,
  "detected_symptoms": ["list", "of", "concerning", "symptoms"],
  "reasoning": "brief explanation"}}

Return ONLY valid JSON.\
"""

NATURAL_RESPONSE_PROMPT = """\
You are a friendly, caring clinic nurse messaging a patient during their \
monitoring period. The patient has sent a message and you need to respond \
naturally and warmly.

Patient name: {patient_name}
Patient message: "{patient_message}"
Appointment date: {appointment_date}
Risk level: {risk_level}
Was this a reply to a check-in question: {is_checkin_reply}

Rules:
1. Acknowledge what the patient actually said — reference their words
2. NEVER start with "Thank you for your message" or "Thank you for contacting"
3. Keep it brief: 2-4 sentences
4. Sound like a caring nurse, not a template
5. If the patient asked a question, answer it directly
6. Mention their upcoming appointment naturally if relevant
7. If high/critical risk, gently remind about emergency contacts

Return ONLY your message text.\
"""

HEARTBEAT_CHECKIN_PROMPT = """\
You are a friendly clinic nurse sending a scheduled check-in to a patient \
during their monitoring period. Generate a warm, natural check-in message.

Patient name: {patient_name}
Days since assessment: {days}
Condition: {condition}
Risk level: {risk_level}
Appointment date: {appointment_date}
Scheduled question to include: {question}

Rules:
1. Sound warm and personal — like a nurse who remembers this patient
2. NEVER start with "Hi [name], this is your scheduled check-in"
3. Include the clinical question naturally
4. Keep it 2-4 sentences
5. Reference day count or appointment only if it adds value

Return ONLY your message text.\
"""

BOOKING_WELCOME_PROMPT = """\
You are a friendly clinic nurse welcoming a patient into their monitoring \
period after their appointment has been booked.

Patient name: {patient_name}
Risk level: {risk_level}
Number of planned check-ins: {total_messages}
Condition: {condition}

Rules:
1. Warmly welcome them to the monitoring period
2. Explain the check-in plan briefly and naturally
3. Reassure them they can message anytime
4. Sound like a caring nurse, 3-5 sentences
5. NEVER use bullet points or numbered lists

Return ONLY your message text.\
"""

DETERIORATION_QUESTION_PROMPT = """\
You are a clinical triage nurse AI. A patient in monitoring has reported \
worsening symptoms. You need to ask ONE short follow-up question.

Patient context:
- Condition: {condition}
- Current risk level: {risk_level}
- Symptoms reported: {detected_symptoms}
- Trigger message: "{trigger_message}"
- Questions already asked and answered:
{previous_qa}

RULES:
- Ask exactly ONE question about ONE topic. Do NOT combine multiple questions.
- Keep it under 2 sentences.
- Pick the single most important topic not yet covered:
  * Symptom details (location, onset, severity)
  * New or worsening symptoms
  * Functional impact on daily life
  * Emergency red flags
- Be empathetic and conversational.

Return ONLY the question text — one question, one topic.\
"""


class MonitoringAgent(BaseAgent):
    """
    The Guardian. Monitors patients through a risk-stratified, personalized
    communication plan. Adapts at each heartbeat based on patient state.
    """

    agent_name = "monitoring"

    def __init__(self, llm_client=None) -> None:
        self._client = llm_client
        self._model_name = os.getenv("MONITORING_MODEL", "gemini-2.0-flash")

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
        if event.event_type == EventType.BOOKING_COMPLETE:
            return await self._handle_booking_complete(event, diary)

        if event.event_type == EventType.HEARTBEAT:
            return await self._handle_heartbeat(event, diary)

        if event.event_type == EventType.CROSS_PHASE_REPROMPT:
            return self._handle_reprompt(event, diary)

        if event.event_type == EventType.USER_MESSAGE:
            return await self._handle_user_message(event, diary)

        if event.event_type == EventType.DOCUMENT_UPLOADED:
            return self._handle_document(event, diary)

        logger.warning(
            "Monitoring received unexpected event: %s", event.event_type.value
        )
        return AgentResult(updated_diary=diary)

    def _handle_reprompt(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Re-prompt patient after a cross-phase interaction during monitoring."""
        # If a deterioration assessment is active, the assessment flow is
        # already driving the conversation — suppress the reprompt.
        assessment = diary.monitoring.deterioration_assessment
        if assessment.active and not assessment.assessment_complete:
            return AgentResult(updated_diary=diary)

        channel = event.payload.get("channel", "websocket")
        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "We're still here if you need anything — just send us a message anytime."
            ),
            metadata={"patient_id": event.patient_id},
        )
        return AgentResult(updated_diary=diary, responses=[response])

    # ── Event Handlers ──

    async def _handle_booking_complete(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Set up monitoring or resume after reschedule."""
        # Detect reschedule: if a communication plan already exists, this is a rebooking
        is_reschedule = (
            diary.monitoring.communication_plan is not None
            and diary.monitoring.communication_plan.generated
        )

        if is_reschedule:
            return await self._handle_rebook(event, diary)
        return await self._handle_initial_booking(event, diary)

    async def _handle_rebook(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Resume monitoring after a reschedule — keep existing plan, update appointment."""
        diary.monitoring.monitoring_active = True
        diary.monitoring.appointment_date = event.payload.get(
            "appointment_date", diary.monitoring.appointment_date
        )
        diary.header.current_phase = Phase.MONITORING
        channel = event.payload.get("channel", "websocket")

        appt_date = diary.monitoring.appointment_date or "your new appointment"
        patient_name = diary.intake.name or "there"

        # Try LLM for a natural rebook acknowledgement
        rebook_msg = ""
        try:
            if self.client is not None:
                prompt = (
                    f"You are a friendly clinic nurse. The patient {patient_name} "
                    f"just rescheduled their appointment to {appt_date}. "
                    f"Briefly confirm the new date and reassure them that "
                    f"monitoring continues as before. 1-2 sentences, warm tone. "
                    f"Do NOT welcome them to the monitoring program again. "
                    f"Return ONLY your message text."
                )
                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw:
                    rebook_msg = raw.strip()
        except Exception as exc:
            logger.warning("LLM rebook message failed: %s — using fallback", exc)

        if not rebook_msg:
            rebook_msg = (
                f"Your appointment has been rescheduled to {appt_date}. "
                f"We'll continue monitoring you as before — don't hesitate to "
                f"reach out if anything comes up."
            )

        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="reschedule",
            action="Appointment rescheduled, monitoring resumed",
            detail=f"New appointment: {appt_date}",
        ))

        logger.info(
            "Monitoring resumed after reschedule for patient %s — new date: %s",
            event.patient_id, appt_date,
        )

        return AgentResult(updated_diary=diary, responses=[
            AgentResponse(
                recipient="patient",
                channel=channel,
                message=rebook_msg,
                metadata={"patient_id": event.patient_id},
            ),
        ])

    async def _handle_initial_booking(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Set up monitoring with a personalized communication plan."""
        # Baseline
        if not diary.monitoring.baseline:
            baseline = {}
            for doc in diary.clinical.documents:
                if doc.extracted_values:
                    baseline.update(doc.extracted_values)
            diary.monitoring.baseline = baseline

        diary.monitoring.monitoring_active = True
        diary.monitoring.appointment_date = event.payload.get(
            "appointment_date", diary.monitoring.appointment_date
        )
        diary.header.current_phase = Phase.MONITORING

        # ── Generate risk-stratified communication plan ──
        risk_level = diary.header.risk_level.value
        schedule = RISK_SCHEDULES.get(risk_level, RISK_SCHEDULES[RiskLevel.LOW.value])

        plan = CommunicationPlan(
            risk_level=risk_level,
            total_messages=schedule["total_messages"],
            check_in_days=schedule["check_days"],
        )

        # Generate questions and welcome message in parallel for speed
        questions_task = self._generate_monitoring_questions(diary, schedule["total_messages"])
        welcome_task = self._generate_natural_welcome(diary, plan)
        questions, welcome_msg = await asyncio.gather(questions_task, welcome_task)

        plan.questions = self._assign_questions_to_schedule(questions, schedule["check_days"])
        plan.generated = True

        diary.monitoring.communication_plan = plan

        # Set first check-in
        if plan.check_in_days:
            diary.monitoring.next_scheduled_check = str(plan.check_in_days[0])

        channel = event.payload.get("channel", "websocket")

        if not welcome_msg:
            # Fallback to template
            risk_msg = {
                "critical": "Given the urgency of your case, we'll be checking in frequently",
                "high": "We'll be monitoring you closely with regular check-ins",
                "medium": "We'll check in with you periodically",
                "low": "We'll touch base with you a few times",
            }
            freq_text = risk_msg.get(risk_level, "We'll check in with you periodically")
            welcome_msg = (
                f"Your monitoring period has begun. {freq_text} before your "
                f"appointment to make sure everything is on track. "
                f"We have {plan.total_messages} scheduled check-ins planned. "
                f"If you have any concerns or new symptoms at any time, "
                f"don't hesitate to message us."
            )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=welcome_msg,
            metadata={"patient_id": event.patient_id},
        )

        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="monitoring_setup",
            action="Monitoring activated with communication plan",
            detail=(
                f"Risk: {risk_level}, Messages: {plan.total_messages}, "
                f"Schedule: {plan.check_in_days}, "
                f"Questions: {len(plan.questions)}"
            ),
        ))

        logger.info(
            "Monitoring plan created for patient %s — risk=%s, messages=%d, questions=%d",
            event.patient_id,
            risk_level,
            plan.total_messages,
            len(plan.questions),
        )

        return AgentResult(updated_diary=diary, responses=[response])

    async def _handle_heartbeat(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """
        Execute scheduled check-in from the communication plan.

        Adaptive loop: re-evaluate patient state at each heartbeat.
        If risk has changed, adjust remaining schedule.
        """
        if not diary.monitoring.monitoring_active:
            return AgentResult(updated_diary=diary)

        # P0: Check for stalled deterioration assessment
        stalled_result = self._check_stalled_assessment(event, diary)
        if stalled_result is not None:
            return stalled_result

        # P0: Check for stalled phase transitions (non-monitoring phases)
        stale_result = self._check_phase_staleness(event, diary)
        if stale_result is not None:
            return stale_result

        days = event.payload.get("days_since_appointment", 0)
        channel = event.payload.get("channel", "websocket")
        plan = diary.monitoring.communication_plan

        responses = []
        emitted_events = []

        # Find the scheduled question for this day
        scheduled_q = None
        for q in plan.questions:
            if q.day == days and not q.sent:
                scheduled_q = q
                break

        # If no exact match, find the nearest unsent question within ±3 days
        if scheduled_q is None:
            for q in plan.questions:
                if not q.sent and abs(q.day - days) <= 3:
                    scheduled_q = q
                    break

        if scheduled_q:
            # Deliver the personalized question with LLM-generated natural framing
            message = await self._generate_natural_checkin(days, diary, scheduled_q.question)
            scheduled_q.sent = True

            responses.append(AgentResponse(
                recipient="patient",
                channel=channel,
                message=message,
                metadata={"patient_id": event.patient_id},
            ))

            diary.monitoring.add_entry(MonitoringEntry(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                type=f"scheduled_checkin_day{days}",
                action=f"Scheduled question delivered (category: {scheduled_q.category})",
                detail=scheduled_q.question[:200],
            ))
        else:
            # Generic milestone check-in
            message = await self._generate_natural_checkin(days, diary, None)
            if message:
                responses.append(AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=message,
                    metadata={"patient_id": event.patient_id},
                ))

                diary.monitoring.add_entry(MonitoringEntry(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    type=f"heartbeat_{days}d",
                    action="General check-in",
                    detail=message[:200],
                ))

        # Set next check-in
        unsent = [q for q in plan.questions if not q.sent]
        if unsent:
            diary.monitoring.next_scheduled_check = str(unsent[0].day)
        else:
            future_days = sorted(d for d in plan.check_in_days if d > days)
            diary.monitoring.next_scheduled_check = (
                str(future_days[0]) if future_days else None
            )

        # GP reminder check
        if diary.gp_channel.has_pending_queries():
            gp_reminder = EventEnvelope.handoff(
                event_type=EventType.GP_REMINDER,
                patient_id=event.patient_id,
                source_agent="monitoring",
                payload={"channel": channel},
                correlation_id=event.correlation_id,
            )
            emitted_events.append(gp_reminder)

        return AgentResult(
            updated_diary=diary,
            emitted_events=emitted_events,
            responses=responses,
        )

    async def _handle_user_message(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Reactive risk-aware response to patient message with interactive assessment."""
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")

        # ── If we're in an active deterioration assessment, process the answer ──
        # Assessment takes priority over cross-phase routing — every patient
        # message during an active assessment belongs to the monitoring agent.
        assessment = diary.monitoring.deterioration_assessment
        if assessment.active and not assessment.assessment_complete:
            return await self._process_deterioration_answer(event, diary, text, channel)

        # ── Cross-phase suppression: if cross-phase content detected and no
        #    monitoring-priority keywords, let the cross-phase agent handle it ──
        has_cross_phase = event.payload.get("_has_cross_phase_content", False)
        if has_cross_phase and not self._has_monitoring_priority_keywords(text):
            diary.monitoring.add_entry(MonitoringEntry(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                type="cross_phase_data",
                action="Cross-phase content detected — routed to specialist agent",
                detail=text[:200],
            ))
            return AgentResult(updated_diary=diary)

        # ── Post-emergency: patient already told to call 999, acknowledge simply ──
        if assessment.assessment_complete and assessment.severity == "emergency":
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "Thank you for your message. As a reminder, please call 999 "
                    "or go to your nearest A&E as soon as possible. "
                    "Our clinical team has been notified and will follow up."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        text_lower = text.lower()

        # ── Reschedule intent detection — hand off to BookingAgent ──
        reschedule_keywords = [
            "reschedule", "change my appointment", "different time",
            "different date", "can't make it", "cant make it",
            "change the date", "change the time", "move my appointment",
            "switch my appointment", "another time", "another date",
            "need to change", "want to change",
        ]
        if any(kw in text_lower for kw in reschedule_keywords):
            reschedule_event = EventEnvelope.handoff(
                event_type=EventType.RESCHEDULE_REQUEST,
                patient_id=event.patient_id,
                source_agent="monitoring",
                payload={"channel": channel, "text": text},
                correlation_id=event.correlation_id,
            )
            return AgentResult(
                updated_diary=diary,
                emitted_events=[reschedule_event],
            )

        # ── Negation-aware keyword detection ──
        def _keyword_present(text: str, keyword: str) -> bool:
            """Check if keyword is present and NOT preceded by a negation."""
            idx = text.find(keyword)
            if idx < 0:
                return False
            prefix = text[max(0, idx - 25):idx]
            negations = [
                "no ", "not ", "don't have ", "dont have ", "haven't ",
                "havent ", "without ", "deny ", "denies ", "denied ",
                "no sign of ", "never ", "doesn't ", "isn't ",
            ]
            return not any(neg in prefix for neg in negations)

        # Emergency keywords — skip assessment and go straight to escalation
        emergency_keywords = [
            "unconscious", "seizure", "collapse", "hematemesis",
            "encephalopathy", "chest pain", "can't breathe",
            "confusion", "confused", "bleeding", "blood", "jaundice",
        ]
        emergency_flags = [kw for kw in emergency_keywords if _keyword_present(text_lower, kw)]

        if emergency_flags:
            return self._immediate_emergency_escalation(event, diary, emergency_flags, channel)

        # Red flag keywords — start interactive deterioration assessment
        red_flag_keywords = [
            "jaundice", "confusion", "confused", "bleeding", "blood",
            "ascites", "hematemesis", "melena",
            "worse", "worsening", "deteriorating",
            "chest pain", "breathless", "breathlessness", "palpitations",
            "fainting", "fainted", "collapse", "severe pain",
            "swelling", "numbness", "seizure", "unconscious",
        ]
        detected_flags = [kw for kw in red_flag_keywords if _keyword_present(text_lower, kw)]

        if detected_flags:
            return await self._start_deterioration_assessment(
                event, diary, detected_flags, text, channel
            )

        # ── Record response to the last unanswered scheduled question ──
        plan = diary.monitoring.communication_plan
        last_sent = None
        for q in reversed(plan.questions):
            if q.sent and q.response is None:
                last_sent = q
                break
        if last_sent:
            last_sent.response = text

        # ── Pattern-based detection (always runs, catches "clay stool", "fever", etc.) ──
        concerning_patterns, pattern_detected = self._check_concerning_patterns(
            text_lower, diary
        )
        if concerning_patterns and pattern_detected:
            return await self._start_deterioration_assessment(
                event, diary, pattern_detected, text, channel
            )

        # ── Extract lab values from free-text and check thresholds ──
        text_lab_values = self._extract_lab_values_from_text(text)
        if text_lab_values:
            abs_alerts = self._check_absolute_thresholds(text_lab_values)
            if abs_alerts:
                diary.monitoring.add_entry(MonitoringEntry(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    type="lab_text_alert",
                    action="Critical lab values reported in chat",
                    detail=str(text_lab_values),
                ))
                alert_event = EventEnvelope.handoff(
                    event_type=EventType.DETERIORATION_ALERT,
                    patient_id=event.patient_id,
                    source_agent="monitoring",
                    payload={
                        "new_values": text_lab_values,
                        "alerts": abs_alerts,
                        "channel": channel,
                    },
                    correlation_id=event.correlation_id,
                )
                response = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=(
                        "Thank you for sharing those results. Some of these values "
                        "are outside the expected range and need urgent attention. "
                        "I've flagged this to the clinical team and they will be in "
                        "touch with you shortly. If you feel unwell, please contact "
                        "your GP or call 111."
                    ),
                    metadata={"patient_id": event.patient_id},
                )
                return AgentResult(
                    updated_diary=diary,
                    emitted_events=[alert_event],
                    responses=[response],
                )

        # ── Decide if LLM evaluation is needed ──
        # Short/neutral messages and clearly reassuring responses skip LLM
        # to prevent false-positive escalations on "no", "ok", "no change", etc.
        stable_indicators = [
            "the same", "no change", "no changes", "feeling fine",
            "feeling good", "feeling okay", "feeling ok", "feeling well",
            "feeling better", "i'm fine", "i am fine", "all good",
            "no concerns", "nothing new", "no new symptoms", "no worse",
            "not worse", "no issues", "no problems", "stable",
            "improving", "better", "good thanks", "ok thanks",
            "okay thanks", "fine thanks", "no new", "nothing to report",
        ]
        # Very short responses (< 40 chars) that don't contain red flags
        # are inherently neutral: "no", "ok", "yes", "thanks", "nothing"
        is_short_neutral = len(text.strip()) < 40 and not detected_flags
        is_reassuring = any(ind in text_lower for ind in stable_indicators)
        skip_llm = is_short_neutral or is_reassuring

        # ── Evaluate clinical significance (LLM) only for substantive messages ──
        if not skip_llm:
            concerning, detected = await self._evaluate_checkin_response(
                diary, text, last_sent
            )
            if concerning and detected:
                return await self._start_deterioration_assessment(
                    event, diary, detected, text, channel
                )

        # Normal message — context-aware acknowledgement
        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="patient_message",
            action="Message received",
            detail=text[:200],
        ))

        # If this was a reply to a scheduled check-in, give a tailored ack
        response_msg = await self._generate_natural_response(
            diary, text, is_checkin_reply=last_sent is not None
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=response_msg,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    # ── P3: Lab Value Validation ──

    # Plausible ranges for common lab values — values outside these
    # are likely extraction errors and should be flagged, not used
    LAB_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
        "bilirubin": (0.0, 50.0),           # mg/dL
        "total_bilirubin": (0.0, 50.0),
        "ALT": (0.0, 5000.0),              # U/L
        "alt": (0.0, 5000.0),
        "AST": (0.0, 5000.0),              # U/L
        "ast": (0.0, 5000.0),
        "INR": (0.5, 10.0),                # ratio
        "inr": (0.5, 10.0),
        "creatinine": (0.0, 30.0),         # mg/dL
        "platelets": (0.0, 1000.0),        # x10^9/L
        "platelet_count": (0.0, 1000.0),
        "albumin": (0.0, 60.0),            # g/L
        "hemoglobin": (0.0, 25.0),         # g/dL
        "hb": (0.0, 25.0),
        "sodium": (100.0, 200.0),          # mEq/L
        "potassium": (1.0, 10.0),          # mEq/L
        "glucose": (0.0, 1000.0),          # mg/dL
        "wbc": (0.0, 200.0),              # x10^9/L
    }

    def _validate_lab_values(
        self, values: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """
        P3: Validate extracted lab values against plausible ranges.

        Returns (validated_values, flagged_warnings).
        Values outside plausible range are excluded and flagged.
        """
        validated: dict[str, Any] = {}
        warnings: list[str] = []

        for param, val in values.items():
            try:
                num = float(val)
            except (ValueError, TypeError):
                validated[param] = val  # Non-numeric values pass through
                continue

            bounds = self.LAB_PLAUSIBLE_RANGES.get(param)
            if bounds is not None:
                lo, hi = bounds
                if num < lo or num > hi:
                    warnings.append(
                        f"{param}={num} outside plausible range [{lo}-{hi}] — excluded"
                    )
                    logger.warning(
                        "Lab value out of range: %s=%.1f (range [%.1f, %.1f]) — likely extraction error",
                        param, num, lo, hi,
                    )
                    continue

            validated[param] = val

        return validated, warnings

    def _handle_document(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Compare new lab results against baseline."""
        channel = event.payload.get("channel", "websocket")
        new_values = event.payload.get("extracted_values", {})

        if not new_values:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message="Thank you for uploading that document. We've added it to your file.",
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # P3: Validate lab values before using them
        new_values, lab_warnings = self._validate_lab_values(new_values)
        if lab_warnings:
            diary.monitoring.add_entry(MonitoringEntry(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                type="lab_validation_warning",
                action=f"Flagged {len(lab_warnings)} implausible lab values",
                detail="; ".join(lab_warnings),
            ))
            diary.monitoring.alerts_fired.append(
                f"Lab validation: {len(lab_warnings)} values excluded"
            )

        if not new_values:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "Thank you for uploading your results. Some values looked "
                    "unusual and have been flagged for manual review by our team."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # Safety: rebuild baseline from clinical docs if empty
        if not diary.monitoring.baseline:
            baseline = {}
            for doc in diary.clinical.documents:
                if doc.extracted_values:
                    baseline.update(doc.extracted_values)
            diary.monitoring.baseline = baseline
            if baseline:
                logger.info("Rebuilt monitoring baseline from clinical docs: %s", list(baseline.keys()))

        comparison = self._compare_values(diary.monitoring.baseline, new_values)
        deteriorating = comparison.get("deteriorating", [])

        # Safety net: even without baseline comparison, check if new lab values
        # exceed absolute critical thresholds (same rules as RiskScorer).
        # This catches cases where baseline is empty or doesn't have matching params.
        if not deteriorating:
            absolute_alerts = self._check_absolute_thresholds(new_values)
            if absolute_alerts:
                deteriorating = absolute_alerts
                comparison["deteriorating"] = deteriorating
                comparison.setdefault("changes", []).extend(absolute_alerts)

        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="lab_update",
            action="New lab values compared to baseline",
            detail=f"Changes: {len(comparison.get('changes', []))}",
            new_values=new_values,
            comparison=comparison,
        ))

        if deteriorating:
            diary.monitoring.alerts_fired.append(
                f"Lab deterioration: {', '.join(d['param'] for d in deteriorating)}"
            )

            alert = EventEnvelope.handoff(
                event_type=EventType.DETERIORATION_ALERT,
                patient_id=event.patient_id,
                source_agent="monitoring",
                payload={
                    "new_values": new_values,
                    "comparison": comparison,
                    "reason": f"Deterioration in: {', '.join(d['param'] for d in deteriorating)}",
                    "channel": channel,
                },
                correlation_id=event.correlation_id,
            )

            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "We've reviewed your latest results and noticed some changes "
                    "that need attention. Our clinical team has been notified and "
                    "will reassess your case. We may need to adjust your "
                    "appointment timing."
                ),
                metadata={"patient_id": event.patient_id},
            )

            return AgentResult(
                updated_diary=diary,
                emitted_events=[alert],
                responses=[response],
            )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "Thank you for uploading your latest results. We've compared "
                "them to your baseline values and everything looks stable. "
                "Your appointment is still on track."
            ),
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    # ── Communication Plan Generation ──

    async def _generate_monitoring_questions(
        self, diary: PatientDiary, num_questions: int
    ) -> list[ScheduledQuestion]:
        """Generate personalized monitoring questions using LLM, ranked by importance."""
        try:
            if self.client is None:
                return self._fallback_monitoring_questions(diary, num_questions)

            prompt = MONITORING_QUESTIONS_PROMPT.format(
                condition=diary.clinical.condition_context or "unknown",
                risk_level=diary.header.risk_level.value,
                chief_complaint=diary.clinical.chief_complaint or "not specified",
                medications=", ".join(diary.clinical.current_medications) or "none",
                history=", ".join(diary.clinical.medical_history) or "none",
                lifestyle=json.dumps(diary.clinical.lifestyle_factors) if diary.clinical.lifestyle_factors else "none",
                red_flags=", ".join(diary.clinical.red_flags) or "none",
                num_questions=num_questions,
            )

            raw_response = await llm_generate(self.client, self._model_name, prompt)
            if raw_response is None:
                return self._fallback_monitoring_questions(diary, num_questions)

            raw = raw_response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            questions_data = json.loads(raw)
            questions = []
            for item in questions_data[:num_questions]:
                questions.append(ScheduledQuestion(
                    question=item.get("question", ""),
                    day=0,  # assigned later
                    priority=item.get("priority", len(questions) + 1),
                    category=item.get("category", "general"),
                ))

            # Sort by priority (lower = more important)
            questions.sort(key=lambda q: q.priority)
            return questions

        except Exception as exc:
            logger.warning("LLM question generation failed: %s — using fallback", exc)
            return self._fallback_monitoring_questions(diary, num_questions)

    def _fallback_monitoring_questions(
        self, diary: PatientDiary, num_questions: int
    ) -> list[ScheduledQuestion]:
        """Condition-aware fallback monitoring questions."""
        questions: list[ScheduledQuestion] = []
        condition = (diary.clinical.condition_context or "").lower()
        risk = diary.header.risk_level.value

        # Red-flag symptom monitoring (highest priority)
        if diary.clinical.red_flags:
            flags_text = ", ".join(diary.clinical.red_flags[:3])
            questions.append(ScheduledQuestion(
                question=(
                    f"We previously noted {flags_text}. "
                    f"Have you experienced any of these symptoms recently? "
                    f"If so, please describe any changes."
                ),
                day=0, priority=1, category="symptom",
            ))

        # Condition-specific questions
        if any(kw in condition for kw in ["cirrhosis", "liver", "hepat"]):
            questions.extend([
                ScheduledQuestion(
                    question="Have you noticed any yellowing of your skin or eyes, or any changes in the colour of your urine?",
                    day=0, priority=2, category="symptom",
                ),
                ScheduledQuestion(
                    question="How has your alcohol consumption been since your last check-in? Have you been able to reduce or avoid it?",
                    day=0, priority=3, category="lifestyle",
                ),
            ])
        elif any(kw in condition for kw in ["mash", "nafld", "nash", "fatty"]):
            questions.extend([
                ScheduledQuestion(
                    question="Have you been able to make any changes to your diet or exercise routine? How are you finding it?",
                    day=0, priority=2, category="lifestyle",
                ),
                ScheduledQuestion(
                    question="Have you noticed any changes in your weight since your last check-in?",
                    day=0, priority=3, category="lifestyle",
                ),
            ])

        # Medication adherence
        if diary.clinical.current_medications:
            meds_text = ", ".join(diary.clinical.current_medications[:3])
            questions.append(ScheduledQuestion(
                question=f"How are you getting on with your medications ({meds_text})? Any side effects or difficulties?",
                day=0, priority=4, category="medication",
            ))

        # Lab follow-up
        questions.append(ScheduledQuestion(
            question="If you have any new lab results or test reports, please upload them so we can compare to your baseline.",
            day=0, priority=5, category="labs",
        ))

        # General wellbeing
        questions.append(ScheduledQuestion(
            question="How are you feeling overall? Is there anything about your health that's been worrying you?",
            day=0, priority=6, category="general",
        ))

        # Pain follow-up
        if diary.clinical.pain_level is not None and diary.clinical.pain_level > 3:
            questions.append(ScheduledQuestion(
                question=f"You previously rated your pain at {diary.clinical.pain_level}/10. How is your pain now? Has it changed?",
                day=0, priority=2, category="symptom",
            ))

        # Sort by priority and take top N
        questions.sort(key=lambda q: q.priority)
        return questions[:num_questions]

    def _assign_questions_to_schedule(
        self, questions: list[ScheduledQuestion], check_days: list[int]
    ) -> list[ScheduledQuestion]:
        """Assign questions to scheduled check-in days, spreading them out."""
        if not questions or not check_days:
            return questions

        # Distribute questions across check-in days
        for i, q in enumerate(questions):
            day_idx = i % len(check_days)
            q.day = check_days[day_idx]

        return questions

    # ── P0: Stalled Assessment & Phase Recovery ──

    def _check_stalled_assessment(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult | None:
        """
        P0: Force-complete a deterioration assessment that has been waiting
        for patient responses beyond ASSESSMENT_TIMEOUT_HOURS.

        Returns an AgentResult if the assessment was force-completed, None otherwise.
        """
        assessment = diary.monitoring.deterioration_assessment
        if not assessment.active or assessment.assessment_complete:
            return None

        if assessment.started is None:
            return None

        now = datetime.now(timezone.utc)
        # Ensure both are offset-aware for subtraction
        started = assessment.started
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - started).total_seconds() / 3600

        if elapsed_hours < ASSESSMENT_TIMEOUT_HOURS:
            return None

        # Force-complete with partial data — escalate conservatively
        logger.warning(
            "Stalled assessment for patient %s — %d hours elapsed, force-completing",
            event.patient_id,
            int(elapsed_hours),
        )

        assessment.assessment_complete = True
        answered = [q for q in assessment.questions if q.answer is not None]

        # Conservative escalation: if we have any answers, assess them;
        # otherwise default to moderate (better safe than sorry)
        if answered:
            # Use the fallback severity scorer on whatever data we have
            result = self._fallback_severity_assessment(diary, assessment)
            # Bump severity up one level for safety (patient went silent)
            severity_escalation = {
                "mild": "moderate",
                "moderate": "severe",
            }
            result["severity"] = severity_escalation.get(
                result["severity"], result["severity"]
            )
            result["reasoning"] = (
                f"Assessment timed out after {int(elapsed_hours)}h with "
                f"{len(answered)}/{len(assessment.questions)} answers. "
                f"Severity escalated conservatively. "
                + (result.get("reasoning") or "")
            )
        else:
            result = {
                "severity": "moderate",
                "reasoning": (
                    f"Assessment timed out after {int(elapsed_hours)}h with "
                    f"no answers. Escalating conservatively."
                ),
                "bring_forward_appointment": True,
                "urgency": "soon",
            }

        assessment.severity = result["severity"]
        assessment.reasoning = result.get("reasoning", "")
        assessment.recommendation = self._determine_recommendation(
            assessment.severity, diary
        )

        channel = event.payload.get("channel", "websocket")

        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="assessment_timeout",
            action=(
                f"Assessment force-completed after {int(elapsed_hours)}h — "
                f"severity={assessment.severity}"
            ),
            detail=assessment.reasoning or "",
        ))

        responses = []
        emitted_events = []

        # Notify patient
        responses.append(AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "We noticed you started telling us about some symptoms but "
                "we haven't heard back from you. To be safe, we've flagged "
                "your case for clinical review. If you're feeling okay, "
                "please let us know. If your symptoms have worsened, "
                "please call NHS 111 or attend A&E."
            ),
            metadata={"patient_id": event.patient_id},
        ))

        # Emit DETERIORATION_ALERT for moderate/severe/emergency
        if assessment.severity in ("moderate", "severe", "emergency"):
            alert = EventEnvelope.handoff(
                event_type=EventType.DETERIORATION_ALERT,
                patient_id=event.patient_id,
                source_agent="monitoring",
                payload={
                    "reason": f"Stalled assessment timeout ({int(elapsed_hours)}h)",
                    "source": "assessment_timeout",
                    "assessment": {
                        "severity": assessment.severity,
                        "recommendation": assessment.recommendation,
                        "reasoning": assessment.reasoning,
                        "symptoms": assessment.detected_symptoms,
                        "questions": [
                            {"q": q.question, "a": q.answer}
                            for q in assessment.questions
                        ],
                    },
                    "channel": channel,
                },
                correlation_id=event.correlation_id,
            )
            emitted_events.append(alert)

        return AgentResult(
            updated_diary=diary,
            emitted_events=emitted_events,
            responses=responses,
        )

    def _check_phase_staleness(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult | None:
        """
        P0: Detect patients stuck in a phase beyond expected SLA.

        Only fires for pre-monitoring phases (intake, clinical, booking).
        Monitoring phase is long-lived by design and doesn't stale.
        Returns an AgentResult with a staff alert if stale, None otherwise.
        """
        phase = diary.header.current_phase.value
        threshold_hours = PHASE_STALE_THRESHOLDS.get(phase)
        if threshold_hours is None:
            return None  # monitoring/closed — no staleness check

        entered_at = diary.header.phase_entered_at
        if entered_at is None:
            return None

        now = datetime.now(timezone.utc)
        if entered_at.tzinfo is None:
            entered_at = entered_at.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - entered_at).total_seconds() / 3600

        if elapsed_hours < threshold_hours:
            return None

        # Already alerted for this staleness? Check entries to avoid spam.
        stale_alert_key = f"phase_stale_{phase}"
        if any(e.type == stale_alert_key for e in diary.monitoring.entries):
            return None

        logger.warning(
            "Patient %s stuck in phase '%s' for %dh (threshold: %dh)",
            event.patient_id,
            phase,
            int(elapsed_hours),
            threshold_hours,
        )

        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type=stale_alert_key,
            action=f"Patient stuck in {phase} phase for {int(elapsed_hours)}h",
            detail=(
                f"Threshold: {threshold_hours}h. "
                f"Phase entered at: {entered_at.isoformat()}"
            ),
        ))

        diary.monitoring.alerts_fired.append(
            f"Phase stale: {phase} for {int(elapsed_hours)}h"
        )

        channel = event.payload.get("channel", "websocket")

        # Nudge the patient
        patient_name = diary.intake.name or "there"
        nudge_messages = {
            "intake": (
                f"Hi {patient_name}, we noticed we haven't finished collecting "
                f"your details yet. Would you like to continue? We just need a "
                f"few more pieces of information to get you booked in."
            ),
            "clinical": (
                f"Hi {patient_name}, we still have a few questions for your "
                f"clinical assessment. When you're ready, please send us a "
                f"message and we'll pick up where we left off."
            ),
            "booking": (
                f"Hi {patient_name}, we offered you some appointment slots "
                f"but haven't heard back. Would you like to choose one? "
                f"Just reply with your preferred option number."
            ),
        }

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=nudge_messages.get(phase, f"Hi {patient_name}, just checking in — are you still there?"),
            metadata={"patient_id": event.patient_id, "phase_stale": True},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    # ── Condition-specific concerning symptom patterns ──
    # These catch clinically significant answers that don't match simple keywords
    CONCERNING_PATTERNS: dict[str, list[str]] = {
        # Liver / hepatobiliary red flags
        "liver": [
            "red urine", "dark urine", "brown urine", "cola urine", "tea urine",
            "clay stool", "clay colored", "clay colour", "pale stool", "white stool", "grey stool",
            "light colored stool", "light colour", "chalky stool",
            "yellow skin", "yellow eyes", "yellowing",
            "swollen abdomen", "belly swelling", "stomach swelling",
            "swollen legs", "swollen ankles", "ankle swelling", "leg swelling",
            "itching", "itchy skin", "skin itching",
            "bruising easily", "easy bruising", "bruises",
            "vomiting blood", "blood in vomit",
            "black stool", "tarry stool", "dark stool", "blood in stool",
            "tarry", "black and tarry", "melena",
            "nosebleed", "nose bleed", "gum bleeding",
            "weight gain", "gained weight", "fluid",
            "foggy", "forgetful", "memory", "disoriented",
        ],
        # General red flags (apply to all conditions)
        "general": [
            "red urine", "dark urine", "blood in urine",
            "blood in stool", "black stool", "tarry stool", "tarry", "black and tarry",
            "can't eat", "cannot eat", "not eating", "stopped eating",
            "can't sleep", "cannot sleep",
            "lost weight", "weight loss", "losing weight",
            "fever", "temperature", "high temperature", "chills",
            "night sweats", "drenching sweat",
            "lump", "new lump", "mass",
            "very tired", "exhausted", "extreme fatigue", "can't get up",
            "passing out", "fainted", "dizzy", "dizziness",
            "short of breath", "difficulty breathing", "breathless",
            "rash", "skin rash", "new rash",
            "can't walk", "cannot walk", "unable to walk",
            "vomiting", "constant vomiting", "can't keep food down",
            "severe pain", "unbearable pain", "worst pain",
        ],
    }

    # ── Response Generation ──

    # ── Deterioration Assessment Flow ──

    @staticmethod
    def _has_monitoring_priority_keywords(text: str) -> bool:
        """Check for keywords that monitoring must handle — NOT suppressed by cross-phase."""
        text_lower = text.lower()
        # True emergencies
        emergency_kws = [
            "unconscious", "seizure", "collapse", "hematemesis",
            "encephalopathy", "chest pain", "can't breathe",
            "confusion", "confused", "bleeding", "blood", "jaundice",
            "severe pain",
        ]
        # Deterioration triggers (monitoring owns the assessment flow)
        deterioration_kws = [
            "worse", "worsening", "deteriorating", "fatigue", "tired",
            "swelling", "numbness", "fainting", "fainted",
            "breathless", "breathlessness", "palpitations",
        ]
        all_kws = emergency_kws + deterioration_kws
        return any(kw in text_lower for kw in all_kws)

    def _check_concerning_patterns(
        self, text_lower: str, diary: PatientDiary
    ) -> tuple[bool, list[str]]:
        """
        Negation-aware pattern-based detection for concerning symptoms.

        Checks condition-specific patterns (e.g. "clay stool" for liver)
        and general patterns (e.g. "fever", "vomiting").

        Returns (is_concerning, list_of_detected_symptoms).
        """
        detected: list[str] = []
        condition = (diary.clinical.condition_context or "").lower()

        def _is_negated(text: str, pattern: str) -> bool:
            idx = text.find(pattern)
            if idx < 0:
                return False
            prefix = text[max(0, idx - 25):idx]
            negation_phrases = [
                "no ", "not ", "don't have ", "dont have ",
                "haven't ", "havent ", "no sign of ", "without ",
                "deny ", "denies ", "denied ", "never ",
                "don't ", "doesn't ", "isn't ", "aren't ",
                "no evidence of ", "negative for ", "absent ",
            ]
            return any(neg in prefix for neg in negation_phrases)

        # Always check general patterns
        for pattern in self.CONCERNING_PATTERNS.get("general", []):
            if pattern in text_lower and not _is_negated(text_lower, pattern):
                detected.append(pattern)

        # Check condition-specific patterns
        if any(kw in condition for kw in ["cirrhosis", "liver", "hepat", "hep"]):
            for pattern in self.CONCERNING_PATTERNS.get("liver", []):
                if pattern in text_lower and not _is_negated(text_lower, pattern):
                    if pattern not in detected:
                        detected.append(pattern)

        if detected:
            logger.info(
                "Concerning symptoms detected (pattern): %s", detected,
            )
            return True, detected

        return False, []

    async def _evaluate_checkin_response(
        self,
        diary: PatientDiary,
        text: str,
        scheduled_question: ScheduledQuestion | None,
    ) -> tuple[bool, list[str]]:
        """
        LLM-based evaluation for subtler concerns that patterns don't catch.

        Returns (is_concerning, list_of_detected_symptoms).
        """
        # ── LLM-based evaluation for subtler concerns ──
        try:
            if self.client is not None and scheduled_question is not None:
                prompt = CHECKIN_RESPONSE_EVALUATION_PROMPT.format(
                    condition=diary.clinical.condition_context or "unknown",
                    risk_level=diary.header.risk_level.value,
                    red_flags=", ".join(diary.clinical.red_flags) or "none",
                    question=scheduled_question.question[:300],
                    answer=text[:500],
                )

                raw_response = await llm_generate(
                    self.client, self._model_name, prompt,
                )

                if raw_response is not None:
                    raw = raw_response.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()

                    result = json.loads(raw)
                    if result.get("concerning"):
                        llm_symptoms = result.get("detected_symptoms", ["reported concern"])
                        logger.info(
                            "Concerning symptoms detected in check-in response (LLM): %s",
                            llm_symptoms,
                        )
                        return True, llm_symptoms

        except Exception as exc:
            logger.warning("LLM check-in evaluation failed: %s", exc)

        return False, []

    async def _start_deterioration_assessment(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        detected_flags: list[str],
        trigger_text: str,
        channel: str,
    ) -> AgentResult:
        """Start an interactive clinical assessment when deterioration is suspected."""
        assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=detected_flags,
            trigger_message=trigger_text,
            started=datetime.now(timezone.utc),
        )
        diary.monitoring.deterioration_assessment = assessment

        diary.monitoring.alerts_fired.append(
            f"Deterioration assessment started: {', '.join(detected_flags)}"
        )
        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="deterioration_assessment_started",
            action="Interactive assessment initiated",
            detail=f"Detected symptoms: {detected_flags}. Trigger: {trigger_text[:200]}",
        ))

        # Generate first question
        first_question = await self._generate_assessment_question(diary, assessment, 0)
        assessment.questions.append(DeteriorationQuestion(
            question=first_question,
            category="description",
        ))

        patient_name = diary.intake.name or "there"
        response_msg = (
            f"I'm sorry to hear that, {patient_name}. "
            f"Let me ask a couple of quick questions to help our team.\n\n"
            f"{first_question}"
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=response_msg,
            metadata={"patient_id": event.patient_id},
        )

        logger.info(
            "Deterioration assessment started for patient %s — flags: %s",
            event.patient_id,
            detected_flags,
        )

        return AgentResult(updated_diary=diary, responses=[response])

    async def _process_deterioration_answer(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        text: str,
        channel: str,
    ) -> AgentResult:
        """Process an answer during the interactive deterioration assessment."""
        assessment = diary.monitoring.deterioration_assessment

        # Record the answer to the last unanswered question
        unanswered = [q for q in assessment.questions if q.answer is None]
        if unanswered:
            unanswered[0].answer = text

        # Extract structured clinical data from the answer
        self._extract_assessment_data(text, diary)

        # Check for emergency keywords in the answer itself
        # But respect negations: "no jaundice", "no confusion" should not trigger
        emergency_keywords = [
            "unconscious", "seizure", "collapse", "hematemesis",
            "chest pain", "can't breathe", "cannot breathe",
            "confusion", "confused", "bleeding", "blood", "jaundice",
        ]
        text_lower = text.lower()
        negation_patterns = ["no ", "not ", "don't have ", "haven't ", "without ", "deny ", "denies "]
        emergency_flags = []
        for kw in emergency_keywords:
            if kw in text_lower:
                # Check if preceded by a negation
                idx = text_lower.index(kw)
                prefix = text_lower[max(0, idx - 20):idx]
                if not any(neg in prefix for neg in negation_patterns):
                    emergency_flags.append(kw)
        if emergency_flags:
            assessment.assessment_complete = True
            assessment.severity = "emergency"
            assessment.recommendation = "emergency"
            return self._immediate_emergency_escalation(
                event, diary, emergency_flags, channel
            )

        answered_count = sum(1 for q in assessment.questions if q.answer is not None)

        # After 3 questions (or 2 if we have enough info), complete the assessment
        if answered_count >= 3:
            return await self._complete_deterioration_assessment(event, diary, channel)

        # Generate the next follow-up question
        category_sequence = ["description", "new_symptoms", "severity"]
        next_category = category_sequence[answered_count] if answered_count < len(category_sequence) else "functional"

        next_question = await self._generate_assessment_question(diary, assessment, answered_count)
        assessment.questions.append(DeteriorationQuestion(
            question=next_question,
            category=next_category,
        ))

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=next_question,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    async def _complete_deterioration_assessment(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        channel: str,
    ) -> AgentResult:
        """Complete the assessment: evaluate severity and take action."""
        assessment = diary.monitoring.deterioration_assessment
        assessment.assessment_complete = True

        # Assess severity using LLM with fallback
        result = await self._assess_severity(diary, assessment)
        assessment.severity = result.get("severity", "moderate")
        assessment.reasoning = result.get("reasoning", "")
        assessment.recommendation = self._determine_recommendation(
            assessment.severity, diary
        )

        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="deterioration_assessment_complete",
            action=f"Assessment complete: {assessment.severity} — {assessment.recommendation}",
            detail=assessment.reasoning or "",
        ))

        # Generate the patient-facing response
        response_msg = self._generate_assessment_outcome(diary, assessment, result)

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=response_msg,
            metadata={
                "patient_id": event.patient_id,
                "severity": assessment.severity,
                "recommendation": assessment.recommendation,
            },
        )

        emitted_events = []

        # Emit DETERIORATION_ALERT with full assessment context for clinical agent
        if assessment.severity in ("moderate", "severe", "emergency"):
            alert = EventEnvelope.handoff(
                event_type=EventType.DETERIORATION_ALERT,
                patient_id=event.patient_id,
                source_agent="monitoring",
                payload={
                    "reason": f"Deterioration assessment: {assessment.severity}",
                    "source": "deterioration_assessment",
                    "assessment": {
                        "severity": assessment.severity,
                        "recommendation": assessment.recommendation,
                        "reasoning": assessment.reasoning,
                        "symptoms": assessment.detected_symptoms,
                        "questions": [
                            {"q": q.question, "a": q.answer}
                            for q in assessment.questions
                        ],
                    },
                    "channel": channel,
                },
                correlation_id=event.correlation_id,
            )
            emitted_events.append(alert)

        logger.info(
            "Deterioration assessment complete for patient %s — severity=%s, recommendation=%s",
            event.patient_id,
            assessment.severity,
            assessment.recommendation,
        )

        return AgentResult(
            updated_diary=diary,
            emitted_events=emitted_events,
            responses=[response],
        )

    def _immediate_emergency_escalation(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        emergency_flags: list[str],
        channel: str,
    ) -> AgentResult:
        """Immediate escalation for life-threatening symptoms — no assessment needed."""
        # Mark assessment as complete so subsequent messages don't trigger a new one
        assessment = diary.monitoring.deterioration_assessment
        assessment.active = True
        assessment.assessment_complete = True
        assessment.severity = "emergency"
        assessment.recommendation = "emergency"
        assessment.detected_symptoms = list(set(
            assessment.detected_symptoms + emergency_flags
        ))

        # Deactivate monitoring — patient is being directed to A&E
        diary.monitoring.monitoring_active = False

        diary.monitoring.alerts_fired.append(
            f"EMERGENCY: {', '.join(emergency_flags)}"
        )
        diary.monitoring.add_entry(MonitoringEntry(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            type="emergency_escalation",
            action="Immediate emergency escalation",
            detail=f"Emergency flags: {emergency_flags}",
        ))

        alert = EventEnvelope.handoff(
            event_type=EventType.DETERIORATION_ALERT,
            patient_id=event.patient_id,
            source_agent="monitoring",
            payload={
                "reason": f"Emergency symptoms: {', '.join(emergency_flags)}",
                "source": "emergency_escalation",
                "assessment": {
                    "severity": "emergency",
                    "recommendation": "emergency",
                    "symptoms": emergency_flags,
                },
                "channel": channel,
            },
            correlation_id=event.correlation_id,
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "This sounds like it could be a medical emergency. "
                "Please call 999 or go to your nearest A&E immediately. "
                "Do not wait — our clinical team has been alerted and "
                "will follow up, but your immediate safety is the priority."
            ),
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(
            updated_diary=diary,
            emitted_events=[alert],
            responses=[response],
        )

    async def _generate_assessment_question(
        self,
        diary: PatientDiary,
        assessment: DeteriorationAssessment,
        question_index: int,
    ) -> str:
        """Generate the next assessment question using LLM with fallback."""
        try:
            if self.client is not None:
                previous_qa = ""
                for q in assessment.questions:
                    if q.answer:
                        previous_qa += f"Q: {q.question}\nA: {q.answer}\n"
                    else:
                        previous_qa += f"Q: {q.question}\nA: (awaiting answer)\n"

                if not previous_qa:
                    previous_qa = "(No questions asked yet)"

                prompt = DETERIORATION_QUESTION_PROMPT.format(
                    condition=diary.clinical.condition_context or "unknown",
                    risk_level=diary.header.risk_level.value,
                    detected_symptoms=", ".join(assessment.detected_symptoms),
                    trigger_message=assessment.trigger_message[:300],
                    previous_qa=previous_qa,
                )

                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw and is_response_complete(raw.strip()):
                    return raw.strip()
                if raw:
                    logger.warning("LLM assessment question appears truncated — using fallback")

        except Exception as exc:
            logger.warning("LLM question generation failed: %s — using fallback", exc)

        return self._fallback_assessment_question(diary, assessment, question_index)

    def _fallback_assessment_question(
        self,
        diary: PatientDiary,
        assessment: DeteriorationAssessment,
        question_index: int,
    ) -> str:
        """Pattern-based fallback questions for deterioration assessment."""
        condition = (diary.clinical.condition_context or "").lower()
        symptoms = ", ".join(assessment.detected_symptoms)

        if question_index == 0:
            return (
                f"Can you describe in more detail what you're experiencing? "
                f"When did these symptoms start, and have they been getting "
                f"progressively worse or do they come and go?"
            )

        if question_index == 1:
            # Condition-specific follow-up
            if any(kw in condition for kw in ["cirrhosis", "liver", "hepat"]):
                return (
                    "Have you noticed any of the following: yellowing of your "
                    "skin or eyes, swelling in your abdomen or legs, dark or "
                    "bloody stools, or any episodes of confusion?"
                )
            return (
                "Have you noticed any new symptoms besides what you mentioned? "
                "For example, any fever, unexpected weight changes, or changes "
                "in your appetite or sleep?"
            )

        if question_index == 2:
            return (
                "On a scale of 1 to 10, how severe would you say your symptoms "
                "are right now?"
            )

        return (
            "Is there anything else about how you're feeling that you think "
            "is important for us to know?"
        )

    @staticmethod
    def _extract_assessment_data(text: str, diary: PatientDiary) -> None:
        """Extract structured clinical data from assessment answers into the diary."""
        text_lower = text.lower().strip()
        pain_match = re.search(r'(\d{1,2})\s*(?:/\s*10|out\s+of\s+10)', text_lower)
        if pain_match:
            level = int(pain_match.group(1))
            if 0 <= level <= 10:
                diary.clinical.pain_level = level

        # Pain location
        location_patterns = [
            r'(?:pain\s+(?:in|at|around)\s+(?:my\s+|the\s+)?)([\w\s]+?)(?:\.|,|$)',
            r'(?:upper|lower)\s+(?:right|left)\s+(?:abdomen|quadrant|side)',
            r'(?:right|left)\s+(?:upper|lower)\s+(?:abdomen|quadrant|side)',
        ]
        for pattern in location_patterns:
            loc_match = re.search(pattern, text_lower)
            if loc_match:
                location = loc_match.group(0).strip().rstrip('.,')
                if diary.clinical.pain_location is None:
                    diary.clinical.pain_location = location
                break

    async def _assess_severity(
        self,
        diary: PatientDiary,
        assessment: DeteriorationAssessment,
    ) -> dict[str, Any]:
        """Assess deterioration severity using LLM with fallback."""
        try:
            if self.client is not None:
                qa_pairs = ""
                for q in assessment.questions:
                    qa_pairs += f"Q: {q.question}\nA: {q.answer or '(no answer)'}\n\n"

                prompt = DETERIORATION_ASSESSMENT_PROMPT.format(
                    condition=diary.clinical.condition_context or "unknown",
                    risk_level=diary.header.risk_level.value,
                    chief_complaint=diary.clinical.chief_complaint or "not specified",
                    red_flags=", ".join(diary.clinical.red_flags) or "none",
                    appointment_date=diary.monitoring.appointment_date or "upcoming",
                    trigger_message=assessment.trigger_message[:300],
                    qa_pairs=qa_pairs,
                )

                raw_response = await llm_generate(
                    self.client, self._model_name, prompt,
                    critical=True,  # P4: severity assessment is safety-critical
                )
                if raw_response is None:
                    return self._fallback_severity_assessment(diary, assessment)

                raw = raw_response.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                return json.loads(raw)

        except Exception as exc:
            logger.warning("LLM severity assessment failed: %s — using fallback", exc)

        return self._fallback_severity_assessment(diary, assessment)

    def _fallback_severity_assessment(
        self,
        diary: PatientDiary,
        assessment: DeteriorationAssessment,
    ) -> dict[str, Any]:
        """Rule-based fallback for severity assessment."""
        # Collect all text from the assessment
        all_text = assessment.trigger_message.lower()
        for q in assessment.questions:
            if q.answer:
                all_text += " " + q.answer.lower()

        # Check for severity indicators
        severe_keywords = [
            "jaundice", "confusion", "bleeding", "blood", "ascites",
            "can't move", "cannot move", "bedridden", "very severe",
            "10/10", "9/10", "8/10",
        ]
        moderate_keywords = [
            "worse", "worsening", "getting worse", "increased",
            "more pain", "more fatigue", "struggling", "difficult",
            "7/10", "6/10", "5/10",
        ]

        severe_count = sum(1 for kw in severe_keywords if kw in all_text)
        moderate_count = sum(1 for kw in moderate_keywords if kw in all_text)

        if severe_count >= 2:
            return {
                "severity": "severe",
                "reasoning": f"Multiple severe symptoms detected: {severe_count} indicators",
                "bring_forward_appointment": True,
                "urgency": "urgent",
                "additional_instructions": "Please contact NHS 111 for immediate advice.",
            }

        if severe_count >= 1 or moderate_count >= 2:
            return {
                "severity": "moderate",
                "reasoning": "Worsening symptoms that need clinical review",
                "bring_forward_appointment": True,
                "urgency": "soon",
                "additional_instructions": "We will be in touch about bringing your appointment forward.",
            }

        return {
            "severity": "mild",
            "reasoning": "Some concerning symptoms but appears manageable",
            "bring_forward_appointment": False,
            "urgency": "routine",
            "additional_instructions": "Continue monitoring and let us know if things change.",
        }

    def _determine_recommendation(
        self, severity: str, diary: PatientDiary
    ) -> str:
        """Map severity to a concrete recommendation."""
        if severity == "emergency":
            return "emergency"
        if severity == "severe":
            return "urgent_referral"
        if severity == "moderate":
            return "bring_forward"
        return "continue_monitoring"

    def _generate_assessment_outcome(
        self,
        diary: PatientDiary,
        assessment: DeteriorationAssessment,
        result: dict[str, Any],
    ) -> str:
        """Generate the patient-facing response after assessment is complete."""
        severity = assessment.severity
        appt_date = diary.monitoring.appointment_date or "your upcoming appointment"
        additional = result.get("additional_instructions", "")

        if severity == "emergency":
            return (
                "Based on what you've told me, this needs immediate attention. "
                "Please call 999 or go to your nearest A&E right away. "
                "Our clinical team has been alerted."
            )

        if severity == "severe":
            msg = (
                "Thank you for answering those questions. Based on your responses, "
                "your symptoms need urgent clinical attention. Our clinical team "
                "has been notified and will review your case as a priority. "
                "We will contact you about bringing your appointment forward."
            )
            if additional:
                msg += f"\n\nIn the meantime: {additional}"
            return msg

        if severity == "moderate":
            msg = (
                "Thank you for taking the time to answer those questions. "
                "We've passed your responses to our clinical team for review. "
                f"Based on what you've described, we're looking into bringing "
                f"your appointment (currently {appt_date}) forward to see you sooner."
            )
            if additional:
                msg += f"\n\n{additional}"
            msg += (
                "\n\nIf your symptoms get worse before we contact you, "
                "please call NHS 111 for advice."
            )
            return msg

        # mild
        msg = (
            "Thank you for letting us know and for answering those questions. "
            "We've noted the changes in your file. Based on your responses, "
            "your symptoms appear stable for now, and your current appointment "
            f"on {appt_date} remains in place."
        )
        if additional:
            msg += f"\n\n{additional}"
        msg += (
            "\n\nPlease don't hesitate to message us again if things change "
            "or you have any new concerns."
        )
        return msg

    async def _generate_natural_checkin(
        self, days: int, diary: PatientDiary, question: str | None
    ) -> str:
        """Generate a natural heartbeat check-in message with LLM, fallback to template."""
        try:
            if self.client is not None:
                prompt = HEARTBEAT_CHECKIN_PROMPT.format(
                    patient_name=diary.intake.name or "there",
                    days=days,
                    condition=diary.clinical.condition_context or "general",
                    risk_level=diary.header.risk_level.value,
                    appointment_date=diary.monitoring.appointment_date or "upcoming",
                    question=question or "Just a general check-in — how are you feeling?",
                )
                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw:
                    return raw.strip()
        except Exception as exc:
            logger.warning("LLM heartbeat check-in failed: %s — using fallback", exc)

        if question:
            patient_name = diary.intake.name or "there"
            return (
                f"Hi {patient_name}, this is your scheduled check-in "
                f"(day {days} of monitoring).\n\n{question}"
            )
        return self._fallback_milestone_message(days, diary)

    async def _generate_natural_welcome(
        self, diary: PatientDiary, plan: CommunicationPlan
    ) -> str:
        """Generate a natural monitoring welcome message with LLM, fallback to template."""
        try:
            if self.client is not None:
                prompt = BOOKING_WELCOME_PROMPT.format(
                    patient_name=diary.intake.name or "there",
                    risk_level=diary.header.risk_level.value,
                    total_messages=plan.total_messages,
                    condition=diary.clinical.condition_context or "your condition",
                )
                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw:
                    return raw.strip()
        except Exception as exc:
            logger.warning("LLM booking welcome failed: %s — using fallback", exc)
        return ""  # empty means caller uses the template

    async def _generate_natural_response(
        self, diary: PatientDiary, patient_text: str, is_checkin_reply: bool
    ) -> str:
        """Generate a natural LLM response with deterministic fallback."""
        try:
            if self.client is not None:
                prompt = NATURAL_RESPONSE_PROMPT.format(
                    patient_name=diary.intake.name or "there",
                    patient_message=patient_text[:500],
                    appointment_date=diary.monitoring.appointment_date or "your upcoming appointment",
                    risk_level=diary.header.risk_level.value,
                    is_checkin_reply="yes" if is_checkin_reply else "no",
                )
                raw = await llm_generate(self.client, self._model_name, prompt)
                if raw and not raw.strip().lower().startswith("thank you for your message"):
                    return raw.strip()
        except Exception as exc:
            logger.warning("LLM natural response failed: %s — using fallback", exc)

        if is_checkin_reply:
            return self._fallback_checkin_acknowledgement(diary, patient_text)
        return self._fallback_normal_response(diary)

    def _fallback_checkin_acknowledgement(
        self, diary: PatientDiary, patient_text: str
    ) -> str:
        """Deterministic fallback acknowledgement after a patient answers a check-in."""
        patient_name = diary.intake.name or "there"
        appt_date = diary.monitoring.appointment_date or "your upcoming appointment"

        # Check if there's a next scheduled check-in
        plan = diary.monitoring.communication_plan
        unsent = [q for q in plan.questions if not q.sent]
        next_info = ""
        if unsent:
            next_info = (
                f" We'll check in with you again around day {unsent[0].day}."
            )

        return (
            f"Thank you for the update, {patient_name}. We've noted your response "
            f"and everything looks on track. Your monitoring is ongoing and "
            f"your appointment on {appt_date} remains in place.{next_info} "
            f"If anything changes or you have new concerns, please don't "
            f"hesitate to message us."
        )

    def _fallback_normal_response(self, diary: PatientDiary) -> str:
        """Deterministic fallback risk-aware response for normal messages."""
        risk = diary.header.risk_level
        appt_date = diary.monitoring.appointment_date or "your upcoming appointment"

        # Check if there's a next scheduled check-in to mention
        plan = diary.monitoring.communication_plan
        unsent = [q for q in plan.questions if not q.sent]
        next_info = ""
        if unsent:
            next_info = f" Our next scheduled check-in is in about {unsent[0].day} days."

        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return (
                f"Thank you for your message. Your monitoring is ongoing and "
                f"your appointment on {appt_date} is coming up.{next_info} "
                f"Given your clinical profile, if you notice any worsening symptoms, "
                f"please contact NHS 111 or attend A&E without delay."
            )
        return (
            f"Thank you for your message. Your monitoring is ongoing and "
            f"your appointment on {appt_date} is coming up.{next_info} "
            f"If you have any concerns about your symptoms, please let us know "
            f"and we'll escalate to our clinical team."
        )

    def _fallback_milestone_message(
        self, days: int, diary: PatientDiary
    ) -> str:
        """Deterministic fallback contextual milestone check-in message."""
        patient_name = diary.intake.name or "there"
        appt_date = diary.monitoring.appointment_date or "your upcoming appointment"
        condition = diary.clinical.condition_context or ""

        if days <= 7:
            return (
                f"Hi {patient_name}, just a quick check-in. It's been {days} days "
                f"since your assessment. How are you feeling? Any new concerns?"
            )

        if days <= 14:
            return (
                f"Hi {patient_name}, it's been 2 weeks since your assessment. "
                f"If you have any new lab results or test reports, please upload "
                f"them so we can track your progress before your appointment "
                f"on {appt_date}."
            )

        if days <= 30:
            return (
                f"Hi {patient_name}, just checking in! It's been about a month "
                f"since your assessment. How are you feeling? If you have any "
                f"concerns or new symptoms, please let us know."
            )

        if days <= 60:
            return (
                f"Hi {patient_name}, it's been 2 months now. If your GP has "
                f"ordered any follow-up tests, please upload the results so "
                f"we can compare them to your baseline values."
            )

        return (
            f"Hi {patient_name}, it's been {days} days since your initial "
            f"assessment. We'd like to check in — how are you feeling? "
            f"Any new symptoms or changes to report?"
        )

    # ── Lab Comparison ──

    def _compare_values(
        self,
        baseline: dict[str, Any],
        new_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare new lab values against baseline, calculate % changes."""
        changes: list[dict[str, Any]] = []
        deteriorating: list[dict[str, Any]] = []

        for param, new_val in new_values.items():
            try:
                new_num = float(new_val)
            except (ValueError, TypeError):
                continue

            baseline_val = baseline.get(param)
            if baseline_val is None:
                changes.append({
                    "param": param,
                    "baseline": None,
                    "new": new_num,
                    "change_pct": None,
                    "status": "new_value",
                })
                continue

            try:
                baseline_num = float(baseline_val)
            except (ValueError, TypeError):
                continue

            if baseline_num == 0:
                change_pct = 100.0 if new_num > 0 else 0.0
            else:
                change_pct = ((new_num - baseline_num) / abs(baseline_num)) * 100

            status = "stable"
            threshold = DETERIORATION_THRESHOLDS.get(param)

            if threshold is not None:
                if threshold < 0:
                    if change_pct <= threshold:
                        status = "deteriorating"
                else:
                    if change_pct >= threshold:
                        status = "deteriorating"

            change_entry = {
                "param": param,
                "baseline": baseline_num,
                "new": new_num,
                "change_pct": round(change_pct, 1),
                "status": status,
            }
            changes.append(change_entry)

            if status == "deteriorating":
                deteriorating.append(change_entry)

        return {
            "changes": changes,
            "deteriorating": deteriorating,
            "total_compared": len(changes),
        }

    # Absolute lab value thresholds — mirrors RiskScorer HARD_RULES
    # (param, operator, threshold, description)
    ABSOLUTE_LAB_THRESHOLDS: list[tuple[str, str, float, str]] = [
        ("bilirubin", ">", 5.0, "Bilirubin > 5 mg/dL"),
        ("total_bilirubin", ">", 5.0, "Total bilirubin > 5 mg/dL"),
        ("ALT", ">", 500, "ALT > 500 U/L"),
        ("alt", ">", 500, "ALT > 500 U/L"),
        ("AST", ">", 500, "AST > 500 U/L"),
        ("ast", ">", 500, "AST > 500 U/L"),
        ("platelets", "<", 50, "Platelets < 50 x10^9/L"),
        ("platelet_count", "<", 50, "Platelet count < 50 x10^9/L"),
        ("INR", ">", 2.0, "INR > 2.0"),
        ("inr", ">", 2.0, "INR > 2.0"),
        ("creatinine", ">", 3.0, "Creatinine > 3.0 mg/dL"),
        ("albumin", "<", 2.5, "Albumin < 2.5 g/dL"),
    ]

    def _check_absolute_thresholds(
        self, new_values: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Check new lab values against absolute critical thresholds.

        This is a safety net for when baseline is empty or doesn't have
        matching parameters — critically abnormal values should ALWAYS
        fire an alert regardless of baseline comparison.
        """
        alerts: list[dict[str, Any]] = []

        for param, operator, threshold, description in self.ABSOLUTE_LAB_THRESHOLDS:
            raw_val = new_values.get(param)
            if raw_val is None:
                continue

            try:
                num_val = float(raw_val)
            except (ValueError, TypeError):
                continue

            triggered = False
            if operator == ">" and num_val > threshold:
                triggered = True
            elif operator == "<" and num_val < threshold:
                triggered = True

            if triggered:
                alerts.append({
                    "param": param,
                    "baseline": None,
                    "new": num_val,
                    "change_pct": None,
                    "status": "deteriorating",
                    "reason": f"Absolute threshold exceeded: {description}",
                })
                logger.warning(
                    "Absolute lab threshold triggered: %s=%.1f (%s)",
                    param, num_val, description,
                )

        return alerts

    @staticmethod
    def _extract_lab_values_from_text(text: str) -> dict[str, float]:
        """Extract lab parameter names and numeric values from free-text.

        Handles patterns like:
          - "bilirubin 8", "ALT 600", "bilirubin is 8.0"
          - "bilirubin 8 and ALT 600"
          - "my bilirubin was 8"
        """
        import re

        # Map of common lab name variants → canonical parameter name
        lab_aliases: dict[str, str] = {
            "bilirubin": "bilirubin",
            "bili": "bilirubin",
            "alt": "ALT",
            "alanine transaminase": "ALT",
            "ast": "AST",
            "aspartate transaminase": "AST",
            "albumin": "albumin",
            "inr": "INR",
            "platelets": "platelets",
            "plt": "platelets",
            "creatinine": "creatinine",
            "sodium": "sodium",
            "potassium": "potassium",
            "haemoglobin": "haemoglobin",
            "hemoglobin": "haemoglobin",
            "hb": "haemoglobin",
            "wbc": "WBC",
            "white blood cell": "WBC",
            "crp": "CRP",
        }

        results: dict[str, float] = {}
        text_lower = text.lower()

        for alias, canonical in lab_aliases.items():
            # Match: alias (optional "is"/"was"/"of"/"="/":") number
            pattern = rf'\b{re.escape(alias)}\b[\s:=]*(?:is|was|of|at)?\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, text_lower)
            if match:
                try:
                    results[canonical] = float(match.group(1))
                except ValueError:
                    continue

        return results

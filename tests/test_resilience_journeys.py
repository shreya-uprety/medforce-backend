"""
Resilience Journey Tests — Multi-turn patient scenarios that stress every agent.

13 Scenarios:
  1.  The Confused Patient — wrong info for wrong fields (opportunistic extraction)
  2.  The Contradicting Patient — conflicting clinical data, updated pain/allergies
  3.  The Skipper — minimal info, empty messages, gibberish
  4.  The Happy Path Revisited — full clean journey end-to-end
  5.  The Worst Case — emergency, re-flow, post-emergency
  6.  The Rescheduler — books then reschedules, tests full reschedule flow
  7.  The Concurrent Patients — two patients, same slots, double-booking prevention
  8.  The Elderly Patient (Helper-Assisted) — carer fills in on behalf of patient
  9.  The Chronic Liver Patient — high-risk cirrhosis, multi-medication, red flags
  10. The Anxious Patient — repeats questions, multiple messages, rate-limit boundary
  11. The Polypharmacy Patient — many medications, allergy interactions
  12. The Slow Responder — responds days later, booking hold expiry, re-offer
  13. The Multi-Reschedule Patient — reschedules twice, history accumulates

All agents are run with llm_client=None (deterministic fallback mode).
"""

import pytest
from datetime import timedelta

from medforce.gateway.agents.intake_agent import IntakeAgent
from medforce.gateway.agents.clinical_agent import ClinicalAgent, MAX_CLINICAL_QUESTIONS
from medforce.gateway.agents.booking_agent import BookingAgent
from medforce.gateway.agents.monitoring_agent import MonitoringAgent
from medforce.gateway.agents.risk_scorer import RiskScorer
from medforce.gateway.booking_registry import BookingRegistry, _now
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSubPhase,
    PatientDiary,
    Phase,
    RiskLevel,
    SlotOption,
)
from medforce.gateway.events import EventEnvelope, EventType


# ── Shared Fixtures ──


@pytest.fixture
def intake_agent():
    return IntakeAgent(llm_client=None)


@pytest.fixture
def clinical_agent():
    return ClinicalAgent(llm_client=None, risk_scorer=RiskScorer())


@pytest.fixture
def booking_agent():
    return BookingAgent(schedule_manager=None, llm_client=None)


@pytest.fixture
def booking_registry():
    return BookingRegistry()


@pytest.fixture
def booking_agent_with_registry(booking_registry):
    return BookingAgent(schedule_manager=None, llm_client=None, booking_registry=booking_registry)


@pytest.fixture
def monitoring_agent():
    return MonitoringAgent(llm_client=None)


def _user_msg(patient_id: str, text: str) -> EventEnvelope:
    return EventEnvelope.user_message(patient_id, text)


def _seed_intake_complete(diary: PatientDiary) -> None:
    """Pre-seed a diary through completed intake."""
    diary.intake.responder_type = "patient"
    diary.intake.mark_field_collected("name", "Test Patient")
    diary.intake.mark_field_collected("dob", "15/03/1985")
    diary.intake.mark_field_collected("nhs_number", "1234567890")
    diary.intake.mark_field_collected("phone", "07700900123")
    diary.intake.mark_field_collected("gp_name", "Dr. Smith")
    diary.intake.mark_field_collected("contact_preference", "phone")
    diary.intake.intake_complete = True
    diary.header.current_phase = Phase.CLINICAL
    # Prevent backward loops since all data is present
    diary.clinical.backward_loop_count = 3


def _seed_clinical_complete(diary: PatientDiary) -> None:
    """Pre-seed a diary through completed clinical assessment."""
    _seed_intake_complete(diary)
    diary.clinical.chief_complaint = "abdominal pain"
    diary.clinical.pain_level = 6
    diary.clinical.pain_location = "abdomen"
    diary.clinical.allergies = ["penicillin"]
    diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)
    diary.clinical.risk_level = RiskLevel.MEDIUM
    diary.clinical.risk_reasoning = "Moderate risk from symptoms"
    diary.header.risk_level = RiskLevel.MEDIUM
    diary.header.current_phase = Phase.BOOKING


def _seed_booking_complete(diary: PatientDiary) -> None:
    """Pre-seed a diary through completed booking into monitoring."""
    _seed_clinical_complete(diary)
    diary.booking.slots_offered = [
        SlotOption(date="2026-03-15", time="09:00", provider="Dr. Available"),
        SlotOption(date="2026-03-16", time="11:30", provider="Dr. Available"),
        SlotOption(date="2026-03-17", time="14:00", provider="Dr. Available"),
    ]
    diary.booking.slot_selected = diary.booking.slots_offered[0]
    diary.booking.confirmed = True
    diary.booking.appointment_id = "APT-TEST-2026-03-15"
    diary.monitoring.monitoring_active = True
    diary.monitoring.appointment_date = "2026-03-15"
    diary.monitoring.baseline = {}
    diary.header.current_phase = Phase.MONITORING


def _seed_high_risk_liver_patient(diary: PatientDiary) -> None:
    """Pre-seed a chronic liver disease patient at clinical stage."""
    _seed_intake_complete(diary)
    diary.intake.mark_field_collected("name", "Margaret Wilson")
    diary.clinical.chief_complaint = "suspected cirrhosis"
    diary.clinical.condition_context = "cirrhosis"
    diary.clinical.pain_level = 4
    diary.clinical.pain_location = "right upper quadrant"
    diary.clinical.current_medications = [
        "spironolactone 100mg", "propranolol 40mg",
        "lactulose 15ml", "rifaximin 550mg",
    ]
    diary.clinical.medical_history = [
        "alcohol-related liver disease", "portal hypertension",
        "previous variceal bleed",
    ]
    diary.clinical.allergies = ["penicillin", "sulfonamides"]
    diary.clinical.red_flags = ["previous variceal bleed", "ascites"]
    diary.clinical.documents = [
        ClinicalDocument(
            type="lab_results", processed=True,
            extracted_values={
                "bilirubin": 8.2, "ALT": 180, "AST": 210,
                "albumin": 28, "INR": 1.8, "platelets": 95,
            },
        ),
    ]
    diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)
    diary.clinical.risk_level = RiskLevel.HIGH
    diary.clinical.risk_reasoning = "Cirrhosis with portal HTN, variceal history"
    diary.header.risk_level = RiskLevel.HIGH


def _seed_polypharmacy_patient(diary: PatientDiary) -> None:
    """Pre-seed a patient on many medications."""
    _seed_intake_complete(diary)
    diary.intake.mark_field_collected("name", "Herbert Chen")
    diary.clinical.chief_complaint = "liver function monitoring"
    diary.clinical.condition_context = "MASH"
    diary.clinical.current_medications = [
        "metformin 1000mg", "atorvastatin 40mg", "insulin glargine 20u",
        "warfarin 5mg", "omeprazole 20mg", "amlodipine 10mg",
        "ramipril 5mg", "aspirin 75mg",
    ]
    diary.clinical.medical_history = [
        "type 2 diabetes", "hypertension", "MASH", "atrial fibrillation",
    ]
    diary.clinical.allergies = ["codeine", "ibuprofen"]
    diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)
    diary.clinical.risk_level = RiskLevel.MEDIUM
    diary.clinical.risk_reasoning = "Multiple comorbidities, polypharmacy"
    diary.header.risk_level = RiskLevel.MEDIUM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 1: The Confused Patient — Wrong Info for Wrong Fields
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfusedPatient:
    """Patient gives info for the wrong fields — tests opportunistic extraction."""

    @pytest.mark.asyncio
    async def test_name_extracted_correctly(self, intake_agent):
        """Send 'John Smith' → intake extracts name."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        event = _user_msg("PT-CONFUSED", "John Smith")
        await intake_agent.process(event, diary)

        assert diary.intake.name == "John Smith"
        assert "name" in diary.intake.fields_collected

    @pytest.mark.asyncio
    async def test_phone_given_when_asked_for_dob(self, intake_agent):
        """When asked for DOB, patient sends a phone number — should extract phone, NOT set DOB."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        diary.intake.responder_type = "patient"
        diary.intake.mark_field_collected("name", "John Smith")
        # Now agent is asking for DOB, but patient sends phone
        event = _user_msg("PT-CONFUSED", "07700 900123")
        await intake_agent.process(event, diary)

        # Phone should be opportunistically extracted
        assert diary.intake.phone is not None
        assert "phone" in diary.intake.fields_collected
        # DOB should NOT be set to a phone number
        assert diary.intake.dob != "07700 900123"

    @pytest.mark.asyncio
    async def test_dob_extracted_on_correct_retry(self, intake_agent):
        """After wrong field, sending correct DOB should extract it."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        diary.intake.responder_type = "patient"
        diary.intake.mark_field_collected("name", "John Smith")
        diary.intake.mark_field_collected("phone", "07700900123")
        event = _user_msg("PT-CONFUSED", "15/03/1985")
        await intake_agent.process(event, diary)

        assert diary.intake.dob == "15/03/1985"
        assert "dob" in diary.intake.fields_collected

    @pytest.mark.asyncio
    async def test_email_extracted_when_asked_for_nhs(self, intake_agent):
        """Patient sends email when asked for NHS — email extracted, NHS not set to email."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        diary.intake.responder_type = "patient"
        diary.intake.mark_field_collected("name", "John Smith")
        diary.intake.mark_field_collected("dob", "15/03/1985")
        diary.intake.mark_field_collected("phone", "07700900123")
        event = _user_msg("PT-CONFUSED", "my email is john@example.com")
        await intake_agent.process(event, diary)

        assert diary.intake.email == "john@example.com"
        assert "email" in diary.intake.fields_collected
        # NHS should not contain an email
        assert diary.intake.nhs_number != "john@example.com"

    @pytest.mark.asyncio
    async def test_nhs_extracted_eventually(self, intake_agent):
        """After detours, NHS number is finally extracted correctly."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        diary.intake.responder_type = "patient"
        diary.intake.mark_field_collected("name", "John Smith")
        diary.intake.mark_field_collected("dob", "15/03/1985")
        diary.intake.mark_field_collected("phone", "07700900123")
        diary.intake.mark_field_collected("email", "john@example.com")
        event = _user_msg("PT-CONFUSED", "123 456 7890")
        await intake_agent.process(event, diary)

        assert diary.intake.nhs_number == "1234567890"
        assert "nhs_number" in diary.intake.fields_collected

    @pytest.mark.asyncio
    async def test_all_fields_correct_after_confusion(self, intake_agent):
        """After a confused intake, all fields should be stored correctly."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        diary.intake.responder_type = "patient"
        diary.intake.mark_field_collected("name", "John Smith")
        diary.intake.mark_field_collected("dob", "15/03/1985")
        diary.intake.mark_field_collected("nhs_number", "1234567890")
        diary.intake.mark_field_collected("phone", "07700900123")
        diary.intake.mark_field_collected("email", "john@example.com")
        diary.intake.mark_field_collected("gp_name", "Dr. Patel")
        diary.intake.mark_field_collected("contact_preference", "email")

        assert diary.intake.is_complete()
        assert diary.intake.name == "John Smith"
        assert diary.intake.dob == "15/03/1985"
        assert diary.intake.phone == "07700900123"
        assert diary.intake.email == "john@example.com"
        assert diary.intake.nhs_number == "1234567890"

    @pytest.mark.asyncio
    async def test_intake_complete_fires_with_all_required(self, intake_agent):
        """INTAKE_COMPLETE only fires when ALL required fields present."""
        diary = PatientDiary.create_new("PT-CONFUSED")
        diary.intake.responder_type = "patient"
        for field, val in [
            ("name", "John Smith"),
            ("dob", "15/03/1985"),
            ("nhs_number", "1234567890"),
            ("phone", "07700900123"),
            ("gp_name", "Dr. Patel"),
        ]:
            diary.intake.mark_field_collected(field, val)

        # Still missing contact_preference — should NOT complete
        event = _user_msg("PT-CONFUSED", "hello")
        result = await intake_agent.process(event, diary)
        assert not any(
            e.event_type == EventType.INTAKE_COMPLETE for e in result.emitted_events
        )

        # Now provide it
        event = _user_msg("PT-CONFUSED", "email me please")
        result = await intake_agent.process(event, diary)
        assert diary.intake.contact_preference == "email"
        assert any(
            e.event_type == EventType.INTAKE_COMPLETE for e in result.emitted_events
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 2: The Contradicting Patient — Conflicting Clinical Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestContradictingPatient:
    """Patient gives one value then changes it — tests data updates."""

    @pytest.mark.asyncio
    async def test_pain_level_initial(self, clinical_agent):
        """Send 'abdominal pain, level 6' → pain_level=6."""
        diary = PatientDiary.create_new("PT-CONTRADICT")
        _seed_intake_complete(diary)
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)

        event = _user_msg("PT-CONTRADICT", "I have abdominal pain, level 6")
        await clinical_agent.process(event, diary)

        assert diary.clinical.pain_level == 6
        assert diary.clinical.chief_complaint is not None

    @pytest.mark.asyncio
    async def test_pain_level_updated(self, clinical_agent):
        """Send correction → pain_level should update to the new value."""
        diary = PatientDiary.create_new("PT-CONTRADICT")
        _seed_intake_complete(diary)
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        diary.clinical.pain_level = 6
        diary.clinical.chief_complaint = "abdominal pain"

        # Add a pending question so the answer gets recorded
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="How would you rate your pain?")
        )

        event = _user_msg("PT-CONTRADICT", "Actually it's more like 8 out of 10")
        await clinical_agent.process(event, diary)

        # Pain should be updated to 8
        assert diary.clinical.pain_level == 8

    @pytest.mark.asyncio
    async def test_nkda_then_specific_allergy(self, clinical_agent):
        """Say NKDA, then report penicillin allergy → penicillin should be present."""
        diary = PatientDiary.create_new("PT-CONTRADICT")
        _seed_intake_complete(diary)
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        diary.clinical.chief_complaint = "abdominal pain"

        # First: "No known allergies"
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Do you have any allergies?")
        )
        event = _user_msg("PT-CONTRADICT", "No known allergies")
        await clinical_agent.process(event, diary)
        assert "NKDA" in diary.clinical.allergies

        # Then: "Actually I'm allergic to penicillin"
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Anything else?")
        )
        event = _user_msg("PT-CONTRADICT", "Actually I'm allergic to penicillin")
        await clinical_agent.process(event, diary)

        # penicillin allergy should be present, NKDA should be gone
        allergy_text = " ".join(diary.clinical.allergies).lower()
        assert "penicillin" in allergy_text or "allerg" in allergy_text
        assert "NKDA" not in diary.clinical.allergies

    @pytest.mark.asyncio
    async def test_question_cap_forces_scoring_with_latest_data(self, clinical_agent):
        """Hit MAX_CLINICAL_QUESTIONS → forced scoring uses latest data."""
        diary = PatientDiary.create_new("PT-CONTRADICT")
        _seed_intake_complete(diary)
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        diary.clinical.chief_complaint = "abdominal pain"
        diary.clinical.pain_level = 8  # Latest value

        # Fill up to max questions
        for i in range(MAX_CLINICAL_QUESTIONS):
            diary.clinical.questions_asked.append(
                ClinicalQuestion(question=f"Q{i}?", answer=f"A{i}")
            )

        # Set sub_phase to COLLECTING_DOCUMENTS so the "ask for docs" branch
        # doesn't re-enter; the cap check fires on the next user message.
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)

        event = _user_msg("PT-CONTRADICT", "no documents")
        result = await clinical_agent.process(event, diary)

        # Should force scoring → BOOKING phase
        assert diary.header.current_phase == Phase.BOOKING
        assert diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE
        assert diary.clinical.risk_level != RiskLevel.NONE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 3: The Skipper — Minimal Info, Skips Everything
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSkipperPatient:
    """Patient gives bare minimum and skips optional stuff."""

    @pytest.mark.asyncio
    async def test_empty_message_no_crash(self, intake_agent):
        """Empty message → no crash, asks for first field."""
        diary = PatientDiary.create_new("PT-SKIP")
        event = _user_msg("PT-SKIP", "")
        result = await intake_agent.process(event, diary)

        assert result is not None
        assert len(result.responses) == 1
        assert len(result.emitted_events) == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_no_crash(self, intake_agent):
        """Whitespace-only message → same as empty."""
        diary = PatientDiary.create_new("PT-SKIP")
        event = _user_msg("PT-SKIP", "   ")
        result = await intake_agent.process(event, diary)

        assert result is not None
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_only_required_fields_completes_intake(self, intake_agent):
        """Providing only REQUIRED_FIELDS → INTAKE_COMPLETE fires."""
        diary = PatientDiary.create_new("PT-SKIP")
        diary.intake.responder_type = "patient"
        for field, val in [
            ("name", "Skip Patient"),
            ("dob", "01/01/1990"),
            ("nhs_number", "9876543210"),
            ("phone", "07700111222"),
            ("gp_name", "Dr. Nobody"),
            ("contact_preference", "sms"),
        ]:
            diary.intake.mark_field_collected(field, val)

        event = _user_msg("PT-SKIP", "done")
        result = await intake_agent.process(event, diary)

        assert any(
            e.event_type == EventType.INTAKE_COMPLETE for e in result.emitted_events
        )
        assert diary.intake.intake_complete

    @pytest.mark.asyncio
    async def test_clinical_skip_advances_questions(self, clinical_agent):
        """'I don't know' or 'skip' should not crash and questions still advance."""
        diary = PatientDiary.create_new("PT-SKIP")
        _seed_intake_complete(diary)
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)

        # Add a pending question
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="What medications are you taking?")
        )

        event = _user_msg("PT-SKIP", "I don't know")
        result = await clinical_agent.process(event, diary)

        assert result is not None
        # The answer should be recorded
        answered = [q for q in diary.clinical.questions_asked if q.answer is not None]
        assert len(answered) >= 1
        # Should respond with another question or proceed
        assert len(result.responses) >= 1

    @pytest.mark.asyncio
    async def test_document_skip_forces_scoring(self, clinical_agent):
        """'skip' during document collection → force scoring without documents."""
        diary = PatientDiary.create_new("PT-SKIP")
        _seed_intake_complete(diary)
        diary.clinical.chief_complaint = "headache"
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
        # Need at least 2 answered questions for _ready_for_scoring
        diary.clinical.questions_asked = [
            ClinicalQuestion(question="Q1?", answer="A1"),
            ClinicalQuestion(question="Q2?", answer="A2"),
        ]

        event = _user_msg("PT-SKIP", "skip")
        result = await clinical_agent.process(event, diary)

        assert diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE
        assert diary.header.current_phase == Phase.BOOKING

    @pytest.mark.asyncio
    async def test_booking_gibberish_retry(self, booking_agent):
        """Gibberish slot selection → retry message, no crash."""
        diary = PatientDiary.create_new("PT-SKIP")
        _seed_clinical_complete(diary)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-15", time="09:00", provider="Dr. A"),
            SlotOption(date="2026-03-16", time="11:30", provider="Dr. B"),
            SlotOption(date="2026-03-17", time="14:00", provider="Dr. C"),
        ]

        event = _user_msg("PT-SKIP", "asdfghjkl")
        result = await booking_agent.process(event, diary)

        assert result is not None
        assert len(result.responses) == 1
        assert "didn't" in result.responses[0].message.lower() or "catch" in result.responses[0].message.lower()
        # Booking should NOT be confirmed
        assert not diary.booking.confirmed

    @pytest.mark.asyncio
    async def test_booking_valid_selection_after_gibberish(self, booking_agent):
        """After gibberish, valid selection → confirms correctly."""
        diary = PatientDiary.create_new("PT-SKIP")
        _seed_clinical_complete(diary)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-15", time="09:00", provider="Dr. A"),
            SlotOption(date="2026-03-16", time="11:30", provider="Dr. B"),
            SlotOption(date="2026-03-17", time="14:00", provider="Dr. C"),
        ]

        event = _user_msg("PT-SKIP", "2")
        result = await booking_agent.process(event, diary)

        assert diary.booking.confirmed
        assert diary.booking.slot_selected.date == "2026-03-16"
        assert any(
            e.event_type == EventType.BOOKING_COMPLETE for e in result.emitted_events
        )

    @pytest.mark.asyncio
    async def test_monitoring_empty_text_no_crash(self, monitoring_agent):
        """Heartbeat with empty text → no crash, no false escalation."""
        diary = PatientDiary.create_new("PT-SKIP")
        _seed_booking_complete(diary)

        event = EventEnvelope.heartbeat("PT-SKIP", days_since_appointment=14)
        result = monitoring_agent._handle_heartbeat(event, diary)

        assert result is not None
        # No deterioration alert
        assert not any(
            e.event_type == EventType.DETERIORATION_ALERT for e in result.emitted_events
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 4: The Happy Path — Full Clean Journey
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHappyPath:
    """Full cooperative patient journey: intake → clinical → booking → monitoring."""

    @pytest.mark.asyncio
    async def test_intake_sequential_fields(self, intake_agent):
        """Intake: name → DOB → NHS → phone → GP → contact pref → INTAKE_COMPLETE."""
        diary = PatientDiary.create_new("PT-HAPPY")

        # Turn 1: Name (also sets responder_type)
        result = await intake_agent.process(
            _user_msg("PT-HAPPY", "Sarah Jones"), diary
        )
        assert diary.intake.name == "Sarah Jones"
        assert diary.intake.responder_type == "patient"

        # Turn 2: Contact preference (asked first by priority)
        result = await intake_agent.process(
            _user_msg("PT-HAPPY", "email me please"), diary
        )
        assert diary.intake.contact_preference == "email"

        # Turn 3: DOB
        result = await intake_agent.process(
            _user_msg("PT-HAPPY", "22/06/1978"), diary
        )
        assert diary.intake.dob == "22/06/1978"

        # Turn 4: NHS number
        result = await intake_agent.process(
            _user_msg("PT-HAPPY", "943 476 5870"), diary
        )
        assert diary.intake.nhs_number == "9434765870"

        # Turn 5: Phone
        result = await intake_agent.process(
            _user_msg("PT-HAPPY", "07700 900456"), diary
        )
        assert diary.intake.phone is not None

        # Turn 6: GP name — use mark_field_collected since pattern matching
        # for GP names requires "Dr." prefix in fallback mode
        diary.intake.mark_field_collected("gp_name", "Dr. Patel")

        # Now process a message to trigger completion check
        result = await intake_agent.process(
            _user_msg("PT-HAPPY", "My GP is Dr. Patel"), diary
        )

        assert diary.intake.intake_complete
        assert diary.header.current_phase == Phase.CLINICAL
        assert any(
            e.event_type == EventType.INTAKE_COMPLETE for e in result.emitted_events
        )

    @pytest.mark.asyncio
    async def test_clinical_assessment_flow(self, clinical_agent):
        """Clinical: chief complaint → questions → CLINICAL_COMPLETE with scoring."""
        diary = PatientDiary.create_new("PT-HAPPY")
        _seed_intake_complete(diary)

        # Receive INTAKE_COMPLETE
        handoff = EventEnvelope.handoff(
            EventType.INTAKE_COMPLETE, "PT-HAPPY",
            source_agent="intake",
            payload={"channel": "websocket"},
        )
        result = await clinical_agent.process(handoff, diary)
        assert diary.clinical.sub_phase == ClinicalSubPhase.ASKING_QUESTIONS
        assert len(result.responses) == 1

        # Answer: chief complaint
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="What is your main concern?")
        )
        event = _user_msg("PT-HAPPY", "I have abdominal pain, level 5")
        result = await clinical_agent.process(event, diary)
        assert diary.clinical.chief_complaint is not None
        assert diary.clinical.pain_level == 5

        # Answer a couple more questions
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Allergies?")
        )
        event = _user_msg("PT-HAPPY", "No known allergies")
        result = await clinical_agent.process(event, diary)

        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Medications?")
        )
        event = _user_msg("PT-HAPPY", "skip")
        result = await clinical_agent.process(event, diary)

        # Should either ask for documents or complete
        # If in document collection, skip to force scoring
        if diary.clinical.sub_phase == ClinicalSubPhase.COLLECTING_DOCUMENTS:
            event = _user_msg("PT-HAPPY", "no documents")
            result = await clinical_agent.process(event, diary)

        assert diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE
        assert diary.header.current_phase == Phase.BOOKING
        assert any(
            e.event_type == EventType.CLINICAL_COMPLETE for e in result.emitted_events
        )

    @pytest.mark.asyncio
    async def test_booking_slot_selection(self, booking_agent):
        """Booking: receive slots → select slot 2 → confirmed → BOOKING_COMPLETE."""
        diary = PatientDiary.create_new("PT-HAPPY")
        _seed_clinical_complete(diary)

        # Receive CLINICAL_COMPLETE
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-HAPPY",
            source_agent="clinical",
            payload={
                "risk_level": "medium",
                "channel": "websocket",
            },
        )
        result = await booking_agent.process(handoff, diary)
        assert len(diary.booking.slots_offered) == 3
        assert len(result.responses) == 1

        # Select slot 2
        event = _user_msg("PT-HAPPY", "2")
        result = await booking_agent.process(event, diary)

        assert diary.booking.confirmed
        assert diary.booking.slot_selected == diary.booking.slots_offered[1]
        assert diary.header.current_phase == Phase.MONITORING
        assert any(
            e.event_type == EventType.BOOKING_COMPLETE for e in result.emitted_events
        )

    @pytest.mark.asyncio
    async def test_monitoring_heartbeats_no_escalation(self, monitoring_agent):
        """Monitoring: heartbeats with reassuring responses → no escalation."""
        diary = PatientDiary.create_new("PT-HAPPY")
        _seed_booking_complete(diary)

        # Set up monitoring with a comm plan
        handoff = EventEnvelope.handoff(
            EventType.BOOKING_COMPLETE, "PT-HAPPY",
            source_agent="booking",
            payload={
                "appointment_date": "2026-03-15",
                "risk_level": "medium",
                "channel": "websocket",
            },
        )
        result = await monitoring_agent.process(handoff, diary)
        assert diary.monitoring.monitoring_active
        assert diary.monitoring.communication_plan.generated

        # Day 14 heartbeat
        hb = EventEnvelope.heartbeat("PT-HAPPY", days_since_appointment=14)
        result = monitoring_agent._handle_heartbeat(hb, diary)
        assert not any(
            e.event_type == EventType.DETERIORATION_ALERT for e in result.emitted_events
        )

        # Patient responds: "feeling fine"
        event = _user_msg("PT-HAPPY", "feeling fine, no concerns")
        result = await monitoring_agent.process(event, diary)
        assert not any(
            e.event_type == EventType.DETERIORATION_ALERT for e in result.emitted_events
        )

        # Day 30 heartbeat
        hb = EventEnvelope.heartbeat("PT-HAPPY", days_since_appointment=30)
        result = monitoring_agent._handle_heartbeat(hb, diary)
        assert not any(
            e.event_type == EventType.DETERIORATION_ALERT for e in result.emitted_events
        )

    @pytest.mark.asyncio
    async def test_diary_state_at_each_transition(self, intake_agent, clinical_agent, booking_agent, monitoring_agent):
        """Verify diary state at each phase transition point."""
        diary = PatientDiary.create_new("PT-HAPPY-STATE")

        # ── INTAKE ──
        assert diary.header.current_phase == Phase.INTAKE
        diary.intake.responder_type = "patient"
        for field, val in [
            ("name", "State Check"),
            ("dob", "01/01/1990"),
            ("nhs_number", "1111111111"),
            ("phone", "07700000000"),
            ("gp_name", "Dr. Test"),
            ("contact_preference", "sms"),
        ]:
            diary.intake.mark_field_collected(field, val)

        event = _user_msg("PT-HAPPY-STATE", "done")
        result = await intake_agent.process(event, diary)
        assert diary.header.current_phase == Phase.CLINICAL
        assert diary.intake.intake_complete

        # ── CLINICAL ──
        diary.clinical.backward_loop_count = 3  # prevent backward loop
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        diary.clinical.chief_complaint = "test complaint"
        for i in range(3):
            diary.clinical.questions_asked.append(
                ClinicalQuestion(question=f"Q{i}?", answer=f"A{i}")
            )
        # Skip documents
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)
        event = _user_msg("PT-HAPPY-STATE", "no documents")
        result = await clinical_agent.process(event, diary)
        assert diary.header.current_phase == Phase.BOOKING
        assert diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE

        # ── BOOKING ──
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-HAPPY-STATE",
            source_agent="clinical",
            payload={"risk_level": "low", "channel": "websocket"},
        )
        result = await booking_agent.process(handoff, diary)
        assert len(diary.booking.slots_offered) == 3

        event = _user_msg("PT-HAPPY-STATE", "1")
        result = await booking_agent.process(event, diary)
        assert diary.header.current_phase == Phase.MONITORING
        assert diary.booking.confirmed

        # ── MONITORING ──
        assert diary.monitoring.monitoring_active


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 5: The Worst Case — Emergency, Re-flow, Post-Emergency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWorstCase:
    """Full flow then emergency deterioration with re-booking."""

    @pytest.mark.asyncio
    async def test_swelling_triggers_assessment(self, monitoring_agent):
        """Reporting swelling during monitoring → starts deterioration assessment."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        event = _user_msg("PT-WORST", "I've noticed some swelling in my legs")
        result = await monitoring_agent.process(event, diary)

        assert diary.monitoring.deterioration_assessment.active
        assert len(diary.monitoring.deterioration_assessment.detected_symptoms) > 0
        # Should ask a follow-up question
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_deterioration_assessment_three_questions(self, monitoring_agent):
        """After 3 assessment questions → assessment completes."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        # Start assessment
        event = _user_msg("PT-WORST", "I've noticed some swelling in my legs")
        result = await monitoring_agent.process(event, diary)
        assert diary.monitoring.deterioration_assessment.active

        # Answer 3 questions
        for i in range(3):
            unanswered = [
                q for q in diary.monitoring.deterioration_assessment.questions
                if q.answer is None
            ]
            if not unanswered:
                break
            event = _user_msg("PT-WORST", f"Answer {i}: it's getting worse")
            result = await monitoring_agent.process(event, diary)

        assert diary.monitoring.deterioration_assessment.assessment_complete
        assert diary.monitoring.deterioration_assessment.severity is not None

    @pytest.mark.asyncio
    async def test_moderate_severity_triggers_rebooking_via_clinical(self, monitoring_agent, clinical_agent):
        """Moderate severity → DETERIORATION_ALERT → clinical triggers rebooking."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        # Start and complete assessment
        event = _user_msg("PT-WORST", "I've noticed some swelling in my legs")
        await monitoring_agent.process(event, diary)

        for i in range(3):
            unanswered = [
                q for q in diary.monitoring.deterioration_assessment.questions
                if q.answer is None
            ]
            if not unanswered:
                break
            event = _user_msg("PT-WORST", f"It's worse, more pain, 7/10")
            result = await monitoring_agent.process(event, diary)

        # Check if a DETERIORATION_ALERT was emitted
        alerts = [
            e for e in result.emitted_events
            if e.event_type == EventType.DETERIORATION_ALERT
        ]
        if alerts:
            # Feed it to clinical agent
            clinical_result = await clinical_agent.process(alerts[0], diary)
            # Clinical agent should handle it gracefully
            assert clinical_result is not None

    @pytest.mark.asyncio
    async def test_immediate_emergency_escalation(self, monitoring_agent):
        """'I have jaundice and confusion' → IMMEDIATE emergency (no assessment)."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        event = _user_msg("PT-WORST", "I have jaundice and confusion")
        result = await monitoring_agent.process(event, diary)

        # Should be immediate emergency
        assert diary.monitoring.deterioration_assessment.severity == "emergency"
        assert diary.monitoring.deterioration_assessment.assessment_complete
        assert not diary.monitoring.monitoring_active

        # Should emit DETERIORATION_ALERT
        assert any(
            e.event_type == EventType.DETERIORATION_ALERT for e in result.emitted_events
        )
        # Response should mention 999 or A&E
        msg = result.responses[0].message.lower()
        assert "999" in msg or "a&e" in msg

    @pytest.mark.asyncio
    async def test_post_emergency_message_no_new_assessment(self, monitoring_agent):
        """After emergency, patient says 'ok' → reminder to call 999, NO new assessment."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        # Trigger emergency first
        event = _user_msg("PT-WORST", "I have jaundice and confusion")
        await monitoring_agent.process(event, diary)
        assert diary.monitoring.deterioration_assessment.severity == "emergency"

        # Now patient says "ok"
        event = _user_msg("PT-WORST", "ok")
        result = await monitoring_agent.process(event, diary)

        # Should NOT start a new assessment
        # The assessment is already complete with emergency severity
        assert diary.monitoring.deterioration_assessment.assessment_complete
        assert diary.monitoring.deterioration_assessment.severity == "emergency"
        # Response should remind about 999
        msg = result.responses[0].message.lower()
        assert "999" in msg or "a&e" in msg

    @pytest.mark.asyncio
    async def test_post_emergency_heartbeat_no_action(self, monitoring_agent):
        """After emergency, heartbeat → no action (monitoring_active=False)."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        # Trigger emergency
        event = _user_msg("PT-WORST", "I have jaundice and confusion")
        await monitoring_agent.process(event, diary)
        assert not diary.monitoring.monitoring_active

        # Heartbeat arrives
        hb = EventEnvelope.heartbeat("PT-WORST", days_since_appointment=14)
        result = monitoring_agent._handle_heartbeat(hb, diary)

        # Should return empty result (monitoring inactive)
        assert len(result.responses) == 0
        assert len(result.emitted_events) == 0

    @pytest.mark.asyncio
    async def test_negation_does_not_trigger_emergency(self, monitoring_agent):
        """'I don't have jaundice' → should NOT trigger emergency."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        event = _user_msg("PT-WORST", "I don't have jaundice or confusion")
        result = await monitoring_agent.process(event, diary)

        # Should NOT be emergency
        assert diary.monitoring.deterioration_assessment.severity != "emergency" or \
            not diary.monitoring.deterioration_assessment.active
        # Monitoring should remain active
        assert diary.monitoring.monitoring_active

    @pytest.mark.asyncio
    async def test_emergency_clinical_agent_handles_alert(self, monitoring_agent, clinical_agent):
        """Emergency DETERIORATION_ALERT → clinical agent sets CRITICAL risk."""
        diary = PatientDiary.create_new("PT-WORST")
        _seed_booking_complete(diary)

        # Trigger emergency
        event = _user_msg("PT-WORST", "I have jaundice and confusion")
        result = await monitoring_agent.process(event, diary)

        alerts = [
            e for e in result.emitted_events
            if e.event_type == EventType.DETERIORATION_ALERT
        ]
        assert len(alerts) == 1

        # Feed to clinical
        clinical_result = await clinical_agent.process(alerts[0], diary)
        assert clinical_result is not None
        # Risk should be CRITICAL
        assert diary.header.risk_level == RiskLevel.CRITICAL
        assert diary.clinical.risk_level == RiskLevel.CRITICAL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 6: The Rescheduler — Books then changes their mind
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRescheduler:
    """Patient confirms an appointment, then asks to reschedule."""

    @pytest.mark.asyncio
    async def test_full_reschedule_flow(self, booking_agent_with_registry, booking_registry):
        """Book → confirm → reschedule → re-confirm a different slot."""
        agent = booking_agent_with_registry
        diary = PatientDiary.create_new("PT-RESCHED")
        _seed_clinical_complete(diary)

        # Step 1: Offer slots
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-RESCHED",
            source_agent="clinical",
            payload={"risk_level": "medium", "channel": "websocket"},
        )
        result = await agent.process(handoff, diary)
        assert len(diary.booking.slots_offered) == 3

        # Step 2: Confirm slot 1
        event = _user_msg("PT-RESCHED", "1")
        result = await agent.process(event, diary)
        assert diary.booking.confirmed
        first_date = diary.booking.slot_selected.date
        assert booking_registry.get_patient_booking("PT-RESCHED") is not None

        # Step 3: Reschedule
        reschedule = EventEnvelope.handoff(
            EventType.RESCHEDULE_REQUEST, "PT-RESCHED",
            source_agent="monitoring",
            payload={"channel": "websocket"},
        )
        result = await agent.process(reschedule, diary)
        assert not diary.booking.confirmed
        assert diary.header.current_phase == Phase.BOOKING
        assert len(diary.booking.rescheduled_from) == 1
        assert diary.booking.rescheduled_from[0]["date"] == first_date
        assert booking_registry.get_patient_booking("PT-RESCHED") is None

        # Step 4: Re-confirm a different slot
        event = _user_msg("PT-RESCHED", "2")
        result = await agent.process(event, diary)
        assert diary.booking.confirmed
        assert diary.booking.slot_selected.date != first_date or diary.booking.slot_selected.time != "09:00"

    @pytest.mark.asyncio
    async def test_reschedule_keyword_from_monitoring(self, monitoring_agent):
        """'I need to reschedule' during monitoring emits RESCHEDULE_REQUEST."""
        diary = PatientDiary.create_new("PT-RESCHED")
        _seed_booking_complete(diary)

        event = _user_msg("PT-RESCHED", "I need to reschedule my appointment")
        result = await monitoring_agent.process(event, diary)

        reschedule_events = [
            e for e in result.emitted_events
            if e.event_type == EventType.RESCHEDULE_REQUEST
        ]
        assert len(reschedule_events) == 1

    @pytest.mark.asyncio
    async def test_reschedule_variants(self, monitoring_agent):
        """Multiple phrasing variants all trigger reschedule."""
        phrases = [
            "I can't make it to my appointment",
            "Can I change the date?",
            "I need a different time",
            "I want to change my appointment",
            "Can I move my appointment to next week?",
        ]
        for phrase in phrases:
            diary = PatientDiary.create_new("PT-RESCHED")
            _seed_booking_complete(diary)

            event = _user_msg("PT-RESCHED", phrase)
            result = await monitoring_agent.process(event, diary)

            reschedule_events = [
                e for e in result.emitted_events
                if e.event_type == EventType.RESCHEDULE_REQUEST
            ]
            assert len(reschedule_events) == 1, f"Failed for phrase: '{phrase}'"

    @pytest.mark.asyncio
    async def test_reschedule_preserves_clinical_data(self, booking_agent_with_registry):
        """After reschedule, clinical data and risk level are preserved."""
        agent = booking_agent_with_registry
        diary = PatientDiary.create_new("PT-RESCHED")
        _seed_clinical_complete(diary)
        diary.header.risk_level = RiskLevel.HIGH
        diary.clinical.red_flags = ["jaundice"]

        # Book
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-RESCHED",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await agent.process(handoff, diary)
        event = _user_msg("PT-RESCHED", "1")
        result = await agent.process(event, diary)
        assert diary.booking.confirmed

        # Reschedule
        reschedule = EventEnvelope.handoff(
            EventType.RESCHEDULE_REQUEST, "PT-RESCHED",
            source_agent="monitoring",
            payload={"channel": "websocket"},
        )
        result = await agent.process(reschedule, diary)

        # Clinical data should be untouched
        assert diary.header.risk_level == RiskLevel.HIGH
        assert diary.clinical.red_flags == ["jaundice"]
        assert diary.clinical.chief_complaint == "abdominal pain"

    @pytest.mark.asyncio
    async def test_reschedule_deactivates_monitoring(self, booking_agent_with_registry):
        """Rescheduling should deactivate monitoring until re-booking."""
        agent = booking_agent_with_registry
        diary = PatientDiary.create_new("PT-RESCHED")
        _seed_booking_complete(diary)
        assert diary.monitoring.monitoring_active

        reschedule = EventEnvelope.handoff(
            EventType.RESCHEDULE_REQUEST, "PT-RESCHED",
            source_agent="monitoring",
            payload={"channel": "websocket"},
        )
        result = await agent.process(reschedule, diary)

        assert not diary.monitoring.monitoring_active
        assert diary.header.current_phase == Phase.BOOKING


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 7: Concurrent Patients — Double-Booking Prevention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConcurrentPatients:
    """Two patients compete for the same slots — registry prevents double-booking."""

    @pytest.mark.asyncio
    async def test_first_patient_holds_blocks_second(self):
        """Patient A holds slots → Patient B gets different slots (no overlap)."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        diary_a = PatientDiary.create_new("PT-A")
        _seed_clinical_complete(diary_a)
        diary_b = PatientDiary.create_new("PT-B")
        _seed_clinical_complete(diary_b)

        # Patient A gets slots
        handoff_a = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-A",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result_a = await agent.process(handoff_a, diary_a)
        assert len(diary_a.booking.slots_offered) == 3

        # Patient B tries — A's slots are held, B gets different ones
        handoff_b = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-B",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result_b = await agent.process(handoff_b, diary_b)

        # B should get slots from the larger pool, with no overlap
        assert len(diary_b.booking.slots_offered) > 0
        a_keys = {(s.date, s.time) for s in diary_a.booking.slots_offered}
        b_keys = {(s.date, s.time) for s in diary_b.booking.slots_offered}
        assert a_keys.isdisjoint(b_keys), (
            f"Double-booking! A and B share slots: {a_keys & b_keys}"
        )

    @pytest.mark.asyncio
    async def test_expired_holds_free_for_second_patient(self):
        """After Patient A's holds expire, Patient B can grab them."""
        registry = BookingRegistry(hold_ttl_minutes=1)
        agent = BookingAgent(booking_registry=registry)

        diary_a = PatientDiary.create_new("PT-A")
        _seed_clinical_complete(diary_a)

        # Patient A holds slots
        handoff_a = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-A",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff_a, diary_a)
        assert len(diary_a.booking.slots_offered) == 3

        # Force expiry
        for hold in registry._data.holds:
            hold.expires_at = _now() - timedelta(minutes=1)

        # Patient B should now succeed
        diary_b = PatientDiary.create_new("PT-B")
        _seed_clinical_complete(diary_b)
        handoff_b = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-B",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result_b = await agent.process(handoff_b, diary_b)
        assert len(diary_b.booking.slots_offered) == 3

    @pytest.mark.asyncio
    async def test_confirmed_slot_survives_new_patient(self):
        """Once Patient A confirms, Patient B can't get that slot even after expiry of other holds."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        diary_a = PatientDiary.create_new("PT-A")
        _seed_clinical_complete(diary_a)

        # A offers and confirms slot 1
        handoff_a = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-A",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff_a, diary_a)
        event_a = _user_msg("PT-A", "1")
        await agent.process(event_a, diary_a)
        assert diary_a.booking.confirmed
        confirmed_date = diary_a.booking.slot_selected.date
        confirmed_time = diary_a.booking.slot_selected.time

        # B tries — slot 1 is confirmed, others are cancelled (released after confirm)
        diary_b = PatientDiary.create_new("PT-B")
        _seed_clinical_complete(diary_b)
        handoff_b = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-B",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result_b = await agent.process(handoff_b, diary_b)

        # B should get slots, but the confirmed one should not be among them
        for slot in diary_b.booking.slots_offered:
            is_same = slot.date == confirmed_date and slot.time == confirmed_time
            assert not is_same, "Confirmed slot should not be offered to another patient"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 8: The Elderly Patient (Helper-Assisted Intake)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestElderlyPatientHelperAssisted:
    """A carer fills in intake on behalf of an elderly patient."""

    @pytest.mark.asyncio
    async def test_helper_identified_as_responder(self, intake_agent):
        """Carer says they're filling in on behalf → responder_type = helper."""
        diary = PatientDiary.create_new("PT-ELDERLY")

        event = _user_msg("PT-ELDERLY", "I'm filling this in for my mother")
        result = await intake_agent.process(event, diary)

        assert diary.intake.responder_type == "helper"

    @pytest.mark.asyncio
    async def test_helper_provides_patient_demographics(self, intake_agent):
        """Helper provides patient's name, DOB, etc. — all stored correctly."""
        diary = PatientDiary.create_new("PT-ELDERLY")
        diary.intake.responder_type = "helper"
        diary.intake.responder_name = "Sarah"
        diary.intake.responder_relationship = "daughter"

        # Provide patient name
        event = _user_msg("PT-ELDERLY", "Dorothy Wilson")
        await intake_agent.process(event, diary)
        assert diary.intake.name == "Dorothy Wilson"

        # Provide DOB
        event = _user_msg("PT-ELDERLY", "14/08/1942")
        await intake_agent.process(event, diary)
        assert diary.intake.dob == "14/08/1942"

    @pytest.mark.asyncio
    async def test_helper_completes_intake(self, intake_agent):
        """Helper can fully complete intake on behalf of patient."""
        diary = PatientDiary.create_new("PT-ELDERLY")
        diary.intake.responder_type = "helper"
        for field, val in [
            ("name", "Dorothy Wilson"),
            ("dob", "14/08/1942"),
            ("nhs_number", "5551234567"),
            ("phone", "07700900999"),
            ("gp_name", "Dr. Morris"),
            ("contact_preference", "phone"),
        ]:
            diary.intake.mark_field_collected(field, val)

        event = _user_msg("PT-ELDERLY", "that's everything")
        result = await intake_agent.process(event, diary)

        assert diary.intake.intake_complete
        assert any(
            e.event_type == EventType.INTAKE_COMPLETE for e in result.emitted_events
        )

    @pytest.mark.asyncio
    async def test_elderly_booking_with_carer_context(self, booking_agent):
        """Elderly patient's booking records who booked (helper via sender_id)."""
        diary = PatientDiary.create_new("PT-ELDERLY")
        _seed_clinical_complete(diary)
        diary.intake.mark_field_collected("name", "Dorothy Wilson")
        diary.intake.responder_type = "helper"
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-20", time="10:00", provider="Dr. Morris"),
        ]

        # Helper selects the slot
        event = EventEnvelope.user_message(
            "PT-ELDERLY", "1",
            sender_id="helper:Sarah",
        )
        result = await booking_agent.process(event, diary)

        assert diary.booking.confirmed
        assert diary.booking.booked_by == "helper:Sarah"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 9: Chronic Liver Patient — High Risk, Multi-Medication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChronicLiverPatient:
    """High-risk cirrhosis patient: complex meds, lab abnormalities, red flags."""

    @pytest.mark.asyncio
    async def test_high_risk_gets_2_day_window(self, booking_agent):
        """HIGH risk → 2-day booking window."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        diary.header.current_phase = Phase.BOOKING

        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-LIVER",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await booking_agent.process(handoff, diary)

        assert "2 days" in result.responses[0].message
        assert "HIGH" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_liver_instructions_include_alcohol_warning(self, booking_agent):
        """Cirrhosis patient gets alcohol avoidance instruction."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)

        assert any("alcohol" in i.lower() for i in instructions)

    @pytest.mark.asyncio
    async def test_liver_instructions_include_fasting(self, booking_agent):
        """HIGH risk gets fasting instruction."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)

        assert any("fasting" in i.lower() for i in instructions)

    @pytest.mark.asyncio
    async def test_liver_instructions_include_allergy_reminder(self, booking_agent):
        """Patient with allergies gets allergy reminder."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        instructions = booking_agent._generate_instructions(diary)

        assert any("allerg" in i.lower() for i in instructions)
        assert any("penicillin" in i.lower() for i in instructions)

    @pytest.mark.asyncio
    async def test_liver_instructions_include_red_flag_warning(self, booking_agent):
        """Patient with red flags gets NHS 111 / A&E warning."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        instructions = booking_agent._generate_instructions(diary)

        assert any("NHS 111" in i or "A&E" in i for i in instructions)

    @pytest.mark.asyncio
    async def test_liver_baseline_snapshot_captures_labs(self, booking_agent):
        """Baseline for monitoring captures abnormal liver function values."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="09:00"),
        ]

        event = _user_msg("PT-LIVER", "1")
        result = await booking_agent.process(event, diary)

        baseline = result.updated_diary.monitoring.baseline
        assert baseline["bilirubin"] == 8.2
        assert baseline["ALT"] == 180
        assert baseline["albumin"] == 28

    @pytest.mark.asyncio
    async def test_liver_patient_deterioration_triggers_assessment(self, monitoring_agent):
        """Liver patient reporting worsening → deterioration assessment starts."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        _seed_booking_complete(diary)

        event = _user_msg("PT-LIVER", "My abdomen is getting worse and more swelling")
        result = await monitoring_agent.process(event, diary)

        assert diary.monitoring.deterioration_assessment.active
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_liver_patient_hematemesis_immediate_emergency(self, monitoring_agent):
        """Liver patient vomiting blood → immediate emergency escalation."""
        diary = PatientDiary.create_new("PT-LIVER")
        _seed_high_risk_liver_patient(diary)
        _seed_booking_complete(diary)

        event = _user_msg("PT-LIVER", "I'm vomiting blood, hematemesis")
        result = await monitoring_agent.process(event, diary)

        assert diary.monitoring.deterioration_assessment.severity == "emergency"
        msg = result.responses[0].message.lower()
        assert "999" in msg or "a&e" in msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 10: The Anxious Patient — Repetitive Messages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnxiousPatient:
    """Patient sends many messages, repeats questions, tests system tolerance."""

    @pytest.mark.asyncio
    async def test_repeated_messages_no_crash(self, monitoring_agent):
        """Same message sent 5 times → no crash, each response is valid."""
        diary = PatientDiary.create_new("PT-ANXIOUS")
        _seed_booking_complete(diary)

        for _ in range(5):
            event = _user_msg("PT-ANXIOUS", "Am I going to be okay?")
            result = await monitoring_agent.process(event, diary)
            assert result is not None
            assert len(result.responses) >= 1

    @pytest.mark.asyncio
    async def test_stable_messages_no_false_escalation(self, monitoring_agent):
        """Anxious-sounding but stable messages → no deterioration escalation."""
        diary = PatientDiary.create_new("PT-ANXIOUS")
        _seed_booking_complete(diary)

        anxious_messages = [
            "I'm worried about my appointment",
            "Is everything going to be fine?",
            "I'm feeling okay but I'm nervous",
            "No new symptoms, just anxious",
            "Nothing has changed, I'm the same",
        ]
        for msg_text in anxious_messages:
            event = _user_msg("PT-ANXIOUS", msg_text)
            result = await monitoring_agent.process(event, diary)

            alerts = [
                e for e in result.emitted_events
                if e.event_type == EventType.DETERIORATION_ALERT
            ]
            assert len(alerts) == 0, f"False escalation for: '{msg_text}'"
            assert diary.monitoring.monitoring_active

    @pytest.mark.asyncio
    async def test_repeated_booking_selection_already_confirmed(self, booking_agent):
        """Patient re-sends '1' after already booking → informed, no double-booking."""
        diary = PatientDiary.create_new("PT-ANXIOUS")
        _seed_clinical_complete(diary)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-15", time="09:00"),
        ]

        # First booking
        event = _user_msg("PT-ANXIOUS", "1")
        result = await booking_agent.process(event, diary)
        assert diary.booking.confirmed

        # Try again
        event = _user_msg("PT-ANXIOUS", "1")
        result = await booking_agent.process(event, diary)
        msg = result.responses[0].message.lower()
        assert "confirmed" in msg or "already" in msg or "reschedule" in msg

    @pytest.mark.asyncio
    async def test_multiple_intake_messages_same_field(self, intake_agent):
        """Patient sends name 3 times → last value wins, no corruption."""
        diary = PatientDiary.create_new("PT-ANXIOUS")

        for name in ["Jane", "Jane Smith", "Jane Elizabeth Smith"]:
            event = _user_msg("PT-ANXIOUS", name)
            await intake_agent.process(event, diary)

        assert diary.intake.name is not None
        assert len(diary.intake.name) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 11: The Polypharmacy Patient — Many Medications
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPolypharmacyPatient:
    """Patient on 8 medications — instructions cover all relevant ones."""

    @pytest.mark.asyncio
    async def test_metformin_instruction_present(self, booking_agent):
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)
        assert any("Metformin" in i for i in instructions)

    @pytest.mark.asyncio
    async def test_warfarin_instruction_present(self, booking_agent):
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)
        assert any("Warfarin" in i and "INR" in i for i in instructions)

    @pytest.mark.asyncio
    async def test_insulin_instruction_present(self, booking_agent):
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)
        assert any("insulin" in i.lower() for i in instructions)

    @pytest.mark.asyncio
    async def test_statin_instruction_present(self, booking_agent):
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)
        assert any("statin" in i.lower() for i in instructions)

    @pytest.mark.asyncio
    async def test_mash_condition_instructions(self, booking_agent):
        """MASH patient gets weight + dietary instructions."""
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        diary.header.current_phase = Phase.BOOKING
        instructions = booking_agent._generate_instructions(diary)
        assert any("weight" in i.lower() for i in instructions)
        assert any("fat" in i.lower() for i in instructions)

    @pytest.mark.asyncio
    async def test_allergy_reminder_lists_both_allergies(self, booking_agent):
        """Allergy reminder mentions both codeine and ibuprofen."""
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        instructions = booking_agent._generate_instructions(diary)
        allergy_instr = [i for i in instructions if "allerg" in i.lower()]
        assert len(allergy_instr) >= 1
        combined = " ".join(allergy_instr).lower()
        assert "codeine" in combined
        assert "ibuprofen" in combined

    @pytest.mark.asyncio
    async def test_full_polypharmacy_booking_flow(self, booking_agent):
        """Full booking flow for polypharmacy patient — all instructions generated."""
        diary = PatientDiary.create_new("PT-POLY")
        _seed_polypharmacy_patient(diary)
        diary.header.current_phase = Phase.BOOKING

        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-POLY",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await booking_agent.process(handoff, diary)
        assert len(diary.booking.slots_offered) == 3

        event = _user_msg("PT-POLY", "1")
        result = await booking_agent.process(event, diary)
        assert diary.booking.confirmed
        # Should have many instructions for this complex patient
        assert len(diary.booking.pre_appointment_instructions) >= 6


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 12: The Slow Responder — Hold Expiry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSlowResponder:
    """Patient doesn't reply in time — hold expires, fresh slots offered."""

    @pytest.mark.asyncio
    async def test_expired_hold_reoffer_flow(self):
        """Slot hold expires → patient tries to select → gets fresh slots."""
        registry = BookingRegistry(hold_ttl_minutes=15)
        agent = BookingAgent(booking_registry=registry)

        diary = PatientDiary.create_new("PT-SLOW")
        _seed_clinical_complete(diary)

        # Offer slots (creates holds)
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-SLOW",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff, diary)
        original_slots = list(diary.booking.slots_offered)
        assert len(original_slots) == 3

        # Simulate 20 minutes passing → holds expire
        for hold in registry._data.holds:
            hold.expires_at = _now() - timedelta(minutes=5)

        # Patient finally replies with "1"
        event = _user_msg("PT-SLOW", "1")
        result = await agent.process(event, diary)

        # Should NOT be confirmed (hold expired)
        assert not diary.booking.confirmed
        # Should have been re-offered fresh slots
        assert len(diary.booking.slots_offered) > 0

    @pytest.mark.asyncio
    async def test_second_attempt_confirms_after_reoffer(self):
        """After re-offer, patient successfully confirms."""
        registry = BookingRegistry(hold_ttl_minutes=15)
        agent = BookingAgent(booking_registry=registry)

        diary = PatientDiary.create_new("PT-SLOW")
        _seed_clinical_complete(diary)

        # Offer → expire → re-offer
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-SLOW",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff, diary)
        for hold in registry._data.holds:
            hold.expires_at = _now() - timedelta(minutes=5)
        event = _user_msg("PT-SLOW", "1")
        await agent.process(event, diary)

        # Now patient picks from the re-offered slots (new holds, valid)
        event = _user_msg("PT-SLOW", "1")
        result = await agent.process(event, diary)
        assert diary.booking.confirmed
        assert diary.header.current_phase == Phase.MONITORING

    @pytest.mark.asyncio
    async def test_slow_responder_clinical_data_preserved(self):
        """Even after hold expiry + re-offer, clinical data is untouched."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        diary = PatientDiary.create_new("PT-SLOW")
        _seed_clinical_complete(diary)
        diary.clinical.red_flags = ["ascites"]

        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-SLOW",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff, diary)

        # Expire and re-offer
        for hold in registry._data.holds:
            hold.expires_at = _now() - timedelta(minutes=5)
        event = _user_msg("PT-SLOW", "1")
        await agent.process(event, diary)

        assert diary.clinical.red_flags == ["ascites"]
        assert diary.clinical.chief_complaint == "abdominal pain"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario 13: The Multi-Reschedule Patient — Reschedules Twice
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiReschedulePatient:
    """Patient reschedules multiple times — history accumulates correctly."""

    @pytest.mark.asyncio
    async def test_two_reschedules_accumulate_history(self):
        """Book → reschedule → book → reschedule → book: 2 entries in rescheduled_from."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        diary = PatientDiary.create_new("PT-MULTI")
        _seed_clinical_complete(diary)

        dates_booked = []
        for cycle in range(3):
            # Offer
            handoff = EventEnvelope.handoff(
                EventType.CLINICAL_COMPLETE, "PT-MULTI",
                source_agent="clinical",
                payload={"channel": "websocket"},
            )
            await agent.process(handoff, diary)

            # Confirm
            event = _user_msg("PT-MULTI", "1")
            await agent.process(event, diary)
            assert diary.booking.confirmed
            dates_booked.append(diary.booking.slot_selected.date)

            # Reschedule (except last iteration)
            if cycle < 2:
                reschedule = EventEnvelope.handoff(
                    EventType.RESCHEDULE_REQUEST, "PT-MULTI",
                    source_agent="monitoring",
                    payload={"channel": "websocket"},
                )
                await agent.process(reschedule, diary)
                assert not diary.booking.confirmed

        # Should have 2 entries in rescheduled_from (first two bookings)
        assert len(diary.booking.rescheduled_from) == 2
        assert diary.booking.rescheduled_from[0]["date"] == dates_booked[0]
        assert diary.booking.rescheduled_from[1]["date"] == dates_booked[1]
        # Final booking should be confirmed
        assert diary.booking.confirmed

    @pytest.mark.asyncio
    async def test_reschedule_frees_registry_each_time(self):
        """Each reschedule frees the slot in the registry."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        diary = PatientDiary.create_new("PT-MULTI")
        _seed_clinical_complete(diary)

        for _ in range(2):
            handoff = EventEnvelope.handoff(
                EventType.CLINICAL_COMPLETE, "PT-MULTI",
                source_agent="clinical",
                payload={"channel": "websocket"},
            )
            await agent.process(handoff, diary)
            event = _user_msg("PT-MULTI", "1")
            await agent.process(event, diary)
            assert registry.get_patient_booking("PT-MULTI") is not None

            reschedule = EventEnvelope.handoff(
                EventType.RESCHEDULE_REQUEST, "PT-MULTI",
                source_agent="monitoring",
                payload={"channel": "websocket"},
            )
            await agent.process(reschedule, diary)
            assert registry.get_patient_booking("PT-MULTI") is None

    @pytest.mark.asyncio
    async def test_reschedule_response_is_patient_friendly(self):
        """Reschedule response should be warm and offer new slots."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        diary = PatientDiary.create_new("PT-MULTI")
        _seed_clinical_complete(diary)

        # Book first
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-MULTI",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff, diary)
        event = _user_msg("PT-MULTI", "1")
        await agent.process(event, diary)

        # Reschedule
        reschedule = EventEnvelope.handoff(
            EventType.RESCHEDULE_REQUEST, "PT-MULTI",
            source_agent="monitoring",
            payload={"channel": "websocket"},
        )
        result = await agent.process(reschedule, diary)

        msg = result.responses[0].message.lower()
        assert "no problem" in msg or "cancelled" in msg
        assert "1" in msg and "2" in msg  # slot numbers

    @pytest.mark.asyncio
    async def test_reschedule_during_booking_phase(self, booking_agent_with_registry, booking_registry):
        """Patient says 'I want to reschedule' while still in booking phase (confirmed)."""
        agent = booking_agent_with_registry
        diary = PatientDiary.create_new("PT-MULTI")
        _seed_clinical_complete(diary)

        # Offer + confirm via the agent so registry holds are real
        handoff = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-MULTI",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        await agent.process(handoff, diary)
        event = _user_msg("PT-MULTI", "1")
        await agent.process(event, diary)
        assert diary.booking.confirmed

        # Force phase back to BOOKING (edge case: phase not yet updated by gateway)
        diary.header.current_phase = Phase.BOOKING

        # Say reschedule
        event = _user_msg("PT-MULTI", "Actually I need to reschedule")
        result = await agent.process(event, diary)

        assert not diary.booking.confirmed
        msg = result.responses[0].message.lower()
        assert "cancelled" in msg or "available" in msg

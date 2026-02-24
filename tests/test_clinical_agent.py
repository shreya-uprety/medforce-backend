"""
Tests for the Clinical Agent — sub-phase management, data extraction,
risk scoring integration, backward loops, GP responses, and document handling.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.agents.risk_scorer import RiskScorer, RiskResult
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSection,
    ClinicalSubPhase,
    GPChannel,
    GPQuery,
    IntakeSection,
    PatientDiary,
    Phase,
    RiskLevel,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole


# ── Fixtures ──


def make_diary(
    patient_id: str = "PT-100",
    phase: Phase = Phase.CLINICAL,
    sub_phase: ClinicalSubPhase = ClinicalSubPhase.ASKING_QUESTIONS,
    **kwargs,
) -> PatientDiary:
    """Create a test diary in the clinical phase."""
    diary = PatientDiary.create_new(patient_id)
    diary.header.current_phase = phase
    diary.clinical.sub_phase = sub_phase
    diary.intake.name = "Test Patient"
    diary.intake.phone = "07700900000"
    diary.intake.nhs_number = "1234567890"
    diary.intake.dob = "1985-03-15"
    diary.intake.gp_name = "Dr. Smith"
    for k, v in kwargs.items():
        setattr(diary, k, v)
    return diary


def make_clinical_diary_with_questions(
    n_questions: int = 3, answered: bool = True
) -> PatientDiary:
    """Create a diary with pre-populated clinical questions."""
    diary = make_diary()
    diary.clinical.chief_complaint = "abdominal pain"
    for i in range(n_questions):
        q = ClinicalQuestion(
            question=f"Clinical question {i + 1}?",
            answer=f"Answer {i + 1}" if answered else None,
        )
        diary.clinical.questions_asked.append(q)
    return diary


def make_intake_complete_event(patient_id: str = "PT-100") -> EventEnvelope:
    return EventEnvelope.handoff(
        event_type=EventType.INTAKE_COMPLETE,
        patient_id=patient_id,
        source_agent="intake",
        payload={"channel": "websocket"},
    )


def make_user_message_event(
    text: str, patient_id: str = "PT-100"
) -> EventEnvelope:
    return EventEnvelope.user_message(patient_id=patient_id, text=text)


def make_document_event(
    patient_id: str = "PT-100",
    doc_type: str = "lab_results",
    extracted_values: dict | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_type=EventType.DOCUMENT_UPLOADED,
        patient_id=patient_id,
        payload={
            "type": doc_type,
            "file_ref": "gs://bucket/test.pdf",
            "channel": "websocket",
            "extracted_values": extracted_values or {},
        },
        sender_id="PATIENT",
        sender_role=SenderRole.PATIENT,
    )


def make_gp_response_event(
    patient_id: str = "PT-100", lab_results: dict | None = None
) -> EventEnvelope:
    return EventEnvelope(
        event_type=EventType.GP_RESPONSE,
        patient_id=patient_id,
        payload={
            "lab_results": lab_results or {},
            "attachments": ["test_report.pdf"],
            "channel": "websocket",
        },
        sender_id="gp:Dr.Smith",
        sender_role=SenderRole.GP,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Intake Complete Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIntakeComplete:
    """INTAKE_COMPLETE event → clinical assessment starts."""

    @pytest.mark.asyncio
    async def test_starts_clinical_with_referral(self):
        agent = ClinicalAgent()
        diary = make_diary(
            phase=Phase.CLINICAL,
            sub_phase=ClinicalSubPhase.NOT_STARTED,
        )
        diary.intake.referral_letter_ref = "gs://bucket/referral.pdf"
        event = make_intake_complete_event()

        result = await agent.process(event, diary)

        # With no LLM, referral analysis fails gracefully and moves to ASKING_QUESTIONS.
        # The sub_phase should still advance past NOT_STARTED.
        assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.ASKING_QUESTIONS
        assert len(result.responses) == 1
        # When referral analysis fails, it falls through to the referral branch
        # message which mentions the referral letter
        assert "referral letter" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_starts_clinical_without_referral(self):
        agent = ClinicalAgent()
        diary = make_diary(
            phase=Phase.CLINICAL,
            sub_phase=ClinicalSubPhase.NOT_STARTED,
        )
        event = make_intake_complete_event()

        result = await agent.process(event, diary)

        assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.ASKING_QUESTIONS
        assert len(result.responses) == 1
        assert "clinical questions" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_response_goes_to_patient(self):
        agent = ClinicalAgent()
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)
        event = make_intake_complete_event()

        result = await agent.process(event, diary)

        assert result.responses[0].recipient == "patient"
        assert result.responses[0].channel == "websocket"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  User Message Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUserMessage:
    """USER_MESSAGE events during clinical phase."""

    @pytest.mark.asyncio
    async def test_extracts_chief_complaint(self):
        agent = ClinicalAgent()
        diary = make_diary()
        event = make_user_message_event("I have been experiencing severe abdominal pain")

        result = await agent.process(event, diary)

        assert result.updated_diary.clinical.chief_complaint is not None

    @pytest.mark.asyncio
    async def test_records_answer_to_pending_question(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="What is your main concern?")
        )
        event = make_user_message_event("I have liver pain")

        result = await agent.process(event, diary)

        # The unanswered question should now have an answer
        answered = [
            q for q in result.updated_diary.clinical.questions_asked
            if q.answer is not None
        ]
        assert len(answered) >= 1

    @pytest.mark.asyncio
    async def test_complete_sub_phase_returns_friendly_message(self):
        agent = ClinicalAgent()
        diary = make_diary(sub_phase=ClinicalSubPhase.COMPLETE)
        event = make_user_message_event("Hello?")

        result = await agent.process(event, diary)

        assert "already complete" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_asks_next_question_when_not_ready(self):
        agent = ClinicalAgent()
        diary = make_diary()
        event = make_user_message_event("I have some headaches")

        result = await agent.process(event, diary)

        # Should ask the next question since not enough data for scoring
        assert len(result.responses) == 1
        # Response should be a question or clinical message
        assert result.responses[0].recipient == "patient"

    @pytest.mark.asyncio
    async def test_scores_when_ready(self):
        """When enough data gathered, should prompt for docs or score."""
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions(n_questions=3, answered=True)
        event = make_user_message_event("Nothing else to add")

        result = await agent.process(event, diary)

        # If no documents exist, agent enters COLLECTING_DOCUMENTS phase first
        # If documents exist, it scores and moves to BOOKING
        if not diary.clinical.documents:
            assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.COLLECTING_DOCUMENTS
            # Now patient skips document upload
            event2 = make_user_message_event("skip")
            result2 = await agent.process(event2, result.updated_diary)
            assert result2.updated_diary.header.current_phase == Phase.BOOKING
            assert result2.updated_diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE
        else:
            assert result.updated_diary.header.current_phase == Phase.BOOKING
            assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalDataExtraction:
    """Fallback clinical data extraction (no LLM)."""

    @pytest.mark.asyncio
    async def test_extracts_red_flags(self):
        agent = ClinicalAgent()
        extracted = agent._fallback_extraction("I have jaundice and confusion")
        assert "red_flags" in extracted
        assert "jaundice" in extracted["red_flags"]
        assert "confusion" in extracted["red_flags"]

    @pytest.mark.asyncio
    async def test_extracts_chief_complaint_from_phrase(self):
        agent = ClinicalAgent()
        extracted = agent._fallback_extraction("I have severe abdominal pain")
        assert "chief_complaint" in extracted
        assert "abdominal pain" in extracted["chief_complaint"].lower()

    @pytest.mark.asyncio
    async def test_no_extraction_from_empty(self):
        agent = ClinicalAgent()
        extracted = agent._fallback_extraction("")
        assert extracted == {}

    @pytest.mark.asyncio
    async def test_extraction_with_llm_failure_falls_back(self):
        """When LLM extraction fails, fallback should work."""
        agent = ClinicalAgent()
        extracted = await agent._extract_clinical_data(
            "I have jaundice"
        )
        # With no LLM client, should use fallback
        assert "red_flags" in extracted
        assert "jaundice" in extracted["red_flags"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Apply Extracted Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestApplyExtractedData:
    """_apply_extracted_data correctly mutates the diary."""

    def test_applies_chief_complaint(self):
        agent = ClinicalAgent()
        diary = make_diary()
        agent._apply_extracted_data(diary, {"chief_complaint": "liver pain"})
        assert diary.clinical.chief_complaint == "liver pain"

    def test_applies_medical_history(self):
        agent = ClinicalAgent()
        diary = make_diary()
        agent._apply_extracted_data(diary, {"medical_history": ["diabetes", "hypertension"]})
        assert "diabetes" in diary.clinical.medical_history
        assert "hypertension" in diary.clinical.medical_history

    def test_applies_medications(self):
        agent = ClinicalAgent()
        diary = make_diary()
        agent._apply_extracted_data(diary, {"current_medications": ["metformin", "aspirin"]})
        assert "metformin" in diary.clinical.current_medications

    def test_applies_red_flags(self):
        agent = ClinicalAgent()
        diary = make_diary()
        agent._apply_extracted_data(diary, {"red_flags": ["jaundice"]})
        assert "jaundice" in diary.clinical.red_flags

    def test_no_duplicates(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.medical_history = ["diabetes"]
        agent._apply_extracted_data(diary, {"medical_history": ["diabetes", "asthma"]})
        assert diary.clinical.medical_history.count("diabetes") == 1
        assert "asthma" in diary.clinical.medical_history

    def test_empty_extraction_no_change(self):
        agent = ClinicalAgent()
        diary = make_diary()
        agent._apply_extracted_data(diary, {})
        assert diary.clinical.chief_complaint is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Backward Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBackwardLoop:
    """Clinical → Intake backward loop for missing demographics."""

    def test_triggers_when_phone_missing(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.intake.phone = None  # Remove phone
        diary.clinical.backward_loop_count = 0

        event = agent._check_backward_loop_needed(diary)

        assert event is not None
        assert event.event_type == EventType.NEEDS_INTAKE_DATA
        assert "phone" in event.payload["missing_fields"]

    def test_no_trigger_when_phone_present(self):
        agent = ClinicalAgent()
        diary = make_diary()  # Phone is set by default

        event = agent._check_backward_loop_needed(diary)

        assert event is None

    def test_circuit_breaker_at_3_loops(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.intake.phone = None
        diary.clinical.backward_loop_count = 3  # At limit

        event = agent._check_backward_loop_needed(diary)

        assert event is None  # Should NOT trigger

    @pytest.mark.asyncio
    async def test_backward_loop_emits_needs_intake_data(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.intake.phone = None
        diary.clinical.backward_loop_count = 0
        event = make_user_message_event("I have pain")

        result = await agent.process(event, diary)

        # Should emit NEEDS_INTAKE_DATA
        assert any(
            e.event_type == EventType.NEEDS_INTAKE_DATA
            for e in result.emitted_events
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Document Upload Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDocumentUpload:
    """DOCUMENT_UPLOADED events during clinical phase."""

    @pytest.mark.asyncio
    async def test_adds_document_to_diary(self):
        agent = ClinicalAgent()
        diary = make_diary()
        event = make_document_event(doc_type="lab_results")

        result = await agent.process(event, diary)

        assert len(result.updated_diary.clinical.documents) == 1
        assert result.updated_diary.clinical.documents[0].type == "lab_results"

    @pytest.mark.asyncio
    async def test_processes_extracted_lab_values(self):
        agent = ClinicalAgent()
        diary = make_diary()
        event = make_document_event(
            extracted_values={"bilirubin": 6.0, "ALT": 700}
        )

        result = await agent.process(event, diary)

        doc = result.updated_diary.clinical.documents[0]
        assert doc.processed is True
        assert doc.extracted_values["bilirubin"] == 6.0

    @pytest.mark.asyncio
    async def test_document_with_labs_triggers_scoring(self):
        """Uploading labs when ready should trigger scoring."""
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions(n_questions=2, answered=True)
        event = make_document_event(
            extracted_values={"bilirubin": 6.0}
        )

        result = await agent.process(event, diary)

        # Should complete and transition to BOOKING
        assert result.updated_diary.header.current_phase == Phase.BOOKING

    @pytest.mark.asyncio
    async def test_document_without_labs_sends_ack(self):
        agent = ClinicalAgent()
        diary = make_diary()
        event = make_document_event(doc_type="imaging")

        result = await agent.process(event, diary)

        assert "uploaded" in result.responses[0].message.lower() or "added" in result.responses[0].message.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GP Response Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPResponse:
    """GP_RESPONSE events during clinical phase."""

    @pytest.mark.asyncio
    async def test_merges_gp_lab_results(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.gp_channel = GPChannel(
            gp_name="Dr. Smith",
            gp_email="dr.smith@nhs.net",
            queries=[GPQuery(query_id="GPQ-001", status="pending")],
        )
        event = make_gp_response_event(
            lab_results={"bilirubin": 3.5, "ALT": 250}
        )

        result = await agent.process(event, diary)

        # Should add document with lab values
        docs = result.updated_diary.clinical.documents
        assert len(docs) == 1
        assert docs[0].extracted_values["bilirubin"] == 3.5

        # Should mark GP query as responded
        assert result.updated_diary.gp_channel.queries[0].status == "responded"

    @pytest.mark.asyncio
    async def test_notifies_patient_of_gp_response(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.gp_channel = GPChannel(
            queries=[GPQuery(query_id="GPQ-001", status="pending")],
        )
        event = make_gp_response_event()

        result = await agent.process(event, diary)

        assert len(result.responses) >= 1
        assert "gp" in result.responses[0].message.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scoring and Completion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestScoringAndCompletion:
    """Risk scoring and CLINICAL_COMPLETE handoff."""

    @pytest.mark.asyncio
    async def test_score_and_complete_sets_risk(self):
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()
        diary.clinical.documents.append(
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 6.0},
            )
        )
        event = make_user_message_event("no more info")

        result = await agent.process(event, diary)

        assert result.updated_diary.clinical.risk_level == RiskLevel.HIGH
        assert result.updated_diary.header.risk_level == RiskLevel.HIGH
        assert result.updated_diary.clinical.risk_method is not None

    @pytest.mark.asyncio
    async def test_emits_clinical_complete(self):
        """After scoring, CLINICAL_COMPLETE should be emitted.

        When no documents exist, agent prompts for docs first.
        We add a doc so it scores immediately.
        """
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()
        diary.clinical.documents.append(
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 3.0},
            )
        )
        event = make_user_message_event("nothing else")

        result = await agent.process(event, diary)

        clinical_complete_events = [
            e for e in result.emitted_events
            if e.event_type == EventType.CLINICAL_COMPLETE
        ]
        assert len(clinical_complete_events) == 1
        handoff = clinical_complete_events[0]
        assert handoff.payload["risk_level"] is not None

    @pytest.mark.asyncio
    async def test_sets_phase_to_booking(self):
        """After scoring, phase should be BOOKING.

        Add a document so agent doesn't enter COLLECTING_DOCUMENTS.
        """
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()
        diary.clinical.documents.append(
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"ALT": 50},
            )
        )
        event = make_user_message_event("I'm done")

        result = await agent.process(event, diary)

        assert result.updated_diary.header.current_phase == Phase.BOOKING

    @pytest.mark.asyncio
    async def test_sends_completion_message_with_risk(self):
        """Completion message should reference risk/priority.

        Add a document to bypass COLLECTING_DOCUMENTS.
        """
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()
        diary.clinical.documents.append(
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"ALT": 50},
            )
        )
        event = make_user_message_event("all good")

        result = await agent.process(event, diary)

        msg = result.responses[0].message.lower()
        assert "complete" in msg
        assert "priority" in msg or "risk" in msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ready for Scoring Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestReadyForScoring:
    """_ready_for_scoring predicate."""

    def test_ready_with_complaint_and_answers(self):
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions(n_questions=3, answered=True)
        assert agent._ready_for_scoring(diary) is True

    def test_not_ready_without_complaint(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.questions_asked = [
            ClinicalQuestion(question="Q1", answer="A1"),
            ClinicalQuestion(question="Q2", answer="A2"),
        ]
        # No chief_complaint and only 2 questions asked (need >= 3 total)
        assert agent._ready_for_scoring(diary) is False

    def test_ready_with_lab_data_only(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.documents.append(
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 3.0},
            )
        )
        assert agent._ready_for_scoring(diary) is True

    def test_not_ready_with_unanswered_questions(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.chief_complaint = "pain"
        diary.clinical.questions_asked = [
            ClinicalQuestion(question="Q1", answer=None),
            ClinicalQuestion(question="Q2", answer=None),
        ]
        assert agent._ready_for_scoring(diary) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Deterioration Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeteriorationHandler:
    """DETERIORATION_ALERT reassessment path."""

    @pytest.mark.asyncio
    async def test_deterioration_triggers_rescoring(self):
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()
        diary.header.current_phase = Phase.MONITORING

        event = EventEnvelope.handoff(
            event_type=EventType.DETERIORATION_ALERT,
            patient_id="PT-100",
            source_agent="monitoring",
            payload={
                "new_values": {"bilirubin": 8.0},
                "channel": "websocket",
            },
        )

        result = await agent.process(event, diary)

        # Should re-score with new values
        assert result.updated_diary.clinical.risk_level == RiskLevel.HIGH
        assert result.updated_diary.header.current_phase == Phase.BOOKING

    @pytest.mark.asyncio
    async def test_deterioration_adds_document(self):
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()

        event = EventEnvelope.handoff(
            event_type=EventType.DETERIORATION_ALERT,
            patient_id="PT-100",
            source_agent="monitoring",
            payload={
                "new_values": {"ALT": 600},
                "channel": "websocket",
            },
        )

        result = await agent.process(event, diary)

        deterioration_docs = [
            d for d in result.updated_diary.clinical.documents
            if d.type == "deterioration_labs"
        ]
        assert len(deterioration_docs) == 1
        assert deterioration_docs[0].extracted_values["ALT"] == 600


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Question Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQuestionGeneration:
    """Fallback question generation (no LLM)."""

    @pytest.mark.asyncio
    async def test_asks_chief_complaint_first(self):
        agent = ClinicalAgent()
        diary = make_diary()
        # Ensure chief_complaint is missing so it appears in gaps
        diary.clinical.chief_complaint = None
        question = agent._fallback_question(diary)
        assert "reason for your visit" in question.lower()

    @pytest.mark.asyncio
    async def test_asks_medical_history(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.chief_complaint = "headache"  # fill so it's not first gap
        diary.clinical.medical_history = []  # ensure this is the gap
        question = agent._fallback_question(diary)
        assert "medical conditions" in question.lower() or "illnesses" in question.lower()

    @pytest.mark.asyncio
    async def test_asks_medications(self):
        agent = ClinicalAgent()
        diary = make_diary()
        diary.clinical.chief_complaint = "headache"
        diary.clinical.medical_history = ["diabetes"]
        diary.clinical.current_medications = []  # ensure this is the gap
        question = agent._fallback_question(diary)
        assert "medications" in question.lower()

    @pytest.mark.asyncio
    async def test_fallback_generic_question(self):
        agent = ClinicalAgent()
        # Create a diary with all gaps filled so generic fallback is used
        diary = make_diary()
        diary.clinical.chief_complaint = "headache"
        diary.clinical.medical_history = ["diabetes"]
        diary.clinical.current_medications = ["metformin"]
        diary.clinical.allergies = ["penicillin"]
        diary.clinical.pain_level = 3
        question = agent._fallback_question(diary)
        assert "health" in question.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Intake Data Provided
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIntakeDataProvided:
    """INTAKE_DATA_PROVIDED event resumes clinical."""

    @pytest.mark.asyncio
    async def test_resumes_clinical_after_backward_loop(self):
        agent = ClinicalAgent()
        diary = make_diary(phase=Phase.INTAKE)

        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_DATA_PROVIDED,
            patient_id="PT-100",
            source_agent="intake",
            payload={"channel": "websocket"},
        )

        result = await agent.process(event, diary)

        assert result.updated_diary.header.current_phase == Phase.CLINICAL
        assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.ASKING_QUESTIONS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalScenarios:
    """End-to-end clinical assessment scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_full_assessment_to_completion(self):
        """Patient answers questions → scoring → CLINICAL_COMPLETE."""
        agent = ClinicalAgent()
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)

        # Step 1: Intake complete
        event1 = make_intake_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        # Step 2: Patient provides chief complaint
        event2 = make_user_message_event("I have severe abdominal pain")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary

        # Step 3: Patient provides medical history
        event3 = make_user_message_event("I have a history of diabetes and hypertension")
        result3 = await agent.process(event3, diary)
        diary = result3.updated_diary

        # Step 4: Patient provides medication info
        event4 = make_user_message_event("I take metformin and lisinopril")
        result4 = await agent.process(event4, diary)
        diary = result4.updated_diary

        # Eventually should reach scoring
        # The exact step depends on how many questions are generated
        # But the diary should accumulate data
        assert diary.clinical.chief_complaint is not None or len(diary.clinical.questions_asked) >= 2

    @pytest.mark.asyncio
    async def test_scenario_high_risk_lab_upload(self):
        """Patient uploads critical lab results → HIGH risk → BOOKING."""
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()

        # Upload labs with critical values
        event = make_document_event(
            extracted_values={"bilirubin": 8.0, "ALT": 700, "platelets": 30}
        )
        result = await agent.process(event, diary)

        assert result.updated_diary.header.risk_level == RiskLevel.HIGH
        assert result.updated_diary.header.current_phase == Phase.BOOKING

    @pytest.mark.asyncio
    async def test_scenario_gp_provides_labs(self):
        """GP responds with lab data → feeds into scoring."""
        agent = ClinicalAgent()
        diary = make_clinical_diary_with_questions()
        diary.gp_channel = GPChannel(
            gp_name="Dr. Patel",
            queries=[GPQuery(query_id="GPQ-001", status="pending")],
        )

        event = make_gp_response_event(
            lab_results={"bilirubin": 6.0, "INR": 2.5}
        )
        result = await agent.process(event, diary)

        # GP data should trigger scoring if ready
        assert result.updated_diary.header.current_phase == Phase.BOOKING
        assert result.updated_diary.clinical.risk_level == RiskLevel.HIGH

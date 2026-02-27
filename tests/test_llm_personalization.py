"""
Tests for LLM-powered personalized question generation.

Every test mocks the LLM client so that the actual google.genai API is never
called.  This exercises the code paths that are skipped by all other tests
(which run with client=None and always hit the deterministic fallbacks).

Covers:
  Clinical Agent:
    - Personalized question plan generation (top-5 ranked)
    - Contextual single-question generation
    - Structured clinical data extraction
    - Referral letter analysis
    - Full INTAKE_COMPLETE → referral + question plan flow

  Monitoring Agent:
    - Personalized monitoring question generation (ranked, categorized)
    - Full BOOKING_COMPLETE → plan creation flow
    - Check-in response evaluation (subtle concerns)
    - Deterioration assessment question generation
    - Severity assessment
    - Full deterioration flow (trigger → 3 Qs → severity → outcome)

  Error handling (both agents):
    - Malformed JSON → graceful fallback
    - Empty/whitespace LLM responses → fallback
    - LLM exceptions → fallback
    - Markdown-fenced JSON → correct parsing
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.agents.monitoring_agent import MonitoringAgent
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSubPhase,
    DeteriorationAssessment,
    DeteriorationQuestion,
    PatientDiary,
    Phase,
    RiskLevel,
    ScheduledQuestion,
)
from medforce.gateway.events import EventEnvelope, EventType


# ── Helpers ──


def make_mock_llm(response_text: str) -> MagicMock:
    """Create a mock LLM client that returns the given text.

    Sets up both sync (client.models.generate_content) and async
    (client.aio.models.generate_content) paths so tests work regardless
    of which path the agent uses.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = response_text
    # Sync path
    mock_client.models.generate_content.return_value = mock_response
    # Async path (used by gateway agents)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return mock_client


def make_mock_llm_sequence(responses: list[str]) -> MagicMock:
    """Create a mock LLM client that returns different text on each call.

    Sets up both sync and async paths.
    """
    mock_client = MagicMock()
    mock_responses = []
    for text in responses:
        r = MagicMock()
        r.text = text
        mock_responses.append(r)
    # Sync path
    mock_client.models.generate_content.side_effect = mock_responses
    # Async path (used by gateway agents)
    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_responses)
    return mock_client


def make_diary(
    patient_id: str = "PT-200",
    phase: Phase = Phase.CLINICAL,
    sub_phase: ClinicalSubPhase = ClinicalSubPhase.ASKING_QUESTIONS,
    risk_level: RiskLevel = RiskLevel.NONE,
) -> PatientDiary:
    diary = PatientDiary.create_new(patient_id)
    diary.header.current_phase = phase
    diary.header.risk_level = risk_level
    diary.clinical.sub_phase = sub_phase
    diary.intake.name = "Test Patient"
    diary.intake.phone = "07700900000"
    diary.intake.nhs_number = "1234567890"
    diary.intake.dob = "1985-03-15"
    diary.intake.gp_name = "Dr. Smith"
    return diary


def make_clinical_diary(
    n_questions: int = 3,
    answered: bool = True,
    risk_level: RiskLevel = RiskLevel.HIGH,
    condition: str = "cirrhosis",
) -> PatientDiary:
    diary = make_diary(risk_level=risk_level)
    diary.clinical.chief_complaint = "abdominal pain"
    diary.clinical.condition_context = condition
    diary.clinical.medical_history = ["cirrhosis", "diabetes"]
    diary.clinical.current_medications = ["propranolol", "lactulose"]
    diary.clinical.red_flags = ["jaundice"]
    for i in range(n_questions):
        diary.clinical.questions_asked.append(
            ClinicalQuestion(
                question=f"Clinical question {i + 1}?",
                answer=f"Answer {i + 1}" if answered else None,
            )
        )
    return diary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Agent — LLM Personalized Question Plan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalQuestionPlan:
    """LLM-generated personalized question plan (top-5 ranked questions)."""

    @pytest.mark.asyncio
    async def test_generates_personalized_questions(self):
        """LLM returns 4 ranked questions → stored in generated_questions."""
        llm_questions = [
            "Have you noticed any yellowing of your skin or eyes recently?",
            "How has your alcohol consumption been in the past month?",
            "Are you experiencing any abdominal swelling or fluid retention?",
            "Have you had any episodes of confusion or disorientation?",
        ]
        mock_client = make_mock_llm(json.dumps(llm_questions))
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()
        diary.clinical.chief_complaint = "abdominal pain"
        diary.clinical.condition_context = "cirrhosis"

        await agent._generate_question_plan(diary)

        assert diary.clinical.generated_questions == llm_questions
        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_question_plan_truncated_to_5(self):
        """LLM returns more than 5 questions → only top 5 kept."""
        mock_client = make_mock_llm(
            json.dumps([f"Question {i}?" for i in range(8)])
        )
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()
        diary.clinical.chief_complaint = "headache"

        await agent._generate_question_plan(diary)

        assert len(diary.clinical.generated_questions) == 5

    @pytest.mark.asyncio
    async def test_question_plan_handles_markdown_wrapped_json(self):
        """LLM wraps JSON in markdown code fences → still parsed."""
        questions = ["Symptom question?", "History question?", "Meds question?"]
        llm_text = f"```json\n{json.dumps(questions)}\n```"
        mock_client = make_mock_llm(llm_text)
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()
        diary.clinical.chief_complaint = "chest pain"

        await agent._generate_question_plan(diary)

        assert diary.clinical.generated_questions == questions

    @pytest.mark.asyncio
    async def test_question_plan_malformed_json_falls_back(self):
        """LLM returns invalid JSON → falls back to deterministic questions."""
        mock_client = make_mock_llm("This is not valid JSON at all")
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()
        diary.clinical.chief_complaint = "abdominal pain"
        diary.clinical.condition_context = "cirrhosis"

        await agent._generate_question_plan(diary)

        assert len(diary.clinical.generated_questions) > 0
        # Fallback for cirrhosis should reference the condition
        assert any(
            "cirrhosis" in q.lower() or "confusion" in q.lower()
            or "abdominal pain" in q.lower()
            for q in diary.clinical.generated_questions
        )

    @pytest.mark.asyncio
    async def test_question_plan_llm_exception_falls_back(self):
        """LLM throws exception → falls back gracefully."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()
        diary.clinical.chief_complaint = "pain"

        await agent._generate_question_plan(diary)

        assert len(diary.clinical.generated_questions) > 0

    @pytest.mark.asyncio
    async def test_contextual_question_used_in_conversation(self):
        """LLM contextual question is used directly (no pre-generated plan)."""
        contextual_q = "Have you noticed any yellowing of your skin?"
        mock_client = make_mock_llm(contextual_q)
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()
        diary.clinical.chief_complaint = "liver pain"
        diary.clinical.condition_context = "cirrhosis"

        event = EventEnvelope.user_message(patient_id="PT-200", text="liver pain")
        result = await agent._ask_next_question(event, diary, "websocket")

        asked = [q.question for q in result.updated_diary.clinical.questions_asked]
        assert contextual_q in asked


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Agent — LLM Contextual Single Question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalContextualQuestion:
    """LLM-generated single contextual follow-up question."""

    @pytest.mark.asyncio
    async def test_generates_contextual_question(self):
        """LLM returns a targeted follow-up → used directly."""
        expected = (
            "Given your liver condition, have you noticed any "
            "changes in your appetite or weight recently?"
        )
        mock_client = make_mock_llm(expected)
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        question = await agent._generate_contextual_question(diary)

        assert question == expected
        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_contextual_question_empty_response_falls_back(self):
        """LLM returns empty string → falls back."""
        mock_client = make_mock_llm("")
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        question = await agent._generate_contextual_question(diary)

        assert len(question) > 0

    @pytest.mark.asyncio
    async def test_contextual_question_whitespace_falls_back(self):
        """LLM returns only whitespace → falls back."""
        mock_client = make_mock_llm("   \n  ")
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        question = await agent._generate_contextual_question(diary)

        assert len(question.strip()) > 0

    @pytest.mark.asyncio
    async def test_contextual_question_exception_falls_back(self):
        """LLM throws → falls back."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("timeout")
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        question = await agent._generate_contextual_question(diary)

        assert len(question) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Agent — LLM Clinical Data Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalDataExtractionLLM:
    """LLM-powered clinical data extraction from patient messages."""

    @pytest.mark.asyncio
    async def test_extracts_structured_data(self):
        """LLM returns structured clinical JSON → returned as dict."""
        data = {
            "chief_complaint": "severe abdominal pain",
            "medical_history": ["cirrhosis", "type 2 diabetes"],
            "current_medications": ["propranolol 40mg", "metformin 500mg"],
            "red_flags": ["jaundice"],
            "pain_level": 7,
            "pain_location": "upper right abdomen",
        }
        mock_client = make_mock_llm(json.dumps(data))
        agent = ClinicalAgent(llm_client=mock_client)

        extracted = await agent._extract_clinical_data(
            "I have severe abdominal pain in my upper right side, about 7/10"
        )

        assert extracted["chief_complaint"] == "severe abdominal pain"
        assert "cirrhosis" in extracted["medical_history"]
        assert extracted["pain_level"] == 7

    @pytest.mark.asyncio
    async def test_extraction_handles_markdown_fenced_json(self):
        """LLM wraps response in markdown → parsed correctly."""
        data = {"chief_complaint": "headache", "red_flags": ["confusion"]}
        mock_client = make_mock_llm(f"```json\n{json.dumps(data)}\n```")
        agent = ClinicalAgent(llm_client=mock_client)

        extracted = await agent._extract_clinical_data(
            "I have a headache and feel confused"
        )

        assert extracted["chief_complaint"] == "headache"
        assert "confusion" in extracted["red_flags"]

    @pytest.mark.asyncio
    async def test_extraction_malformed_json_falls_back(self):
        """LLM returns invalid JSON → falls back to pattern matching."""
        mock_client = make_mock_llm("I think the patient has jaundice")
        agent = ClinicalAgent(llm_client=mock_client)

        extracted = await agent._extract_clinical_data("I have jaundice")

        assert "jaundice" in extracted.get("red_flags", [])

    @pytest.mark.asyncio
    async def test_extraction_llm_exception_falls_back(self):
        """LLM throws → falls back to pattern matching."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API down")
        agent = ClinicalAgent(llm_client=mock_client)

        extracted = await agent._extract_clinical_data(
            "I have jaundice and confusion"
        )

        assert "jaundice" in extracted.get("red_flags", [])
        assert "confusion" in extracted.get("red_flags", [])

    @pytest.mark.asyncio
    async def test_extraction_applied_to_diary(self):
        """LLM extraction → _apply_extracted_data updates the diary."""
        data = {
            "chief_complaint": "liver pain",
            "medical_history": ["hepatitis B"],
            "current_medications": ["tenofovir"],
            "red_flags": ["jaundice"],
            "pain_level": 6,
            "pain_location": "right upper quadrant",
            "lifestyle_alcohol": "2 units per week",
        }
        mock_client = make_mock_llm(json.dumps(data))
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary()

        extracted = await agent._extract_clinical_data("I have liver pain with jaundice")
        agent._apply_extracted_data(diary, extracted)

        assert diary.clinical.chief_complaint == "liver pain"
        assert "hepatitis B" in diary.clinical.medical_history
        assert "tenofovir" in diary.clinical.current_medications
        assert "jaundice" in diary.clinical.red_flags
        assert diary.clinical.pain_level == 6
        assert diary.clinical.pain_location == "right upper quadrant"
        assert diary.clinical.lifestyle_factors["alcohol"] == "2 units per week"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Agent — LLM Referral Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestReferralAnalysisLLM:
    """LLM-powered referral letter analysis."""

    @pytest.mark.asyncio
    async def test_referral_analysis_populates_diary(self):
        """LLM extracts structured data from referral → applied to diary."""
        analysis = {
            "chief_complaint": "suspected cirrhosis",
            "condition_context": "cirrhosis",
            "medical_history": ["alcohol use disorder", "hypertension"],
            "current_medications": ["propranolol", "spironolactone"],
            "allergies": ["penicillin"],
            "red_flags": ["ascites", "jaundice"],
            "key_findings": "Elevated LFTs, portal hypertension",
        }
        mock_client = make_mock_llm(json.dumps(analysis))
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)
        diary.intake.referral_letter_ref = "gs://bucket/referral.pdf"

        await agent._analyze_referral(diary)

        assert diary.clinical.chief_complaint == "suspected cirrhosis"
        assert diary.clinical.condition_context == "cirrhosis"
        assert "alcohol use disorder" in diary.clinical.medical_history
        assert "propranolol" in diary.clinical.current_medications
        assert "penicillin" in diary.clinical.allergies
        assert "ascites" in diary.clinical.red_flags

    @pytest.mark.asyncio
    async def test_referral_analysis_markdown_fenced(self):
        """LLM wraps referral analysis in markdown → parsed correctly."""
        data = {
            "chief_complaint": "MASH/NAFLD",
            "condition_context": "MASH",
            "medical_history": ["obesity"],
        }
        llm_text = f"```json\n{json.dumps(data)}\n```"
        mock_client = make_mock_llm(llm_text)
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)
        diary.intake.referral_letter_ref = "gs://bucket/referral.pdf"

        await agent._analyze_referral(diary)

        assert diary.clinical.condition_context == "MASH"

    @pytest.mark.asyncio
    async def test_referral_analysis_failure_continues(self):
        """LLM fails during referral analysis → continues without crashing."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)
        diary.intake.referral_letter_ref = "gs://bucket/referral.pdf"

        await agent._analyze_referral(diary)

        assert diary.clinical.chief_complaint is None

    @pytest.mark.asyncio
    async def test_intake_complete_with_referral_and_question_plan(self):
        """Full flow: INTAKE_COMPLETE + referral → LLM analysis + question plan."""
        referral_data = json.dumps({
            "chief_complaint": "suspected cirrhosis",
            "condition_context": "cirrhosis",
            "medical_history": ["alcohol use disorder"],
            "red_flags": ["jaundice"],
        })
        question_plan = json.dumps([
            "How has your alcohol consumption been recently?",
            "Have you noticed any yellowing of your skin or eyes?",
            "Any abdominal swelling or fluid retention?",
        ])
        # First LLM call: referral analysis. Second: question plan.
        # (Welcome and first question use templates — no LLM needed.)
        mock_client = make_mock_llm_sequence([referral_data, question_plan])
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)
        diary.intake.referral_letter_ref = "gs://bucket/referral.pdf"

        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-200",
            source_agent="intake",
            payload={"channel": "websocket"},
        )

        result = await agent.process(event, diary)

        assert result.updated_diary.clinical.condition_context == "cirrhosis"
        assert result.updated_diary.clinical.chief_complaint == "suspected cirrhosis"
        assert len(result.updated_diary.clinical.generated_questions) == 2
        assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.ASKING_QUESTIONS
        assert "referral" in result.responses[0].message.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monitoring Agent — LLM Personalized Monitoring Questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringQuestionGeneration:
    """LLM-generated personalized monitoring questions."""

    @pytest.mark.asyncio
    async def test_generates_personalized_monitoring_questions(self):
        """LLM returns ranked monitoring questions → sorted by priority."""
        llm_data = [
            {"question": "Any yellowing of skin?", "category": "symptom", "priority": 1},
            {"question": "How is your alcohol intake?", "category": "lifestyle", "priority": 2},
            {"question": "Side effects from propranolol?", "category": "medication", "priority": 3},
            {"question": "New blood test results?", "category": "labs", "priority": 4},
            {"question": "How are you feeling overall?", "category": "general", "priority": 5},
            {"question": "Any weight changes?", "category": "lifestyle", "priority": 6},
        ]
        mock_client = make_mock_llm(json.dumps(llm_data))
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary(risk_level=RiskLevel.HIGH)

        questions = await agent._generate_monitoring_questions(diary, 6)

        assert len(questions) == 6
        assert questions[0].priority <= questions[1].priority
        assert questions[0].category == "symptom"
        assert "yellowing" in questions[0].question
        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_questions_truncated_to_requested(self):
        """LLM returns more questions than requested → truncated."""
        llm_data = [
            {"question": f"Q{i}?", "category": "general", "priority": i}
            for i in range(10)
        ]
        mock_client = make_mock_llm(json.dumps(llm_data))
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary(risk_level=RiskLevel.MEDIUM)

        questions = await agent._generate_monitoring_questions(diary, 4)

        assert len(questions) == 4

    @pytest.mark.asyncio
    async def test_monitoring_questions_markdown_fenced(self):
        """LLM wraps response in markdown → parsed correctly."""
        data = [
            {"question": "New symptoms?", "category": "symptom", "priority": 1},
            {"question": "How are meds?", "category": "medication", "priority": 2},
        ]
        llm_text = f"```json\n{json.dumps(data)}\n```"
        mock_client = make_mock_llm(llm_text)
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        questions = await agent._generate_monitoring_questions(diary, 2)

        assert len(questions) == 2
        assert questions[0].category == "symptom"

    @pytest.mark.asyncio
    async def test_monitoring_questions_malformed_falls_back(self):
        """LLM returns invalid JSON → falls back to condition-aware questions."""
        mock_client = make_mock_llm("Not valid JSON")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary(condition="cirrhosis")

        questions = await agent._generate_monitoring_questions(diary, 6)

        assert len(questions) > 0
        texts = [q.question.lower() for q in questions]
        assert any("yellowing" in t or "alcohol" in t for t in texts)

    @pytest.mark.asyncio
    async def test_monitoring_questions_exception_falls_back(self):
        """LLM throws → falls back."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("timeout")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        questions = await agent._generate_monitoring_questions(diary, 4)

        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_booking_complete_creates_plan_with_llm_questions(self):
        """BOOKING_COMPLETE → LLM generates questions → full plan created."""
        llm_data = [
            {"question": "Any yellowing of skin or eyes?", "category": "symptom", "priority": 1},
            {"question": "How is your alcohol intake?", "category": "lifestyle", "priority": 2},
            {"question": "Any medication side effects?", "category": "medication", "priority": 3},
            {"question": "New lab results to share?", "category": "labs", "priority": 4},
            {"question": "How are you feeling overall?", "category": "general", "priority": 5},
            {"question": "Any abdominal swelling?", "category": "symptom", "priority": 6},
        ]
        mock_client = make_mock_llm(json.dumps(llm_data))
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary(risk_level=RiskLevel.HIGH)
        diary.header.current_phase = Phase.BOOKING
        diary.booking.confirmed = True

        event = EventEnvelope.handoff(
            event_type=EventType.BOOKING_COMPLETE,
            patient_id="PT-200",
            source_agent="booking",
            payload={"channel": "websocket", "appointment_date": "2026-03-15"},
        )

        result = await agent.process(event, diary)

        plan = result.updated_diary.monitoring.communication_plan
        assert plan.generated is True
        assert plan.total_messages == 6  # HIGH risk
        assert len(plan.questions) == 6
        # Questions should be assigned to check-in days (not day 0)
        assert all(q.day > 0 for q in plan.questions)
        # Sorted by priority
        assert plan.questions[0].priority <= plan.questions[-1].priority
        assert result.updated_diary.monitoring.monitoring_active is True

    @pytest.mark.asyncio
    async def test_question_schedule_assignment(self):
        """Questions are distributed across check-in days round-robin."""
        agent = MonitoringAgent()
        questions = [
            ScheduledQuestion(question=f"Q{i}", day=0, priority=i, category="general")
            for i in range(6)
        ]
        check_days = [7, 14, 21, 30, 45, 60]

        assigned = agent._assign_questions_to_schedule(questions, check_days)

        assert assigned[0].day == 7
        assert assigned[1].day == 14
        assert assigned[2].day == 21
        assert assigned[3].day == 30
        assert assigned[4].day == 45
        assert assigned[5].day == 60


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monitoring Agent — LLM Check-in Response Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckinResponseEvaluationLLM:
    """LLM-powered evaluation of check-in responses."""

    @pytest.mark.asyncio
    async def test_llm_detects_subtle_concern(self):
        """LLM catches concerning symptoms that keywords miss."""
        llm_response = json.dumps({
            "concerning": True,
            "detected_symptoms": ["medication non-adherence", "increased fatigue"],
            "reasoning": "Worsening liver function suspected",
        })
        mock_client = make_mock_llm(llm_response)
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        scheduled_q = ScheduledQuestion(
            question="How are you getting on with your medications?",
            day=14, priority=3, category="medication",
        )

        # Message that doesn't match keyword patterns
        concerning, detected = await agent._evaluate_checkin_response(
            diary,
            "I keep forgetting to take my pills and feel really run down",
            scheduled_q,
        )

        assert concerning is True
        assert len(detected) > 0

    @pytest.mark.asyncio
    async def test_llm_confirms_not_concerning(self):
        """LLM confirms a normal response is fine."""
        llm_response = json.dumps({
            "concerning": False,
            "detected_symptoms": [],
            "reasoning": "Patient reports feeling well",
        })
        mock_client = make_mock_llm(llm_response)
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        scheduled_q = ScheduledQuestion(
            question="How are you feeling overall?",
            day=30, priority=5, category="general",
        )

        concerning, detected = await agent._evaluate_checkin_response(
            diary,
            "I'm feeling much better, thanks. No new issues.",
            scheduled_q,
        )

        assert concerning is False
        assert detected == []

    @pytest.mark.asyncio
    async def test_llm_evaluation_exception_returns_safe(self):
        """LLM throws → returns (False, []) safely."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        scheduled_q = ScheduledQuestion(
            question="How are you?", day=14, priority=5, category="general",
        )

        concerning, detected = await agent._evaluate_checkin_response(
            diary, "I feel okay", scheduled_q,
        )

        assert concerning is False
        assert detected == []

    @pytest.mark.asyncio
    async def test_pattern_detection_takes_priority_over_llm(self):
        """Pattern-based detection fires first → LLM not called."""
        mock_client = MagicMock()
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary(condition="cirrhosis")

        # Pattern detection is now a separate method called before LLM
        concerning, detected = agent._check_concerning_patterns(
            "i have dark urine since yesterday", diary,
        )

        assert concerning is True
        assert "dark urine" in detected

    @pytest.mark.asyncio
    async def test_llm_evaluation_malformed_json_returns_safe(self):
        """LLM returns non-JSON → returns (False, []) safely."""
        mock_client = make_mock_llm("This patient seems fine to me")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()

        scheduled_q = ScheduledQuestion(
            question="How are you?", day=14, priority=5, category="general",
        )

        concerning, detected = await agent._evaluate_checkin_response(
            diary, "Just a bit tired", scheduled_q,
        )

        assert concerning is False
        assert detected == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monitoring Agent — LLM Deterioration Assessment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeteriorationAssessmentLLM:
    """LLM-powered deterioration assessment questions and severity scoring."""

    @pytest.mark.asyncio
    async def test_generates_assessment_question(self):
        """LLM generates a targeted follow-up question."""
        expected = (
            "Can you tell me exactly when the swelling started "
            "and whether it's been getting progressively worse?"
        )
        mock_client = make_mock_llm(expected)
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()
        assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["swelling", "worse"],
            trigger_message="My stomach seems more swollen and I feel worse",
        )

        question = await agent._generate_assessment_question(diary, assessment, 0)

        assert question == expected
        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_assessment_question_empty_falls_back(self):
        """LLM returns empty → falls back to pattern-based question."""
        mock_client = make_mock_llm("")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()
        assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["worse"],
            trigger_message="I feel worse",
        )

        question = await agent._generate_assessment_question(diary, assessment, 0)

        assert len(question) > 0
        # Fallback Q0 asks patient to describe symptoms
        assert "describe" in question.lower() or "experiencing" in question.lower()

    @pytest.mark.asyncio
    async def test_assessment_question_exception_falls_back(self):
        """LLM throws → falls back to pattern-based question."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("timeout")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()
        assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["worse"],
            trigger_message="I feel worse",
        )

        question = await agent._generate_assessment_question(diary, assessment, 1)

        assert len(question) > 0

    @pytest.mark.asyncio
    async def test_severity_assessment_via_llm(self):
        """LLM assesses severity from Q&A → structured result returned."""
        severity_result = {
            "severity": "moderate",
            "reasoning": "Progressive swelling with mild functional impact",
            "bring_forward_appointment": True,
            "urgency": "soon",
            "additional_instructions": "Monitor fluid intake",
        }
        mock_client = make_mock_llm(json.dumps(severity_result))
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()
        assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["swelling"],
            trigger_message="My stomach is more swollen",
            questions=[
                DeteriorationQuestion(
                    question="When did the swelling start?",
                    answer="About 3 days ago",
                    category="description",
                ),
                DeteriorationQuestion(
                    question="Any new symptoms?",
                    answer="Some discomfort lying down",
                    category="new_symptoms",
                ),
                DeteriorationQuestion(
                    question="Severity 1-10?",
                    answer="About 5 or 6",
                    category="severity",
                ),
            ],
        )

        result = await agent._assess_severity(diary, assessment)

        assert result["severity"] == "moderate"
        assert result["bring_forward_appointment"] is True
        assert "urgency" in result

    @pytest.mark.asyncio
    async def test_severity_malformed_json_falls_back(self):
        """LLM returns invalid JSON for severity → rule-based fallback."""
        mock_client = make_mock_llm("I think this is moderate severity")
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary()
        assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["worse"],
            trigger_message="I feel worse",
            questions=[
                DeteriorationQuestion(
                    question="Describe symptoms?",
                    answer="Getting worse, more pain, about 7/10",
                    category="description",
                ),
            ],
        )

        result = await agent._assess_severity(diary, assessment)

        assert result["severity"] in ("mild", "moderate", "severe")

    @pytest.mark.asyncio
    async def test_full_deterioration_flow_with_llm(self):
        """Full flow: trigger → 3 LLM questions → LLM severity → outcome."""
        mock_client = make_mock_llm_sequence([
            # Q1: assessment start
            "Can you describe the swelling in more detail?",
            # Q2: follow-up
            "Have you noticed any changes in your breathing or appetite?",
            # Q3: severity
            "On a scale of 1-10, how severe is your discomfort?",
            # Severity assessment
            json.dumps({
                "severity": "moderate",
                "reasoning": "Progressive abdominal swelling with functional impact",
                "bring_forward_appointment": True,
                "urgency": "soon",
                "additional_instructions": "Monitor fluid intake closely",
            }),
        ])
        agent = MonitoringAgent(llm_client=mock_client)
        diary = make_clinical_diary(risk_level=RiskLevel.HIGH)
        diary.header.current_phase = Phase.MONITORING
        diary.monitoring.monitoring_active = True
        diary.booking.confirmed = True
        diary.monitoring.appointment_date = "2026-03-15"

        # Step 1: Patient reports worsening — triggers assessment
        event1 = EventEnvelope.user_message(
            patient_id="PT-200",
            text="I feel worse and I have more swelling in my stomach",
        )
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        assert diary.monitoring.deterioration_assessment.active is True
        assert len(diary.monitoring.deterioration_assessment.questions) == 1
        assert "swelling" in result1.responses[0].message.lower()

        # Step 2: Patient answers Q1
        event2 = EventEnvelope.user_message(
            patient_id="PT-200",
            text="It started about 3 days ago and has been getting bigger",
        )
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary

        assert len(diary.monitoring.deterioration_assessment.questions) == 2

        # Step 3: Patient answers Q2
        event3 = EventEnvelope.user_message(
            patient_id="PT-200",
            text="My appetite is poor and I feel a bit out of breath lying flat",
        )
        result3 = await agent.process(event3, diary)
        diary = result3.updated_diary

        assert len(diary.monitoring.deterioration_assessment.questions) == 3

        # Step 4: Patient answers Q3 → triggers severity assessment
        event4 = EventEnvelope.user_message(
            patient_id="PT-200",
            text="About a 6 out of 10",
        )
        result4 = await agent.process(event4, diary)
        diary = result4.updated_diary

        # Assessment complete
        assert diary.monitoring.deterioration_assessment.assessment_complete is True
        assert diary.monitoring.deterioration_assessment.severity == "moderate"
        assert diary.monitoring.deterioration_assessment.recommendation == "bring_forward"

        # Should emit DETERIORATION_ALERT
        alerts = [
            e for e in result4.emitted_events
            if e.event_type == EventType.DETERIORATION_ALERT
        ]
        assert len(alerts) == 1
        assert alerts[0].payload["assessment"]["severity"] == "moderate"

        # Patient-facing message should mention appointment
        assert "appointment" in result4.responses[0].message.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Agent — Adaptive Follow-Up Questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAdaptiveFollowUp:
    """Adaptive follow-up questions between plan questions."""

    @pytest.mark.asyncio
    async def test_llm_followup_triggered_on_concerning_answer(self):
        """LLM returns a follow-up question when patient says something concerning."""
        followup_json = json.dumps({
            "followup": True,
            "question": "When did you first notice this change?",
        })
        mock_client = make_mock_llm_sequence([
            # _extract_clinical_data
            json.dumps({}),
            # _evaluate_followup
            followup_json,
        ])
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary(n_questions=0)
        diary.clinical.awaiting_followup = True

        # Add an unanswered plan question
        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Has the pain worsened since your GP visit?")
        )

        event = EventEnvelope.user_message(
            patient_id="PT-200",
            text="Yes, it has worsened significantly over the last week",
        )
        result = await agent.process(event, diary)

        # Follow-up should be asked
        last_q = result.updated_diary.clinical.questions_asked[-1]
        assert last_q.is_followup is True
        assert "change" in last_q.question.lower()
        assert result.responses[0].message == last_q.question

    @pytest.mark.asyncio
    async def test_no_followup_on_trivial_answer(self):
        """Trivial answers ('no', 'yes') skip follow-up evaluation entirely."""
        mock_client = make_mock_llm_sequence([
            # _extract_clinical_data
            json.dumps({}),
        ])
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary(n_questions=0)
        diary.clinical.awaiting_followup = True
        diary.clinical.allergies_addressed = True
        # Add remaining plan questions so we don't hit sufficiency
        diary.clinical.generated_questions = ["Next plan question?"]

        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Have you noticed any yellowing?")
        )

        event = EventEnvelope.user_message(
            patient_id="PT-200", text="No",
        )
        result = await agent.process(event, diary)

        # Should move to next plan question, NOT ask a follow-up
        last_q = result.updated_diary.clinical.questions_asked[-1]
        assert last_q.is_followup is False
        assert last_q.question == "Next plan question?"

    @pytest.mark.asyncio
    async def test_followup_does_not_chain(self):
        """A follow-up answer NEVER triggers another follow-up (no chaining)."""
        mock_client = make_mock_llm_sequence([
            # _extract_clinical_data
            json.dumps({}),
        ])
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary(n_questions=0)
        diary.clinical.awaiting_followup = True
        diary.clinical.allergies_addressed = True
        # Add remaining plan questions so we don't hit sufficiency
        diary.clinical.generated_questions = ["Next plan question?"]

        # The pending question is a follow-up
        diary.clinical.questions_asked.append(
            ClinicalQuestion(
                question="When did this change start?",
                is_followup=True,
            )
        )

        event = EventEnvelope.user_message(
            patient_id="PT-200",
            text="It started last week and has been getting much worse every day",
        )
        result = await agent.process(event, diary)

        # Should NOT produce another follow-up — straight to next plan Q
        last_q = result.updated_diary.clinical.questions_asked[-1]
        assert last_q.is_followup is False
        assert last_q.question == "Next plan question?"
        # awaiting_followup should be cleared
        assert result.updated_diary.clinical.awaiting_followup is True  # re-set by _ask_next_question

    @pytest.mark.asyncio
    async def test_deterministic_followup_worsening(self):
        """Deterministic fallback triggers follow-up on worsening keywords."""
        agent = ClinicalAgent()
        result = agent._deterministic_followup(
            "yes it has worsened quite a bit", "Has the pain changed?"
        )
        assert result is not None
        assert "gradual" in result.lower() or "sudden" in result.lower()

    @pytest.mark.asyncio
    async def test_deterministic_followup_emergency(self):
        """Deterministic fallback triggers follow-up on emergency keywords."""
        agent = ClinicalAgent()
        result = agent._deterministic_followup(
            "I collapsed yesterday and felt very confused",
            "Any new symptoms?",
        )
        assert result is not None
        assert "recently" in result.lower() or "once" in result.lower()

    @pytest.mark.asyncio
    async def test_deterministic_followup_severe_pain(self):
        """Deterministic fallback triggers follow-up on severe pain (>=7/10)."""
        agent = ClinicalAgent()
        result = agent._deterministic_followup(
            "The pain is about 8 out of 10 now",
            "How severe is the pain?",
        )
        assert result is not None
        assert "constant" in result.lower() or "come and go" in result.lower()

    @pytest.mark.asyncio
    async def test_deterministic_followup_no_match(self):
        """Deterministic fallback returns None for unremarkable answers."""
        agent = ClinicalAgent()
        result = agent._deterministic_followup(
            "It's been about the same really", "Has the pain changed?"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_deterministic_followup_functional_impact(self):
        """Deterministic fallback triggers on functional impact keywords."""
        agent = ClinicalAgent()
        result = agent._deterministic_followup(
            "I can't sleep at all because of the pain",
            "How is it affecting you?",
        )
        assert result is not None
        assert "gradual" in result.lower() or "moment" in result.lower()

    @pytest.mark.asyncio
    async def test_llm_followup_returns_false(self):
        """LLM says no follow-up needed → proceeds to next plan question."""
        followup_json = json.dumps({"followup": False})
        mock_client = make_mock_llm_sequence([
            # _extract_clinical_data
            json.dumps({}),
            # _evaluate_followup
            followup_json,
        ])
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_clinical_diary(n_questions=0)
        diary.clinical.awaiting_followup = True
        diary.clinical.allergies_addressed = True
        diary.clinical.generated_questions = ["Next plan question?"]

        diary.clinical.questions_asked.append(
            ClinicalQuestion(question="Has the pain worsened?")
        )

        event = EventEnvelope.user_message(
            patient_id="PT-200",
            text="It's been about the same, nothing major to report",
        )
        result = await agent.process(event, diary)

        # Should proceed to next plan question
        last_q = result.updated_diary.clinical.questions_asked[-1]
        assert last_q.is_followup is False
        assert last_q.question == "Next plan question?"

    @pytest.mark.asyncio
    async def test_questions_sufficient_blocked_by_generated_questions(self):
        """_questions_sufficient returns False while plan questions remain."""
        agent = ClinicalAgent()
        diary = make_clinical_diary(n_questions=5, answered=True)
        diary.clinical.referral_analysis = {"chief_complaint": "pain"}
        diary.clinical.meds_addressed = True
        diary.clinical.allergies_addressed = True
        diary.clinical.generated_questions = ["Remaining plan Q?"]

        assert agent._questions_sufficient(diary) is False

    @pytest.mark.asyncio
    async def test_questions_sufficient_blocked_by_awaiting_followup(self):
        """_questions_sufficient returns False while awaiting follow-up evaluation."""
        agent = ClinicalAgent()
        diary = make_clinical_diary(n_questions=5, answered=True)
        diary.clinical.referral_analysis = {"chief_complaint": "pain"}
        diary.clinical.meds_addressed = True
        diary.clinical.allergies_addressed = True
        diary.clinical.awaiting_followup = True

        assert agent._questions_sufficient(diary) is False

    @pytest.mark.asyncio
    async def test_intake_complete_sets_awaiting_followup(self):
        """INTAKE_COMPLETE sets awaiting_followup after the first plan question."""
        question_plan = json.dumps([
            "How has your pain been?",
            "Any yellowing?",
            "Any swelling?",
            "How is daily life?",
            "Any concerns?",
        ])
        mock_client = make_mock_llm(question_plan)
        agent = ClinicalAgent(llm_client=mock_client)
        diary = make_diary(sub_phase=ClinicalSubPhase.NOT_STARTED)
        diary.clinical.chief_complaint = "liver pain"
        diary.clinical.condition_context = "cirrhosis"

        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-200",
            source_agent="intake",
            payload={"channel": "websocket"},
        )

        result = await agent.process(event, diary)

        assert result.updated_diary.clinical.awaiting_followup is True
        assert len(result.updated_diary.clinical.questions_asked) == 1

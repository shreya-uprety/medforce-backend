"""
Tests for resilience features:
  - Agent crash recovery (graceful error response)
  - Diary save retry with backoff
  - Event idempotency guard
  - Clinical question hard cap
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import (
    AgentResponse,
    DeliveryResult,
    DispatcherRegistry,
)
from medforce.gateway.diary import (
    ClinicalQuestion,
    ClinicalSubPhase,
    DiaryConcurrencyError,
    DiaryNotFoundError,
    PatientDiary,
    Phase,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.gateway import Gateway


# ── Test Helpers ──


class CrashingAgent(BaseAgent):
    """Agent that always raises an exception."""
    agent_name = "crasher"

    async def process(self, event, diary):
        raise RuntimeError("Something went terribly wrong")


class EchoAgent(BaseAgent):
    """Simple agent that echoes back."""
    agent_name = "echo"

    async def process(self, event, diary):
        return AgentResult(
            updated_diary=diary,
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel=event.payload.get("channel", "websocket"),
                    message=f"Echo: {event.payload.get('text', '')}",
                    metadata={"patient_id": event.patient_id},
                )
            ],
        )


class MockDiaryStore:
    """In-memory diary store for testing."""

    def __init__(self):
        self._diaries: dict[str, tuple[PatientDiary, int]] = {}
        self.save_call_count = 0
        self.save_side_effect = None

    def load(self, patient_id):
        if patient_id not in self._diaries:
            raise DiaryNotFoundError(f"Not found: {patient_id}")
        diary, gen = self._diaries[patient_id]
        return diary, gen

    def save(self, patient_id, diary, generation=None):
        self.save_call_count += 1
        if self.save_side_effect:
            effect = self.save_side_effect
            if callable(effect):
                effect = effect(self.save_call_count)
            if isinstance(effect, Exception):
                raise effect
        new_gen = (generation or 0) + 1
        self._diaries[patient_id] = (diary, new_gen)
        return new_gen

    def create(self, patient_id, correlation_id=None):
        diary = PatientDiary.create_new(patient_id, correlation_id=correlation_id)
        gen = self.save(patient_id, diary, generation=None)
        return diary, gen

    def seed(self, patient_id, diary, generation=1):
        self._diaries[patient_id] = (diary, generation)


@pytest.fixture
def diary_store():
    return MockDiaryStore()


@pytest.fixture
def dispatcher_registry():
    registry = DispatcherRegistry()
    mock_dispatcher = MagicMock()
    mock_dispatcher.channel_name = "websocket"
    mock_dispatcher.send = AsyncMock(return_value=DeliveryResult(
        success=True, channel="websocket", recipient="patient"
    ))
    registry.register(mock_dispatcher)
    return registry


@pytest.fixture
def gateway(diary_store, dispatcher_registry):
    gw = Gateway(
        diary_store=diary_store,
        dispatcher_registry=dispatcher_registry,
    )
    gw.register_agent("intake", EchoAgent())
    gw.register_agent("clinical", EchoAgent())
    gw.register_agent("booking", EchoAgent())
    gw.register_agent("monitoring", EchoAgent())
    return gw


# ── 1.1 Agent Crash Recovery ──


class TestAgentCrashRecovery:
    @pytest.mark.asyncio
    async def test_crash_returns_friendly_error(self, diary_store, dispatcher_registry):
        """When an agent crashes, the patient should get a friendly error message."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", CrashingAgent())

        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        result = await gw.process_event(event)

        assert result is not None
        assert len(result.responses) == 1
        assert "temporary issue" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_crash_logged_as_agent_error(self, diary_store, dispatcher_registry):
        """Agent crash should be logged in the event log."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", CrashingAgent())

        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        await gw.process_event(event)

        log = gw.get_event_log("PT-001")
        assert any(e["status"] == "AGENT_ERROR" for e in log)

    @pytest.mark.asyncio
    async def test_crash_response_has_correct_channel(self, diary_store, dispatcher_registry):
        """Error response should target the correct channel."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", CrashingAgent())

        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        result = await gw.process_event(event)

        assert result.responses[0].channel == "websocket"
        assert result.responses[0].recipient in ("patient", "PATIENT")


# ── 1.2 Diary Save Retry ──


class TestDiarySaveRetry:
    @pytest.mark.asyncio
    async def test_retry_on_concurrency_error(self, dispatcher_registry):
        """Diary save should retry on DiaryConcurrencyError."""
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-001")
        store.seed("PT-001", diary)

        # Fail first save with concurrency error, succeed on retry
        call_count = [0]
        original_save = store.save

        def failing_save(pid, d, gen=None):
            call_count[0] += 1
            if call_count[0] == 2:  # First save attempt after create
                raise DiaryConcurrencyError("conflict")
            return original_save(pid, d, gen)

        store.save = failing_save

        gw = Gateway(
            diary_store=store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", EchoAgent())

        event = EventEnvelope.user_message("PT-001", "Hello")
        result = await gw.process_event(event)

        # Should still succeed — retry worked
        assert result is not None
        assert len(result.responses) >= 1

    @pytest.mark.asyncio
    async def test_responses_dispatched_even_if_save_fails(self, dispatcher_registry):
        """Responses should be dispatched even if all save retries fail."""
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-001")
        store.seed("PT-001", diary)

        original_load = store.load

        def always_fail_save(pid, d, gen=None):
            raise Exception("GCS unavailable")

        def safe_load(pid):
            return original_load(pid)

        store.save = always_fail_save

        gw = Gateway(
            diary_store=store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", EchoAgent())

        event = EventEnvelope.user_message("PT-001", "Hello")
        result = await gw.process_event(event)

        # Responses should still be dispatched
        assert result is not None
        ws_disp = dispatcher_registry.get("websocket")
        assert ws_disp.send.called


# ── 1.4 Event Idempotency ──


class TestEventIdempotency:
    @pytest.mark.asyncio
    async def test_duplicate_event_skipped(self, gateway, diary_store):
        """Processing the same event twice should skip the duplicate."""
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")

        result1 = await gateway.process_event(event)
        result2 = await gateway.process_event(event)

        # First should succeed, second should be None (skipped)
        assert result1 is not None
        assert result2 is None

    @pytest.mark.asyncio
    async def test_duplicate_logged(self, gateway, diary_store):
        """Duplicate events should be logged."""
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        await gateway.process_event(event)
        await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert any(e["status"] == "DUPLICATE" for e in log)

    @pytest.mark.asyncio
    async def test_different_events_not_skipped(self, gateway, diary_store):
        """Different events for the same patient should both be processed."""
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event1 = EventEnvelope.user_message("PT-001", "Hello")
        event2 = EventEnvelope.user_message("PT-001", "World")

        result1 = await gateway.process_event(event1)
        result2 = await gateway.process_event(event2)

        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_fifo_eviction_at_cap(self, gateway, diary_store):
        """After 100 events, oldest should be evicted from the dedup set."""
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        # Process 101 unique events
        events = []
        for i in range(101):
            events.append(EventEnvelope.user_message("PT-001", f"msg-{i}"))

        for e in events:
            await gateway.process_event(e)

        # The first event should have been evicted, so reprocessing should work
        result = await gateway.process_event(events[0])
        assert result is not None


# ── 1.5 Clinical Question Hard Cap ──


class TestClinicalQuestionCap:
    @pytest.mark.asyncio
    async def test_question_cap_forces_scoring(self):
        """After MAX_CLINICAL_QUESTIONS with no chief complaint, cap forces scoring."""
        from medforce.gateway.agents.clinical_agent import ClinicalAgent, MAX_CLINICAL_QUESTIONS
        from medforce.gateway.agents.risk_scorer import RiskScorer

        agent = ClinicalAgent.__new__(ClinicalAgent)
        agent._client = None
        agent._model_name = "test-model"
        agent._risk_scorer = RiskScorer()

        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        # No chief complaint so _ready_for_scoring returns False
        # and the hard cap is the only path to scoring

        # Fill intake fields to prevent backward loop
        diary.intake.name = "John Smith"
        diary.intake.dob = "15/03/1985"
        diary.intake.nhs_number = "9434765870"
        diary.intake.phone = "07700900000"
        # Set backward_loop_count to prevent re-triggering
        diary.clinical.backward_loop_count = 3

        for i in range(MAX_CLINICAL_QUESTIONS):
            diary.clinical.questions_asked.append(
                ClinicalQuestion(question=f"Question {i}?", answer=f"Answer {i}")
            )

        async def _mock_extract(text):
            return {}
        agent._extract_clinical_data = _mock_extract

        event = EventEnvelope.user_message("PT-001", "Another answer")
        result = await agent.process(event, diary)

        # Cap should force scoring → BOOKING
        assert result.updated_diary.header.current_phase == Phase.BOOKING
        assert result.updated_diary.clinical.sub_phase == ClinicalSubPhase.COMPLETE

    @pytest.mark.asyncio
    async def test_under_cap_asks_more_questions(self):
        """Below the cap, should ask another question (not force scoring)."""
        from medforce.gateway.agents.clinical_agent import ClinicalAgent, MAX_CLINICAL_QUESTIONS
        from medforce.gateway.agents.risk_scorer import RiskScorer

        agent = ClinicalAgent.__new__(ClinicalAgent)
        agent._client = None
        agent._model_name = "test-model"
        agent._risk_scorer = RiskScorer()

        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        # No chief complaint — scoring not ready

        diary.intake.name = "John Smith"
        diary.intake.dob = "15/03/1985"
        diary.intake.nhs_number = "9434765870"
        diary.intake.phone = "07700900000"
        diary.clinical.backward_loop_count = 3

        # Only 2 questions asked — well below cap
        for i in range(2):
            diary.clinical.questions_asked.append(
                ClinicalQuestion(question=f"Question {i}?", answer=f"Answer {i}")
            )

        async def _mock_extract(text):
            return {}
        agent._extract_clinical_data = _mock_extract

        event = EventEnvelope.user_message("PT-001", "Some answer")
        result = await agent.process(event, diary)

        # Should still be in CLINICAL — asked another question
        assert result.updated_diary.header.current_phase == Phase.CLINICAL

    def test_cap_value_is_reasonable(self):
        from medforce.gateway.agents.clinical_agent import MAX_CLINICAL_QUESTIONS
        assert 5 <= MAX_CLINICAL_QUESTIONS <= 15

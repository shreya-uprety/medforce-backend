"""
P2-P4 Tests — Covering:
  P2 #8-10: Health check, observability metrics, dead letter queue
  P3 #11-12: Lab value validation, document deduplication
  P4 #14,16: LLM retry enhancement, input boundary/truncation

All agents run with llm_client=None (deterministic fallback mode).
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from medforce.gateway.agents.booking_agent import BookingAgent
from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.agents.monitoring_agent import MonitoringAgent
from medforce.gateway.agents.risk_scorer import RiskScorer
from medforce.gateway.agents.base_agent import AgentResult
from medforce.gateway.agents.llm_utils import llm_generate
from medforce.gateway.channels import (
    AgentResponse,
    DeliveryResult,
    DispatcherRegistry,
)
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalSubPhase,
    CommunicationPlan,
    DiaryNotFoundError,
    MonitoringEntry,
    PatientDiary,
    Phase,
    RiskLevel,
    ScheduledQuestion,
    SlotOption,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.gateway import Gateway, MAX_MESSAGE_LENGTH


# ── Shared Helpers ──


def _user_msg(patient_id: str, text: str) -> EventEnvelope:
    return EventEnvelope.user_message(patient_id, text)


def _seed_monitoring_diary(patient_id: str = "PT-TEST") -> PatientDiary:
    diary = PatientDiary.create_new(patient_id)
    diary.intake.responder_type = "patient"
    diary.intake.mark_field_collected("name", "Test Patient")
    diary.intake.mark_field_collected("dob", "15/03/1985")
    diary.intake.mark_field_collected("nhs_number", "1234567890")
    diary.intake.mark_field_collected("phone", "07700900123")
    diary.intake.mark_field_collected("gp_name", "Dr. Smith")
    diary.intake.mark_field_collected("contact_preference", "phone")
    diary.intake.intake_complete = True
    diary.clinical.chief_complaint = "abdominal pain"
    diary.clinical.condition_context = "cirrhosis"
    diary.clinical.pain_level = 5
    diary.clinical.red_flags = ["elevated bilirubin"]
    diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)
    diary.clinical.risk_level = RiskLevel.MEDIUM
    diary.header.risk_level = RiskLevel.MEDIUM
    diary.booking.confirmed = True
    diary.booking.slot_selected = SlotOption(date="2026-03-15", time="09:00", provider="Dr. A")
    diary.monitoring.monitoring_active = True
    diary.monitoring.appointment_date = "2026-03-15"
    diary.monitoring.baseline = {"bilirubin": 2.0, "ALT": 45}
    diary.monitoring.communication_plan = CommunicationPlan(
        risk_level="medium", total_messages=4,
        check_in_days=[14, 30, 60, 90],
        questions=[ScheduledQuestion(question="How are you?", day=14, priority=1, category="general")],
        generated=True,
    )
    diary.header.current_phase = Phase.MONITORING
    diary.header.phase_entered_at = datetime.now(timezone.utc)
    return diary


class MockDiaryStore:
    def __init__(self):
        self._diaries = {}

    def load(self, patient_id):
        if patient_id not in self._diaries:
            raise DiaryNotFoundError(f"Not found: {patient_id}")
        return self._diaries[patient_id]

    def save(self, patient_id, diary, generation=None):
        new_gen = (generation or 0) + 1
        self._diaries[patient_id] = (diary, new_gen)
        return new_gen

    def create(self, patient_id, correlation_id=None):
        diary = PatientDiary.create_new(patient_id, correlation_id=correlation_id)
        gen = self.save(patient_id, diary)
        return diary, gen

    def seed(self, patient_id, diary, generation=1):
        self._diaries[patient_id] = (diary, generation)


class CrashingAgent:
    agent_name = "crasher"
    async def process(self, event, diary):
        raise RuntimeError("Agent exploded")


class EchoAgent:
    agent_name = "echo"
    async def process(self, event, diary):
        return AgentResult(
            updated_diary=diary,
            responses=[AgentResponse(
                recipient="patient", channel="websocket",
                message=f"Echo: {event.payload.get('text', '')}",
                metadata={"patient_id": event.patient_id},
            )],
        )


@pytest.fixture
def gateway_with_agents():
    store = MockDiaryStore()
    diary = PatientDiary.create_new("PT-GW")
    store.seed("PT-GW", diary)

    registry = DispatcherRegistry()
    mock_disp = MagicMock()
    mock_disp.channel_name = "websocket"
    mock_disp.send = AsyncMock(return_value=DeliveryResult(
        success=True, channel="websocket", recipient="patient"
    ))
    registry.register(mock_disp)

    gw = Gateway(diary_store=store, dispatcher_registry=registry)
    gw.register_agent("intake", EchoAgent())
    gw.register_agent("clinical", EchoAgent())
    gw.register_agent("booking", EchoAgent())
    gw.register_agent("monitoring", EchoAgent())
    return gw, store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P2 #8: Health Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHealthCheck:
    def test_health_check_with_all_agents(self, gateway_with_agents):
        gw, _ = gateway_with_agents
        health = gw.health_check()

        assert health["healthy"] is True
        assert health["agents_registered"] is True
        assert "intake" in health["agent_names"]
        assert "clinical" in health["agent_names"]
        assert "booking" in health["agent_names"]
        assert "monitoring" in health["agent_names"]
        assert health["diary_store_available"] is True
        assert health["dlq_size"] == 0

    def test_health_check_no_agents(self):
        store = MockDiaryStore()
        registry = DispatcherRegistry()
        gw = Gateway(diary_store=store, dispatcher_registry=registry)

        health = gw.health_check()
        assert health["healthy"] is False
        assert health["agents_registered"] is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P2 #9: Observability & Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetrics:
    @pytest.mark.asyncio
    async def test_events_processed_counter(self, gateway_with_agents):
        gw, _ = gateway_with_agents
        assert gw.get_metrics()["events_processed"] == 0

        await gw.process_event(_user_msg("PT-GW", "Hello"))
        assert gw.get_metrics()["events_processed"] == 1

        await gw.process_event(_user_msg("PT-GW", "World"))
        assert gw.get_metrics()["events_processed"] == 2

    @pytest.mark.asyncio
    async def test_agent_processing_times_tracked(self, gateway_with_agents):
        gw, _ = gateway_with_agents

        await gw.process_event(_user_msg("PT-GW", "Hello"))
        metrics = gw.get_metrics()

        assert "intake" in metrics["agent_processing_summaries"]
        summary = metrics["agent_processing_summaries"]["intake"]
        assert summary["count"] == 1
        assert summary["avg_ms"] >= 0

    @pytest.mark.asyncio
    async def test_events_failed_counter(self):
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-FAIL")
        store.seed("PT-FAIL", diary)
        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", CrashingAgent())

        await gw.process_event(_user_msg("PT-FAIL", "Hello"))
        assert gw.get_metrics()["events_failed"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P2 #10: Dead Letter Queue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeadLetterQueue:
    @pytest.mark.asyncio
    async def test_agent_error_goes_to_dlq(self):
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-DLQ")
        store.seed("PT-DLQ", diary)
        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", CrashingAgent())

        await gw.process_event(_user_msg("PT-DLQ", "Hello"))

        dlq = gw.get_dlq()
        assert len(dlq) == 1
        assert dlq[0]["patient_id"] == "PT-DLQ"
        assert dlq[0]["agent"] == "intake"
        assert dlq[0]["error_type"] == "RuntimeError"
        assert "exploded" in dlq[0]["error_message"]

    @pytest.mark.asyncio
    async def test_dlq_has_traceback(self):
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-DLQ2")
        store.seed("PT-DLQ2", diary)
        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", CrashingAgent())

        await gw.process_event(_user_msg("PT-DLQ2", "Hello"))

        dlq = gw.get_dlq()
        assert len(dlq[0]["traceback"]) > 0

    @pytest.mark.asyncio
    async def test_dlq_bounded(self):
        store = MockDiaryStore()
        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", CrashingAgent())

        # Manually add 600 DLQ entries to test bounding
        for i in range(600):
            gw._dead_letter_queue.append({"id": i})

        # Next error triggers cleanup
        diary = PatientDiary.create_new("PT-DLQ3")
        store.seed("PT-DLQ3", diary)
        await gw.process_event(_user_msg("PT-DLQ3", "Hello"))

        assert len(gw._dead_letter_queue) <= 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P3 #11: Lab Value Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLabValueValidation:
    @pytest.fixture
    def agent(self):
        return MonitoringAgent(llm_client=None)

    def test_valid_values_pass_through(self, agent):
        values = {"bilirubin": 3.5, "ALT": 120, "platelets": 200}
        validated, warnings = agent._validate_lab_values(values)
        assert validated == values
        assert len(warnings) == 0

    def test_out_of_range_excluded(self, agent):
        values = {"bilirubin": 500.0, "ALT": 45}  # bilirubin > 50 is implausible
        validated, warnings = agent._validate_lab_values(values)
        assert "bilirubin" not in validated
        assert "ALT" in validated
        assert len(warnings) == 1
        assert "bilirubin" in warnings[0]

    def test_negative_values_excluded(self, agent):
        values = {"platelets": -50}  # negative platelets makes no sense
        validated, warnings = agent._validate_lab_values(values)
        assert "platelets" not in validated

    def test_zero_values_pass(self, agent):
        values = {"bilirubin": 0.0}  # 0 is within range
        validated, warnings = agent._validate_lab_values(values)
        assert "bilirubin" in validated
        assert len(warnings) == 0

    def test_unknown_params_pass_through(self, agent):
        values = {"exotic_marker": 999.9}  # no range defined
        validated, warnings = agent._validate_lab_values(values)
        assert "exotic_marker" in validated
        assert len(warnings) == 0

    def test_non_numeric_values_pass(self, agent):
        values = {"blood_type": "A+", "ALT": 45}
        validated, warnings = agent._validate_lab_values(values)
        assert validated == values
        assert len(warnings) == 0

    def test_all_implausible_returns_empty(self, agent):
        values = {"bilirubin": 999, "ALT": 99999}
        validated, warnings = agent._validate_lab_values(values)
        assert len(validated) == 0
        assert len(warnings) == 2

    @pytest.mark.asyncio
    async def test_implausible_values_flagged_in_diary(self, agent):
        """Document upload with implausible values should be flagged."""
        diary = _seed_monitoring_diary()
        event = EventEnvelope.handoff(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-TEST",
            source_agent="upload",
            payload={
                "extracted_values": {"bilirubin": 999, "ALT": 45},
                "channel": "websocket",
            },
        )
        result = agent._handle_document(event, diary)

        # Should have validation warning entries
        warn_entries = [
            e for e in diary.monitoring.entries if e.type == "lab_validation_warning"
        ]
        assert len(warn_entries) == 1
        assert any("excluded" in a.lower() for a in diary.monitoring.alerts_fired)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P3 #12: Document Deduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDocumentDeduplication:
    @pytest.fixture
    def clinical_agent(self):
        return ClinicalAgent(llm_client=None, risk_scorer=RiskScorer())

    @pytest.mark.asyncio
    async def test_first_upload_accepted(self, clinical_agent):
        diary = PatientDiary.create_new("PT-DEDUP")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.chief_complaint = "pain"
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)

        event = EventEnvelope.handoff(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-DEDUP",
            source_agent="upload",
            payload={
                "file_ref": "gs://bucket/lab.pdf",
                "type": "lab_results",
                "content_hash": "abc123def456",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        assert len(diary.clinical.documents) == 1
        assert diary.clinical.documents[0].content_hash == "abc123def456"

    @pytest.mark.asyncio
    async def test_duplicate_upload_rejected(self, clinical_agent):
        diary = PatientDiary.create_new("PT-DEDUP2")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.chief_complaint = "pain"
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)

        # First upload
        diary.clinical.documents.append(ClinicalDocument(
            type="lab_results",
            source="patient",
            file_ref="gs://bucket/lab.pdf",
            content_hash="abc123def456",
        ))

        # Same hash — should be rejected
        event = EventEnvelope.handoff(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-DEDUP2",
            source_agent="upload",
            payload={
                "file_ref": "gs://bucket/lab_copy.pdf",
                "type": "lab_results",
                "content_hash": "abc123def456",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        assert len(diary.clinical.documents) == 1  # Still only 1
        assert "already" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_different_hash_accepted(self, clinical_agent):
        diary = PatientDiary.create_new("PT-DEDUP3")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.chief_complaint = "pain"
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)

        diary.clinical.documents.append(ClinicalDocument(
            type="lab_results",
            source="patient",
            file_ref="gs://bucket/lab1.pdf",
            content_hash="hash_one",
        ))

        # Different hash — should be accepted
        event = EventEnvelope.handoff(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-DEDUP3",
            source_agent="upload",
            payload={
                "file_ref": "gs://bucket/lab2.pdf",
                "type": "lab_results",
                "content_hash": "hash_two",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        assert len(diary.clinical.documents) == 2

    @pytest.mark.asyncio
    async def test_no_hash_always_accepted(self, clinical_agent):
        """Documents without content_hash should always be accepted."""
        diary = PatientDiary.create_new("PT-DEDUP4")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.chief_complaint = "pain"
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COLLECTING_DOCUMENTS)

        event = EventEnvelope.handoff(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-DEDUP4",
            source_agent="upload",
            payload={
                "file_ref": "gs://bucket/lab.pdf",
                "type": "lab_results",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        assert len(diary.clinical.documents) == 1

    def test_has_document_hash_method(self):
        """ClinicalSection.has_document_hash works correctly."""
        from medforce.gateway.diary import ClinicalSection
        section = ClinicalSection()
        section.documents.append(ClinicalDocument(
            type="lab_results", content_hash="abc123"
        ))

        assert section.has_document_hash("abc123") is True
        assert section.has_document_hash("xyz789") is False
        assert section.has_document_hash("") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P4 #14: LLM Retry Enhancement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLLMRetryEnhancement:
    @pytest.mark.asyncio
    async def test_default_retries_is_two(self):
        """Default max_retries should now be 2 (3 total attempts)."""
        call_count = 0
        mock_client = MagicMock()
        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("LLM down")
        mock_client.aio.models.generate_content = mock_generate

        result = await llm_generate(mock_client, "test-model", "test prompt")
        assert result is None
        assert call_count == 3  # 1 original + 2 retries

    @pytest.mark.asyncio
    async def test_critical_mode_more_retries(self):
        """Critical mode should use at least 3 retries (4 total attempts)."""
        call_count = 0
        mock_client = MagicMock()
        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("LLM down")
        mock_client.aio.models.generate_content = mock_generate

        result = await llm_generate(
            mock_client, "test-model", "test prompt", critical=True
        )
        assert result is None
        assert call_count == 4  # 1 original + 3 retries

    @pytest.mark.asyncio
    async def test_success_on_retry(self):
        """Should succeed if LLM works on retry."""
        call_count = 0
        mock_client = MagicMock()
        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            response = MagicMock()
            response.text = "Success response"
            return response
        mock_client.aio.models.generate_content = mock_generate

        result = await llm_generate(mock_client, "test-model", "test prompt")
        assert result == "Success response"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_empty_response_triggers_retry(self):
        """Empty LLM responses should trigger retry."""
        call_count = 0
        mock_client = MagicMock()
        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            if call_count < 2:
                response.text = ""
            else:
                response.text = "Good response"
            return response
        mock_client.aio.models.generate_content = mock_generate

        result = await llm_generate(mock_client, "test-model", "prompt")
        assert result == "Good response"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P4 #16: Input Boundary / Truncation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInputBoundary:
    @pytest.mark.asyncio
    async def test_oversized_message_truncated(self, gateway_with_agents):
        """Messages exceeding MAX_MESSAGE_LENGTH should be truncated."""
        gw, _ = gateway_with_agents
        huge_text = "A" * (MAX_MESSAGE_LENGTH + 5000)
        event = _user_msg("PT-GW", huge_text)

        result = await gw.process_event(event)

        # The event payload should have been truncated
        assert len(event.payload["text"]) == MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_normal_message_not_truncated(self, gateway_with_agents):
        """Normal-sized messages should not be truncated."""
        gw, _ = gateway_with_agents
        normal_text = "Hello, I feel fine."
        event = _user_msg("PT-GW", normal_text)

        await gw.process_event(event)

        assert event.payload["text"] == normal_text

    @pytest.mark.asyncio
    async def test_empty_message_handled(self, gateway_with_agents):
        """Empty messages should be processed without error."""
        gw, _ = gateway_with_agents
        event = _user_msg("PT-GW", "")

        result = await gw.process_event(event)
        assert result is not None

    @pytest.mark.asyncio
    async def test_exactly_at_limit_not_truncated(self, gateway_with_agents):
        """Message exactly at MAX_MESSAGE_LENGTH should not be truncated."""
        gw, _ = gateway_with_agents
        exact_text = "B" * MAX_MESSAGE_LENGTH
        event = _user_msg("PT-GW", exact_text)

        await gw.process_event(event)

        assert event.payload["text"] == exact_text

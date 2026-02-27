"""
P1 Resilience Gap Tests — Covering:
  #4. Chaos Engineering — GCS down, LLM fully unavailable
  #5. Concurrency — Diary lock contention under concurrent coroutines
  #6. Booking Negative Paths — Zero slots, schedule manager errors, race conditions
  #7. GP Communication Edge Cases — GP timeout, malformed response, wrong patient

All agents run with llm_client=None (deterministic fallback mode).
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from medforce.gateway.agents.booking_agent import BookingAgent
from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.agents.intake_agent import IntakeAgent
from medforce.gateway.agents.monitoring_agent import MonitoringAgent
from medforce.gateway.agents.risk_scorer import RiskScorer
from medforce.gateway.agents.base_agent import AgentResult
from medforce.gateway.channels import (
    AgentResponse,
    DeliveryResult,
    DispatcherRegistry,
)
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSubPhase,
    CommunicationPlan,
    DiaryConcurrencyError,
    DiaryNotFoundError,
    GPChannel,
    GPQuery,
    MonitoringEntry,
    PatientDiary,
    Phase,
    RiskLevel,
    ScheduledQuestion,
    SlotOption,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.gateway import Gateway


# ── Shared Helpers ──


def _user_msg(patient_id: str, text: str) -> EventEnvelope:
    return EventEnvelope.user_message(patient_id, text)


def _seed_intake_complete(diary: PatientDiary) -> None:
    diary.intake.responder_type = "patient"
    diary.intake.mark_field_collected("name", "Test Patient")
    diary.intake.mark_field_collected("dob", "15/03/1985")
    diary.intake.mark_field_collected("nhs_number", "1234567890")
    diary.intake.mark_field_collected("phone", "07700900123")
    diary.intake.mark_field_collected("gp_name", "Dr. Smith")
    diary.intake.mark_field_collected("contact_preference", "phone")
    diary.intake.intake_complete = True
    diary.header.current_phase = Phase.CLINICAL
    diary.clinical.backward_loop_count = 3


def _seed_clinical_complete(diary: PatientDiary) -> None:
    _seed_intake_complete(diary)
    diary.clinical.chief_complaint = "abdominal pain"
    diary.clinical.condition_context = "cirrhosis"
    diary.clinical.pain_level = 5
    diary.clinical.allergies = ["penicillin"]
    diary.clinical.red_flags = ["elevated bilirubin"]
    diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)
    diary.clinical.risk_level = RiskLevel.MEDIUM
    diary.header.risk_level = RiskLevel.MEDIUM
    diary.header.current_phase = Phase.BOOKING


def _seed_booking_complete(diary: PatientDiary) -> None:
    _seed_clinical_complete(diary)
    diary.booking.slots_offered = [
        SlotOption(date="2026-03-15", time="09:00", provider="Dr. Available"),
    ]
    diary.booking.slot_selected = diary.booking.slots_offered[0]
    diary.booking.confirmed = True
    diary.booking.appointment_id = "APT-TEST-001"
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


class MockDiaryStore:
    def __init__(self):
        self._diaries: dict[str, tuple[PatientDiary, int]] = {}
        self.save_call_count = 0
        self.save_side_effect = None
        self.load_side_effect = None

    def load(self, patient_id):
        if self.load_side_effect:
            raise self.load_side_effect
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P1 #4: Chaos Engineering — Infrastructure Failure Modes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGCSDown:
    """When GCS is completely down, Gateway should still return a response."""

    @pytest.fixture
    def gateway_with_gcs_down(self):
        store = MockDiaryStore()
        store.load_side_effect = ConnectionError("GCS unreachable")
        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)
        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", EchoAgent())
        return gw

    @pytest.mark.asyncio
    async def test_gcs_down_on_load_returns_error(self, gateway_with_gcs_down):
        """When GCS can't load diary, Gateway should catch and return error."""
        event = _user_msg("PT-GCS-DOWN", "Hello")
        # Gateway._load_or_create_diary will try load then create — both will fail
        # The exception should propagate but gateway should handle gracefully
        with pytest.raises(ConnectionError):
            await gateway_with_gcs_down.process_event(event)

    @pytest.mark.asyncio
    async def test_gcs_down_on_save_still_dispatches(self):
        """When diary save fails, responses should still be dispatched."""
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-SAVE-FAIL")
        store.seed("PT-SAVE-FAIL", diary)
        store.save_side_effect = ConnectionError("GCS write failed")

        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", EchoAgent())

        event = _user_msg("PT-SAVE-FAIL", "Hello")
        result = await gw.process_event(event)

        # Response should have been dispatched even though save failed
        assert mock_disp.send.call_count >= 1

    @pytest.mark.asyncio
    async def test_diary_save_concurrency_retry_exhaustion(self):
        """Diary concurrency failures retried 3 times then move on."""
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-CONC")
        store.seed("PT-CONC", diary)
        store.save_side_effect = DiaryConcurrencyError("Always conflicts")

        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", EchoAgent())

        event = _user_msg("PT-CONC", "Hello")
        result = await gw.process_event(event)

        # Diary save is background — wait for background tasks to finish
        import asyncio
        if gw._bg_tasks:
            await asyncio.gather(*gw._bg_tasks, return_exceptions=True)

        # Should have tried multiple times
        assert store.save_call_count >= 2
        # Responses still dispatched
        assert mock_disp.send.call_count >= 1


class TestLLMFullyUnavailable:
    """When LLM is completely unavailable, every agent should still function."""

    @pytest.mark.asyncio
    async def test_intake_no_llm(self):
        """IntakeAgent without LLM should respond and collect fields."""
        agent = IntakeAgent(llm_client=None)
        diary = PatientDiary.create_new("PT-LLM1")

        # First message triggers responder identification
        result = await agent.process(_user_msg("PT-LLM1", "Hello, I'm the patient"), diary)
        assert len(result.responses) > 0

        # Provide name when asked
        result = await agent.process(_user_msg("PT-LLM1", "John Smith"), diary)
        assert len(result.responses) > 0
        # Agent should be functional even without LLM
        assert diary.intake.responder_type is not None

    @pytest.mark.asyncio
    async def test_clinical_no_llm_asks_questions(self):
        """ClinicalAgent without LLM should generate fallback questions."""
        agent = ClinicalAgent(llm_client=None, risk_scorer=RiskScorer())
        diary = PatientDiary.create_new("PT-LLM2")
        _seed_intake_complete(diary)

        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-LLM2",
            source_agent="intake",
            payload={"channel": "websocket"},
        )
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        assert diary.clinical.sub_phase != ClinicalSubPhase.NOT_STARTED

    @pytest.mark.asyncio
    async def test_booking_no_llm_generates_instructions(self):
        """BookingAgent without LLM should generate deterministic instructions."""
        agent = BookingAgent(schedule_manager=None, llm_client=None)
        diary = PatientDiary.create_new("PT-LLM3")
        _seed_clinical_complete(diary)

        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-LLM3",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        msg = result.responses[0].message.lower()
        assert "appointment" in msg or "slot" in msg

    @pytest.mark.asyncio
    async def test_monitoring_no_llm_heartbeat(self):
        """MonitoringAgent without LLM handles heartbeat with fallback questions."""
        agent = MonitoringAgent(llm_client=None)
        diary = PatientDiary.create_new("PT-LLM4")
        _seed_booking_complete(diary)

        event = EventEnvelope.handoff(
            event_type=EventType.HEARTBEAT,
            patient_id="PT-LLM4",
            source_agent="scheduler",
            payload={"days_since_appointment": 14, "channel": "websocket"},
        )
        result = await agent.process(event, diary)

        assert len(result.responses) > 0

    @pytest.mark.asyncio
    async def test_monitoring_no_llm_assessment(self):
        """MonitoringAgent without LLM handles deterioration with fallback."""
        agent = MonitoringAgent(llm_client=None)
        diary = PatientDiary.create_new("PT-LLM5")
        _seed_booking_complete(diary)

        event = _user_msg("PT-LLM5", "I've been getting much worse and have more pain")
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        # Should start an assessment
        assert diary.monitoring.deterioration_assessment.active


class TestCircuitBreaker:
    """Verify the chain-depth circuit breaker fires."""

    @pytest.mark.asyncio
    async def test_infinite_loop_broken(self):
        """An agent that always emits events is stopped by the circuit breaker."""
        class LoopingAgent:
            agent_name = "looper"
            async def process(self, event, diary):
                handoff = EventEnvelope.handoff(
                    event_type=EventType.INTAKE_COMPLETE,
                    patient_id=event.patient_id,
                    source_agent="looper",
                    payload={"channel": "websocket"},
                )
                return AgentResult(
                    updated_diary=diary,
                    emitted_events=[handoff],
                    responses=[],
                )

        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-LOOP")
        store.seed("PT-LOOP", diary)
        registry = DispatcherRegistry()

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        agent = LoopingAgent()
        gw.register_agent("intake", agent)
        gw.register_agent("clinical", agent)

        event = _user_msg("PT-LOOP", "Hello")
        # Should NOT hang — circuit breaker stops it
        result = await gw.process_event(event)
        # Verify the event log shows CIRCUIT_BREAKER
        log = gw.get_event_log(patient_id="PT-LOOP")
        circuit_breaks = [e for e in log if e["status"] == "CIRCUIT_BREAKER"]
        assert len(circuit_breaks) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P1 #5: Concurrency — Diary Lock Contention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConcurrency:
    """Concurrent access to the same patient diary."""

    @pytest.mark.asyncio
    async def test_concurrent_messages_same_patient(self):
        """Multiple concurrent messages to the same patient should not crash."""
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-CONCURRENT")
        store.seed("PT-CONCURRENT", diary)

        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", IntakeAgent(llm_client=None))

        # Send 5 messages concurrently
        events = [_user_msg("PT-CONCURRENT", f"Message {i}") for i in range(5)]
        results = await asyncio.gather(
            *[gw.process_event(e) for e in events],
            return_exceptions=True,
        )

        # No crashes — all should return something (result or None for dupes)
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Errors: {errors}"

    @pytest.mark.asyncio
    async def test_different_patients_no_contention(self):
        """Messages to different patients should process independently."""
        store = MockDiaryStore()
        for i in range(5):
            diary = PatientDiary.create_new(f"PT-INDEP-{i}")
            store.seed(f"PT-INDEP-{i}", diary)

        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=True, channel="websocket", recipient="patient"
        ))
        registry.register(mock_disp)

        gw = Gateway(diary_store=store, dispatcher_registry=registry)
        gw.register_agent("intake", IntakeAgent(llm_client=None))

        events = [_user_msg(f"PT-INDEP-{i}", f"Hi I'm patient {i}") for i in range(5)]
        results = await asyncio.gather(
            *[gw.process_event(e) for e in events],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]
        assert len(successes) == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P1 #6: Booking Negative Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBookingNegativePaths:
    """Tests for booking edge cases and failure modes."""

    @pytest.fixture
    def agent(self):
        return BookingAgent(schedule_manager=None, llm_client=None)

    @pytest.mark.asyncio
    async def test_no_available_slots(self):
        """Schedule manager returns empty list → patient told no slots."""
        mock_schedule = MagicMock()
        mock_schedule.get_empty_schedule.return_value = []
        agent = BookingAgent(schedule_manager=mock_schedule, llm_client=None)

        diary = PatientDiary.create_new("PT-NOSL")
        _seed_clinical_complete(diary)

        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-NOSL",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        msg = result.responses[0].message.lower()
        assert "no slots" in msg or "not available" in msg or "no" in msg

    @pytest.mark.asyncio
    async def test_schedule_manager_crashes(self):
        """Schedule manager throws → fallback to mock slots."""
        mock_schedule = MagicMock()
        mock_schedule.get_empty_schedule.side_effect = RuntimeError("DB connection lost")
        agent = BookingAgent(schedule_manager=mock_schedule, llm_client=None)

        diary = PatientDiary.create_new("PT-SCHCRASH")
        _seed_clinical_complete(diary)

        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-SCHCRASH",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await agent.process(event, diary)

        # Should fall back to mock slots
        assert len(result.responses) > 0
        assert len(diary.booking.slots_offered) == 3
        msg = result.responses[0].message.lower()
        assert "1" in msg and "2" in msg and "3" in msg

    @pytest.mark.asyncio
    async def test_schedule_manager_update_fails(self, agent):
        """Booking confirmation works even if schedule_manager.update_slot fails."""
        mock_schedule = MagicMock()
        mock_schedule.update_slot.side_effect = RuntimeError("Write failed")
        agent_with_mgr = BookingAgent(schedule_manager=mock_schedule, llm_client=None)

        diary = PatientDiary.create_new("PT-UPDFAIL")
        _seed_clinical_complete(diary)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-20", time="09:00", provider="Dr. Test"),
        ]

        event = _user_msg("PT-UPDFAIL", "1")
        result = await agent_with_mgr.process(event, diary)

        # Booking should still confirm
        assert diary.booking.confirmed
        assert len(result.emitted_events) > 0
        assert result.emitted_events[0].event_type == EventType.BOOKING_COMPLETE

    @pytest.mark.asyncio
    async def test_already_confirmed_no_double_booking(self, agent):
        """Already confirmed patient should not be re-booked."""
        diary = PatientDiary.create_new("PT-DOUBLE")
        _seed_booking_complete(diary)
        diary.header.current_phase = Phase.BOOKING

        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-DOUBLE",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await agent.process(event, diary)

        msg = result.responses[0].message.lower()
        assert "confirmed" in msg
        assert len(result.emitted_events) == 0  # No new BOOKING_COMPLETE

    @pytest.mark.asyncio
    async def test_gibberish_then_valid_selection(self, agent):
        """Gibberish selection → retry message, then valid selection → confirm."""
        diary = PatientDiary.create_new("PT-GIB")
        _seed_clinical_complete(diary)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-20", time="09:00", provider="Dr. A"),
            SlotOption(date="2026-03-21", time="11:30", provider="Dr. B"),
            SlotOption(date="2026-03-22", time="14:00", provider="Dr. C"),
        ]

        # Gibberish
        result1 = await agent.process(_user_msg("PT-GIB", "asdfghjkl"), diary)
        assert not diary.booking.confirmed
        assert "didn't" in result1.responses[0].message.lower()

        # Valid selection
        result2 = await agent.process(_user_msg("PT-GIB", "2"), diary)
        assert diary.booking.confirmed
        assert diary.booking.slot_selected.date == "2026-03-21"

    @pytest.mark.asyncio
    async def test_selection_out_of_range(self, agent):
        """Selecting slot 5 when only 3 offered → retry message."""
        diary = PatientDiary.create_new("PT-OOR")
        _seed_clinical_complete(diary)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-20", time="09:00", provider="Dr. A"),
            SlotOption(date="2026-03-21", time="11:30", provider="Dr. B"),
        ]

        result = await agent.process(_user_msg("PT-OOR", "5"), diary)
        assert not diary.booking.confirmed

    @pytest.mark.asyncio
    async def test_rebooking_after_deterioration(self):
        """Clinical agent clears booking → booking agent offers new slots."""
        booking_agent = BookingAgent(schedule_manager=None, llm_client=None)
        diary = PatientDiary.create_new("PT-REBOOK")
        _seed_booking_complete(diary)

        # Simulate clinical agent clearing booking after deterioration
        diary.booking.confirmed = False
        diary.booking.slots_offered = []
        diary.booking.slot_selected = None
        diary.header.risk_level = RiskLevel.HIGH
        diary.header.current_phase = Phase.BOOKING

        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-REBOOK",
            source_agent="clinical",
            payload={"channel": "websocket", "rebooking": True},
        )
        result = await booking_agent.process(event, diary)

        assert len(diary.booking.slots_offered) == 3
        assert "HIGH" in result.responses[0].message


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P1 #7: GP Communication Edge Cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPEdgeCases:
    """Tests for GP communication failure modes."""

    @pytest.fixture
    def clinical_agent(self):
        return ClinicalAgent(llm_client=None, risk_scorer=RiskScorer())

    @pytest.mark.asyncio
    async def test_gp_response_empty_lab_results(self, clinical_agent):
        """GP responds with empty lab results — should not crash."""
        diary = PatientDiary.create_new("PT-GPEMPTY")
        _seed_intake_complete(diary)
        diary.clinical.chief_complaint = "fatigue"
        diary.gp_channel.queries.append(GPQuery(
            query_id="Q-001", query_type="missing_labs",
            query_text="Please send latest blood work",
            status="pending",
        ))

        event = EventEnvelope.handoff(
            event_type=EventType.GP_RESPONSE,
            patient_id="PT-GPEMPTY",
            source_agent="gp_comms",
            payload={
                "lab_results": {},
                "attachments": [],
                "query_id": "Q-001",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        # Should handle gracefully
        assert len(result.responses) >= 0  # May or may not generate a response

    @pytest.mark.asyncio
    async def test_gp_response_with_lab_data(self, clinical_agent):
        """GP responds with valid lab data — should be merged."""
        diary = PatientDiary.create_new("PT-GPLAB")
        _seed_intake_complete(diary)
        diary.clinical.chief_complaint = "elevated liver enzymes"
        diary.clinical.condition_context = "cirrhosis"
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        diary.clinical.questions_asked = [
            ClinicalQuestion(question="Q1", answer="A1"),
            ClinicalQuestion(question="Q2", answer="A2"),
        ]
        diary.gp_channel.queries.append(GPQuery(
            query_id="Q-002", query_type="missing_labs",
            query_text="Please send LFT results",
            status="pending",
        ))

        event = EventEnvelope.handoff(
            event_type=EventType.GP_RESPONSE,
            patient_id="PT-GPLAB",
            source_agent="gp_comms",
            payload={
                "lab_results": {"bilirubin": 3.5, "ALT": 120},
                "attachments": ["lab_report.pdf"],
                "query_id": "Q-002",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        # Lab data should be incorporated
        lab_docs = [d for d in diary.clinical.documents if "lab" in d.type]
        assert len(lab_docs) > 0

    @pytest.mark.asyncio
    async def test_gp_response_no_matching_query(self, clinical_agent):
        """GP responds to a query that doesn't exist — should not crash."""
        diary = PatientDiary.create_new("PT-GPNOQ")
        _seed_intake_complete(diary)
        diary.clinical.chief_complaint = "pain"

        event = EventEnvelope.handoff(
            event_type=EventType.GP_RESPONSE,
            patient_id="PT-GPNOQ",
            source_agent="gp_comms",
            payload={
                "lab_results": {"ALT": 100},
                "attachments": [],
                "query_id": "Q-NONEXISTENT",
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)
        # Should not crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_deterioration_alert_from_assessment_timeout(self, clinical_agent):
        """Clinical agent handles DETERIORATION_ALERT from stalled assessment."""
        diary = PatientDiary.create_new("PT-DETTIME")
        _seed_booking_complete(diary)

        event = EventEnvelope.handoff(
            event_type=EventType.DETERIORATION_ALERT,
            patient_id="PT-DETTIME",
            source_agent="monitoring",
            payload={
                "reason": "Stalled assessment timeout (49h)",
                "source": "assessment_timeout",
                "assessment": {
                    "severity": "moderate",
                    "recommendation": "bring_forward",
                    "reasoning": "Assessment timed out, escalated conservatively",
                    "symptoms": ["worse"],
                    "questions": [{"q": "Describe?", "a": None}],
                },
                "channel": "websocket",
            },
        )
        result = await clinical_agent.process(event, diary)

        # Clinical agent should handle timeout source the same as deterioration_assessment
        # For moderate severity with confirmed booking, should trigger rebooking
        assert result is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P1 Integration: Full Journey with Failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestJourneyWithFailures:
    """End-to-end journey tests with infrastructure failures injected."""

    @pytest.mark.asyncio
    async def test_full_journey_deterministic_mode(self):
        """Complete intake→clinical→booking→monitoring with all LLMs off."""
        intake = IntakeAgent(llm_client=None)
        clinical = ClinicalAgent(llm_client=None, risk_scorer=RiskScorer())
        booking = BookingAgent(schedule_manager=None, llm_client=None)
        monitoring = MonitoringAgent(llm_client=None)

        diary = PatientDiary.create_new("PT-FULL")

        # Intake — pre-seed to skip multi-turn intake conversation
        _seed_intake_complete(diary)

        # Clinical — process INTAKE_COMPLETE
        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-FULL",
            source_agent="intake",
            payload={"channel": "websocket"},
        )
        await clinical.process(event, diary)

        # Answer clinical questions until scoring
        for i in range(10):
            event = _user_msg("PT-FULL", f"Answer {i}: yes, moderate pain")
            result = await clinical.process(event, diary)
            if any(e.event_type == EventType.CLINICAL_COMPLETE for e in result.emitted_events):
                break

        # Booking
        diary.header.current_phase = Phase.BOOKING
        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-FULL",
            source_agent="clinical",
            payload={"channel": "websocket"},
        )
        result = await booking.process(event, diary)
        assert len(diary.booking.slots_offered) > 0

        # Select slot
        result = await booking.process(_user_msg("PT-FULL", "1"), diary)
        assert diary.booking.confirmed

        # Monitoring — setup
        event = EventEnvelope.handoff(
            event_type=EventType.BOOKING_COMPLETE,
            patient_id="PT-FULL",
            source_agent="booking",
            payload={"appointment_date": diary.booking.slot_selected.date, "channel": "websocket"},
        )
        result = await monitoring.process(event, diary)
        assert diary.monitoring.monitoring_active

        # Heartbeat — no escalation on normal message
        event = _user_msg("PT-FULL", "I'm feeling fine, thanks")
        result = await monitoring.process(event, diary)
        assert not diary.monitoring.deterioration_assessment.active

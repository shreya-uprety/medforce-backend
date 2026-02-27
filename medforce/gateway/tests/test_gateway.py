"""
Tests for the Gateway Router.

Covers:
  - Strategy A: explicit routing for handoff events
  - Strategy B: phase-based routing for external events
  - Permission checking integration
  - Circuit breaker (max chain depth)
  - Diary auto-creation for new patients
  - Event logging
  - Response dispatch via DispatcherRegistry
  - Patient scenarios with full routing flow
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import (
    AgentResponse,
    DispatcherRegistry,
    DeliveryResult,
)
from medforce.gateway.diary import (
    DiaryNotFoundError,
    DiaryStore,
    PatientDiary,
    Phase,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.gateway import Gateway, MAX_CHAIN_DEPTH
from medforce.gateway.permissions import PermissionChecker


# ── Test Helpers ──


class EchoAgent(BaseAgent):
    """Simple agent that echoes the event back as a response."""

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


class PhaseAdvanceAgent(BaseAgent):
    """Agent that advances diary phase and emits a handoff event."""

    agent_name = "phase_advance"

    def __init__(self, next_phase: Phase, handoff_event: EventType):
        self._next_phase = next_phase
        self._handoff_event = handoff_event

    async def process(self, event, diary):
        diary.header.current_phase = self._next_phase
        handoff = EventEnvelope.handoff(
            event_type=self._handoff_event,
            patient_id=event.patient_id,
            source_agent=self.agent_name,
        )
        return AgentResult(
            updated_diary=diary,
            emitted_events=[handoff],
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel="websocket",
                    message=f"Moving to {self._next_phase.value}",
                    metadata={"patient_id": event.patient_id},
                )
            ],
        )


class InfiniteLoopAgent(BaseAgent):
    """Agent that always emits another event (for circuit breaker testing)."""

    agent_name = "infinite"

    async def process(self, event, diary):
        loop_event = EventEnvelope.handoff(
            event_type=EventType.NEEDS_INTAKE_DATA,
            patient_id=event.patient_id,
            source_agent="infinite",
            payload={"missing_fields": ["name"]},
        )
        return AgentResult(
            updated_diary=diary,
            emitted_events=[loop_event],
        )


class _MockGCS:
    """Minimal GCS stub so _persist_chat_history doesn't crash."""

    def create_file_from_string(self, content, path, content_type="text/plain"):
        pass

    def read_file_as_bytes(self, path):
        return None


class MockDiaryStore:
    """In-memory diary store for testing."""

    def __init__(self):
        self._diaries: dict[str, tuple[PatientDiary, int]] = {}
        self._gcs = _MockGCS()

    def load(self, patient_id):
        if patient_id not in self._diaries:
            raise DiaryNotFoundError(f"Not found: {patient_id}")
        diary, gen = self._diaries[patient_id]
        return diary, gen

    def save(self, patient_id, diary, generation=None):
        new_gen = (generation or 0) + 1
        self._diaries[patient_id] = (diary, new_gen)
        return new_gen

    def create(self, patient_id, correlation_id=None):
        diary = PatientDiary.create_new(patient_id, correlation_id=correlation_id)
        gen = self.save(patient_id, diary, generation=None)
        return diary, gen

    def seed(self, patient_id, diary, generation=1):
        """Seed a diary directly for testing."""
        self._diaries[patient_id] = (diary, generation)


@pytest.fixture
def diary_store():
    return MockDiaryStore()


@pytest.fixture
def dispatcher_registry():
    registry = DispatcherRegistry()
    # Use a mock dispatcher that always succeeds
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
    # Register echo agent for all common routes
    echo = EchoAgent()
    gw.register_agent("intake", echo)
    gw.register_agent("clinical", echo)
    gw.register_agent("booking", echo)
    gw.register_agent("monitoring", echo)
    return gw


# ── Strategy A: Explicit Routing ──


class TestExplicitRouting:
    @pytest.mark.asyncio
    async def test_intake_complete_routes_to_clinical(self, gateway, diary_store):
        """INTAKE_COMPLETE should route to the clinical agent."""
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.INTAKE
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.handoff(
            EventType.INTAKE_COMPLETE, "PT-001", source_agent="intake"
        )
        result = await gateway.process_event(event)

        assert result is not None
        # Check event log shows routing to clinical
        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "clinical" for e in routed)

    @pytest.mark.asyncio
    async def test_clinical_complete_routes_to_booking(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.handoff(
            EventType.CLINICAL_COMPLETE, "PT-001", source_agent="clinical"
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "booking" for e in routed)

    @pytest.mark.asyncio
    async def test_booking_complete_routes_to_monitoring(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.BOOKING
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.handoff(
            EventType.BOOKING_COMPLETE, "PT-001", source_agent="booking"
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "monitoring" for e in routed)

    @pytest.mark.asyncio
    async def test_needs_intake_data_routes_to_intake(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.handoff(
            EventType.NEEDS_INTAKE_DATA, "PT-001", source_agent="clinical",
            payload={"missing_fields": ["phone"]},
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "intake" for e in routed)

    @pytest.mark.asyncio
    async def test_heartbeat_routes_to_monitoring(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.MONITORING
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.heartbeat("PT-001", days_since_appointment=14)
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "monitoring" for e in routed)

    @pytest.mark.asyncio
    async def test_unregistered_explicit_target(self, gateway, diary_store):
        """GP_QUERY routes to gp_comms, which is not registered — should log."""
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.handoff(
            EventType.GP_QUERY, "PT-001", source_agent="clinical"
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert any(e["status"] == "AGENT_NOT_FOUND" for e in log)


# ── Strategy B: Phase-Based Routing ──


class TestPhaseRouting:
    @pytest.mark.asyncio
    async def test_user_message_in_intake_routes_to_intake(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.INTAKE
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "My name is John Smith")
        result = await gateway.process_event(event)

        assert result is not None
        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "intake" for e in routed)

    @pytest.mark.asyncio
    async def test_user_message_in_clinical_routes_to_clinical(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "I have headaches")
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "clinical" for e in routed)

    @pytest.mark.asyncio
    async def test_user_message_in_closed_no_target(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLOSED
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello?")
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert any(e["status"] == "NO_TARGET" for e in log)

    @pytest.mark.asyncio
    async def test_document_upload_in_clinical(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary_store.seed("PT-001", diary)

        event = EventEnvelope(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-001",
            payload={"file_ref": "lab_results.pdf", "channel": "websocket"},
            sender_role=SenderRole.PATIENT,
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "clinical" for e in routed)


# ── Permission Checking ──


class TestGatewayPermissions:
    @pytest.mark.asyncio
    async def test_patient_always_allowed(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        result = await gateway.process_event(event)
        assert result is not None
        # Should not have any PERMISSION_DENIED in log
        log = gateway.get_event_log("PT-001")
        assert not any(e["status"] == "PERMISSION_DENIED" for e in log)

    @pytest.mark.asyncio
    async def test_unverified_helper_denied(self, gateway, diary_store):
        """An unverified helper should be denied."""
        from medforce.gateway.diary import HelperEntry

        diary = PatientDiary.create_new("PT-001")
        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-001", name="Sarah", verified=False,
            permissions=["full_access"],
        ))
        diary_store.seed("PT-001", diary)

        event = EventEnvelope(
            event_type=EventType.USER_MESSAGE,
            patient_id="PT-001",
            payload={"text": "Hi", "channel": "websocket"},
            sender_id="HELPER-001",
            sender_role=SenderRole.HELPER,
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert any(e["status"] == "PERMISSION_DENIED" for e in log)
        # Should return a rejection response
        assert len(result.responses) == 1
        assert "permission" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_verified_helper_allowed(self, gateway, diary_store):
        """A verified helper with send_messages permission should be allowed."""
        from medforce.gateway.diary import HelperEntry

        diary = PatientDiary.create_new("PT-001")
        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-001", name="Sarah", verified=True,
            permissions=["send_messages"],
        ))
        diary_store.seed("PT-001", diary)

        event = EventEnvelope(
            event_type=EventType.USER_MESSAGE,
            patient_id="PT-001",
            payload={"text": "How is John?", "channel": "websocket"},
            sender_id="HELPER-001",
            sender_role=SenderRole.HELPER,
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert not any(e["status"] == "PERMISSION_DENIED" for e in log)

    @pytest.mark.asyncio
    async def test_unknown_helper_denied(self, gateway, diary_store):
        """A helper not in the registry should be denied."""
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope(
            event_type=EventType.USER_MESSAGE,
            patient_id="PT-001",
            payload={"text": "Hi", "channel": "websocket"},
            sender_id="HELPER-999",
            sender_role=SenderRole.HELPER,
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert any(e["status"] == "PERMISSION_DENIED" for e in log)


# ── Circuit Breaker ──


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_stops_infinite_loop(self, diary_store, dispatcher_registry):
        """Gateway should stop processing after MAX_CHAIN_DEPTH events."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", InfiniteLoopAgent())

        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.handoff(
            EventType.NEEDS_INTAKE_DATA, "PT-001", source_agent="clinical",
            payload={"missing_fields": ["name"]},
        )
        await gw.process_event(event)

        log = gw.get_event_log("PT-001")
        circuit_breaker = [e for e in log if e["status"] == "CIRCUIT_BREAKER"]
        assert len(circuit_breaker) > 0
        # Total routed events should be capped at MAX_CHAIN_DEPTH
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert len(routed) <= MAX_CHAIN_DEPTH


# ── Diary Auto-Creation ──


class TestDiaryCreation:
    @pytest.mark.asyncio
    async def test_new_patient_diary_created(self, gateway, diary_store):
        """First contact from a new patient should auto-create a diary."""
        import asyncio

        event = EventEnvelope.user_message("PT-NEW-001", "Hello")
        result = await gateway.process_event(event)

        # Diary save is a background task — drain before asserting
        if gateway._bg_tasks:
            await asyncio.gather(*gateway._bg_tasks, return_exceptions=True)

        assert result is not None
        # Diary should now exist in the store
        diary, gen = diary_store.load("PT-NEW-001")
        assert diary.header.patient_id == "PT-NEW-001"
        assert diary.header.current_phase == Phase.INTAKE

    @pytest.mark.asyncio
    async def test_existing_patient_diary_loaded(self, gateway, diary_store):
        """Existing patient diary should be loaded, not overwritten."""
        diary = PatientDiary.create_new("PT-001")
        diary.intake.mark_field_collected("name", "John Smith")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "My DOB is 15/03/1985")
        result = await gateway.process_event(event)

        # Diary should still have the name
        updated_diary, _ = diary_store.load("PT-001")
        assert updated_diary.intake.name == "John Smith"


# ── Event Logging ──


class TestEventLogging:
    @pytest.mark.asyncio
    async def test_event_log_populated(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        assert len(log) > 0
        assert log[0]["patient_id"] == "PT-001"

    @pytest.mark.asyncio
    async def test_event_log_filtered_by_patient(self, gateway, diary_store):
        d1 = PatientDiary.create_new("PT-001")
        d2 = PatientDiary.create_new("PT-002")
        diary_store.seed("PT-001", d1)
        diary_store.seed("PT-002", d2)

        await gateway.process_event(EventEnvelope.user_message("PT-001", "Hi"))
        await gateway.process_event(EventEnvelope.user_message("PT-002", "Hello"))

        log1 = gateway.get_event_log("PT-001")
        log2 = gateway.get_event_log("PT-002")

        assert all(e["patient_id"] == "PT-001" for e in log1)
        assert all(e["patient_id"] == "PT-002" for e in log2)

    @pytest.mark.asyncio
    async def test_event_log_limit(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        for i in range(10):
            await gateway.process_event(
                EventEnvelope.user_message("PT-001", f"Message {i}")
            )

        log = gateway.get_event_log("PT-001", limit=3)
        assert len(log) == 3


# ── Conversation Logging ──


class TestConversationLogging:
    @pytest.mark.asyncio
    async def test_user_message_logged_in_diary(self, gateway, diary_store):
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello I'm John")
        await gateway.process_event(event)

        updated, _ = diary_store.load("PT-001")
        # Should have at least 2 entries: inbound + outbound
        assert len(updated.conversation_log) >= 2
        inbound = [c for c in updated.conversation_log if "PATIENT" in c.direction and "AGENT" in c.direction and c.direction.index("PATIENT") < c.direction.index("AGENT")]
        assert len(inbound) >= 1


# ── Response Dispatch ──


class TestResponseDispatch:
    @pytest.mark.asyncio
    async def test_responses_dispatched(self, gateway, diary_store, dispatcher_registry):
        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        await gateway.process_event(event)

        # The mock dispatcher's send should have been called
        ws_dispatcher = dispatcher_registry.get("websocket")
        assert ws_dispatcher.send.called

    @pytest.mark.asyncio
    async def test_failed_dispatch_logged(self, diary_store):
        """If dispatch fails, it should be logged but not crash."""
        registry = DispatcherRegistry()
        mock_disp = MagicMock()
        mock_disp.channel_name = "websocket"
        mock_disp.send = AsyncMock(return_value=DeliveryResult(
            success=False, channel="websocket", recipient="patient",
            error="Connection closed",
        ))
        registry.register(mock_disp)

        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=registry,
        )
        gw.register_agent("intake", EchoAgent())

        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Hello")
        # Should not raise even though dispatch failed
        result = await gw.process_event(event)
        assert result is not None


# ── Agent Registration ──


class TestAgentRegistration:
    def test_register_and_retrieve(self, gateway):
        assert "intake" in gateway.registered_agents
        assert "clinical" in gateway.registered_agents
        assert gateway.get_agent("intake") is not None
        assert gateway.get_agent("nonexistent") is None

    def test_register_overwrites(self, gateway):
        old = gateway.get_agent("intake")
        new_agent = EchoAgent()
        gateway.register_agent("intake", new_agent)
        assert gateway.get_agent("intake") is new_agent
        assert gateway.get_agent("intake") is not old


# ── Patient Scenarios ──


class TestGatewayScenarios:
    @pytest.mark.asyncio
    async def test_scenario_new_patient_first_message(self, gateway, diary_store):
        """Brand new patient sends first message → diary created, routed to intake."""
        import asyncio

        event = EventEnvelope.user_message("PT-BRAND-NEW", "Hi, I've been referred")
        result = await gateway.process_event(event)

        # Diary save is a background task — drain before asserting
        if gateway._bg_tasks:
            await asyncio.gather(*gateway._bg_tasks, return_exceptions=True)

        assert result is not None
        diary, _ = diary_store.load("PT-BRAND-NEW")
        assert diary.header.current_phase == Phase.INTAKE
        assert len(result.responses) >= 1

    @pytest.mark.asyncio
    async def test_scenario_phase_transition_intake_to_clinical(self, diary_store, dispatcher_registry):
        """Intake emits INTAKE_COMPLETE → clinical agent receives the event."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )

        # Intake agent that completes and hands off
        intake = PhaseAdvanceAgent(Phase.CLINICAL, EventType.INTAKE_COMPLETE)
        clinical = EchoAgent()
        gw.register_agent("intake", intake)
        gw.register_agent("clinical", clinical)

        diary = PatientDiary.create_new("PT-001")
        diary_store.seed("PT-001", diary)

        event = EventEnvelope.user_message("PT-001", "Done with intake")
        await gw.process_event(event)

        log = gw.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        agents_called = [e["detail"] for e in routed]
        # Intake was called first, then clinical via handoff
        assert "intake" in agents_called
        assert "clinical" in agents_called

    @pytest.mark.asyncio
    async def test_scenario_helper_with_limited_access_denied(self, gateway, diary_store):
        """Helper with view-only cannot send messages."""
        from medforce.gateway.diary import HelperEntry

        diary = PatientDiary.create_new("PT-001")
        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-VIEW", name="Friend", verified=True,
            permissions=["view_status"],
        ))
        diary_store.seed("PT-001", diary)

        event = EventEnvelope(
            event_type=EventType.USER_MESSAGE,
            patient_id="PT-001",
            payload={"text": "How is the patient?", "channel": "websocket"},
            sender_id="HELPER-VIEW",
            sender_role=SenderRole.HELPER,
        )
        result = await gateway.process_event(event)

        assert len(result.responses) == 1
        assert "permission" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_scenario_gp_response_routes_to_clinical(self, gateway, diary_store):
        """GP sends a response → should route to clinical agent."""
        diary = PatientDiary.create_new("PT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary_store.seed("PT-001", diary)

        event = EventEnvelope(
            event_type=EventType.GP_RESPONSE,
            patient_id="PT-001",
            payload={"lab_results": {"ALT": 45}},
            sender_id="GP-DrPatel",
            sender_role=SenderRole.GP,
        )
        result = await gateway.process_event(event)

        log = gateway.get_event_log("PT-001")
        routed = [e for e in log if e["status"] == "ROUTED"]
        assert any(e["detail"] == "clinical" for e in routed)

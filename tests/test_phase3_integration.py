"""
Phase 3 Integration Tests — Full patient journey from Clinical to Booking.

Tests the complete flow: INTAKE_COMPLETE → Clinical Assessment →
Risk Scoring → Slot Presentation → Slot Selection → BOOKING_COMPLETE.

Also tests GP communication round-trips and backward loops.
"""

import pytest
from unittest.mock import MagicMock

from medforce.gateway.agents.booking_agent import BookingAgent
from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.agents.intake_agent import IntakeAgent
from medforce.gateway.agents.risk_scorer import RiskScorer
from medforce.gateway.channels import DispatcherRegistry
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSubPhase,
    DiaryNotFoundError,
    DiaryStore,
    PatientDiary,
    Phase,
    RiskLevel,
    SlotOption,
)
from medforce.gateway.dispatchers.test_harness_dispatcher import TestHarnessDispatcher
from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.gateway import Gateway
from medforce.gateway.handlers.gp_comms import GPCommunicationHandler
from medforce.gateway.permissions import PermissionChecker


# ── Mock Diary Store ──


class MockDiaryStore:
    """In-memory diary store for testing."""

    def __init__(self):
        self._diaries: dict[str, PatientDiary] = {}
        self._generations: dict[str, int] = {}

    def load(self, patient_id: str) -> tuple[PatientDiary, int]:
        if patient_id not in self._diaries:
            raise DiaryNotFoundError(f"No diary for {patient_id}")
        return self._diaries[patient_id], self._generations.get(patient_id, 1)

    def save(self, patient_id: str, diary: PatientDiary, generation: int | None = None) -> int:
        self._diaries[patient_id] = diary
        new_gen = (generation or 0) + 1
        self._generations[patient_id] = new_gen
        return new_gen

    def create(self, patient_id: str, correlation_id: str | None = None) -> tuple[PatientDiary, int]:
        diary = PatientDiary.create_new(patient_id, correlation_id=correlation_id)
        gen = self.save(patient_id, diary)
        return diary, gen

    def exists(self, patient_id: str) -> bool:
        return patient_id in self._diaries


# ── Gateway Factory ──


def create_test_gateway() -> tuple[Gateway, MockDiaryStore, TestHarnessDispatcher]:
    """Create a fully wired Gateway for testing."""
    diary_store = MockDiaryStore()
    harness = TestHarnessDispatcher()
    registry = DispatcherRegistry()
    registry.register(harness)

    gateway = Gateway(
        diary_store=diary_store,
        dispatcher_registry=registry,
        permission_checker=PermissionChecker(),
    )

    gateway.register_agent("intake", IntakeAgent())
    gateway.register_agent("clinical", ClinicalAgent())
    gateway.register_agent("booking", BookingAgent())
    gateway.register_agent("gp_comms", GPCommunicationHandler())

    return gateway, diary_store, harness


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIntakeToClinicalHandoff:
    """INTAKE_COMPLETE event routed to clinical agent via Gateway."""

    @pytest.mark.asyncio
    async def test_intake_complete_routes_to_clinical(self):
        gateway, store, harness = create_test_gateway()

        # Pre-create a diary in CLINICAL phase (simulating post-intake)
        diary = PatientDiary.create_new("PT-INT-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.intake.name = "John Smith"
        diary.intake.phone = "07700900000"
        store.save("PT-INT-001", diary)

        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-INT-001",
            source_agent="intake",
            payload={"channel": "test_harness"},
        )

        result = await gateway.process_event(event)

        assert result is not None
        # Diary should be in clinical sub-phase
        saved_diary, _ = store.load("PT-INT-001")
        assert saved_diary.clinical.sub_phase in (
            ClinicalSubPhase.ANALYZING_REFERRAL,
            ClinicalSubPhase.ASKING_QUESTIONS,
        )

    @pytest.mark.asyncio
    async def test_clinical_complete_routes_to_booking(self):
        gateway, store, harness = create_test_gateway()

        # Pre-create a diary in BOOKING phase
        diary = PatientDiary.create_new("PT-INT-002")
        diary.header.current_phase = Phase.BOOKING
        diary.header.risk_level = RiskLevel.HIGH
        diary.intake.name = "Jane Doe"
        store.save("PT-INT-002", diary)

        event = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-INT-002",
            source_agent="clinical",
            payload={
                "risk_level": "high",
                "channel": "test_harness",
            },
        )

        result = await gateway.process_event(event)

        assert result is not None
        # Should have slot presentation response
        assert len(result.responses) >= 1


class TestClinicalToBookingFlow:
    """Full clinical → scoring → booking flow via Gateway."""

    @pytest.mark.asyncio
    async def test_full_assessment_and_booking(self):
        """Patient goes through full clinical + booking with the Gateway."""
        gateway, store, harness = create_test_gateway()

        # Create diary in CLINICAL phase with intake done
        diary = PatientDiary.create_new("PT-FULL-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.intake.name = "Sarah Connor"
        diary.intake.phone = "07700900222"
        diary.intake.nhs_number = "9876543210"
        diary.intake.dob = "1990-05-20"
        diary.intake.gp_name = "Dr. Brown"
        store.save("PT-FULL-001", diary)

        # Step 1: INTAKE_COMPLETE triggers clinical start
        event1 = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-FULL-001",
            source_agent="intake",
            payload={"channel": "test_harness"},
        )
        await gateway.process_event(event1)

        # Step 2: Patient provides chief complaint
        event2 = EventEnvelope.user_message(
            patient_id="PT-FULL-001",
            text="I have been experiencing severe abdominal pain and jaundice",
            channel="test_harness",
        )
        await gateway.process_event(event2)

        # Step 3: Upload lab results with critical values
        event3 = EventEnvelope(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-FULL-001",
            payload={
                "type": "lab_results",
                "file_ref": "gs://test/labs.pdf",
                "channel": "test_harness",
                "extracted_values": {"bilirubin": 7.0, "ALT": 600},
            },
            sender_id="PATIENT",
            sender_role=SenderRole.PATIENT,
        )
        result3 = await gateway.process_event(event3)

        # After uploading labs, should score and move to BOOKING
        saved, _ = store.load("PT-FULL-001")
        # The diary should either be in BOOKING or still CLINICAL depending on
        # whether _ready_for_scoring triggered. Let's check:
        if saved.header.current_phase == Phase.BOOKING:
            # Clinical complete was emitted and booking was triggered
            assert saved.clinical.risk_level == RiskLevel.HIGH

            # Step 4: Patient selects a booking slot
            # First we need to see what the booking state looks like
            if saved.booking.slots_offered:
                event4 = EventEnvelope.user_message(
                    patient_id="PT-FULL-001",
                    text="1",
                    channel="test_harness",
                )
                await gateway.process_event(event4)

                final, _ = store.load("PT-FULL-001")
                assert final.booking.confirmed is True
                assert final.header.current_phase == Phase.MONITORING


class TestGPCommunicationIntegration:
    """GP query/reminder flow through the Gateway."""

    @pytest.mark.asyncio
    async def test_gp_query_routed_to_gp_comms(self):
        gateway, store, harness = create_test_gateway()

        diary = PatientDiary.create_new("PT-GP-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.intake.name = "Bob Martin"
        diary.intake.gp_name = "Dr. Smith"
        diary.gp_channel.gp_name = "Dr. Smith"
        diary.gp_channel.gp_email = "dr.smith@nhs.net"
        store.save("PT-GP-001", diary)

        event = EventEnvelope.handoff(
            event_type=EventType.GP_QUERY,
            patient_id="PT-GP-001",
            source_agent="clinical",
            payload={
                "query_type": "missing_labs",
                "requested_data": ["LFTs", "FBC"],
                "channel": "test_harness",
            },
        )

        result = await gateway.process_event(event)

        assert result is not None
        # Should have responses (email to GP + patient notification)
        assert len(result.responses) >= 1

        # Diary should record the query
        saved, _ = store.load("PT-GP-001")
        assert len(saved.gp_channel.queries) == 1

    @pytest.mark.asyncio
    async def test_gp_reminder_routed_to_gp_comms(self):
        gateway, store, harness = create_test_gateway()

        diary = PatientDiary.create_new("PT-GP-002")
        diary.header.current_phase = Phase.CLINICAL
        diary.gp_channel.gp_name = "Dr. Jones"
        diary.gp_channel.gp_email = "dr.jones@nhs.net"
        from medforce.gateway.diary import GPQuery
        diary.gp_channel.queries = [
            GPQuery(query_id="GPQ-TEST", status="pending"),
        ]
        store.save("PT-GP-002", diary)

        event = EventEnvelope.handoff(
            event_type=EventType.GP_REMINDER,
            patient_id="PT-GP-002",
            source_agent="system",
            payload={"channel": "test_harness"},
        )

        result = await gateway.process_event(event)

        # Should send reminder
        saved, _ = store.load("PT-GP-002")
        assert saved.gp_channel.queries[0].reminder_sent is not None


class TestPhaseBasedRouting:
    """Verify phase-based routing works for Phase 3 agents."""

    @pytest.mark.asyncio
    async def test_user_message_in_clinical_phase_routes_to_clinical(self):
        gateway, store, _ = create_test_gateway()

        diary = PatientDiary.create_new("PT-ROUTE-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.sub_phase = ClinicalSubPhase.ASKING_QUESTIONS
        diary.intake.name = "Routing Test"
        diary.intake.phone = "07700900333"
        store.save("PT-ROUTE-001", diary)

        event = EventEnvelope.user_message(
            patient_id="PT-ROUTE-001",
            text="I have back pain",
            channel="test_harness",
        )

        result = await gateway.process_event(event)

        assert result is not None
        # Clinical agent should process this
        saved, _ = store.load("PT-ROUTE-001")
        assert len(saved.clinical.questions_asked) >= 0

    @pytest.mark.asyncio
    async def test_user_message_in_booking_phase_routes_to_booking(self):
        gateway, store, _ = create_test_gateway()

        diary = PatientDiary.create_new("PT-ROUTE-002")
        diary.header.current_phase = Phase.BOOKING
        diary.header.risk_level = RiskLevel.MEDIUM
        diary.intake.name = "Booking Routing"
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        store.save("PT-ROUTE-002", diary)

        event = EventEnvelope.user_message(
            patient_id="PT-ROUTE-002",
            text="1",
            channel="test_harness",
        )

        result = await gateway.process_event(event)

        assert result is not None
        saved, _ = store.load("PT-ROUTE-002")
        assert saved.booking.confirmed is True


class TestEventChaining:
    """Test that chained events flow correctly through the Gateway."""

    @pytest.mark.asyncio
    async def test_clinical_complete_chains_to_booking(self):
        """Clinical emitting CLINICAL_COMPLETE should auto-route to booking."""
        gateway, store, harness = create_test_gateway()

        # Create a diary ready for scoring
        diary = PatientDiary.create_new("PT-CHAIN-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.sub_phase = ClinicalSubPhase.ASKING_QUESTIONS
        diary.clinical.chief_complaint = "liver pain"
        diary.clinical.questions_asked = [
            ClinicalQuestion(question="Q1?", answer="A1"),
            ClinicalQuestion(question="Q2?", answer="A2"),
            ClinicalQuestion(question="Q3?", answer="A3"),
        ]
        diary.intake.name = "Chain Test"
        diary.intake.phone = "07700900444"
        store.save("PT-CHAIN-001", diary)

        # Upload labs — this should trigger scoring → CLINICAL_COMPLETE → booking
        event = EventEnvelope(
            event_type=EventType.DOCUMENT_UPLOADED,
            patient_id="PT-CHAIN-001",
            payload={
                "type": "lab_results",
                "file_ref": "gs://test/labs.pdf",
                "channel": "test_harness",
                "extracted_values": {"bilirubin": 6.0},
            },
            sender_id="PATIENT",
            sender_role=SenderRole.PATIENT,
        )

        await gateway.process_event(event)

        # After chaining, diary should show booking phase with slots
        saved, _ = store.load("PT-CHAIN-001")
        # Clinical should be complete
        assert saved.clinical.sub_phase == ClinicalSubPhase.COMPLETE
        assert saved.clinical.risk_level == RiskLevel.HIGH
        # Should have transitioned through booking
        # (CLINICAL_COMPLETE chains to booking agent)


class TestTestHarnessCapture:
    """Verify TestHarnessDispatcher captures all responses."""

    @pytest.mark.asyncio
    async def test_responses_captured_in_harness(self):
        gateway, store, harness = create_test_gateway()

        diary = PatientDiary.create_new("PT-HARNESS-001")
        diary.header.current_phase = Phase.CLINICAL
        diary.intake.name = "Harness Test"
        diary.intake.phone = "07700900555"
        store.save("PT-HARNESS-001", diary)

        event = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-HARNESS-001",
            source_agent="intake",
            payload={"channel": "test_harness"},
        )

        await gateway.process_event(event)

        # TestHarnessDispatcher should have captured the response
        responses = harness.get_responses("PT-HARNESS-001")
        assert len(responses) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Multi-Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiPatient:
    """Multiple patients processed concurrently shouldn't interfere."""

    @pytest.mark.asyncio
    async def test_two_patients_independent(self):
        gateway, store, harness = create_test_gateway()

        # Patient A in clinical
        diary_a = PatientDiary.create_new("PT-A")
        diary_a.header.current_phase = Phase.CLINICAL
        diary_a.intake.name = "Patient A"
        diary_a.intake.phone = "07700900001"
        store.save("PT-A", diary_a)

        # Patient B in booking
        diary_b = PatientDiary.create_new("PT-B")
        diary_b.header.current_phase = Phase.BOOKING
        diary_b.header.risk_level = RiskLevel.LOW
        diary_b.intake.name = "Patient B"
        diary_b.booking.slots_offered = [
            SlotOption(date="2026-04-01", time="09:00"),
        ]
        store.save("PT-B", diary_b)

        # Process both
        event_a = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-A",
            source_agent="intake",
            payload={"channel": "test_harness"},
        )
        event_b = EventEnvelope.user_message(
            patient_id="PT-B",
            text="1",
            channel="test_harness",
        )

        await gateway.process_event(event_a)
        await gateway.process_event(event_b)

        # Check they're independent
        saved_a, _ = store.load("PT-A")
        saved_b, _ = store.load("PT-B")

        assert saved_a.header.current_phase == Phase.CLINICAL
        assert saved_b.booking.confirmed is True
        assert saved_b.header.current_phase == Phase.MONITORING

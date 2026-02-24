"""
P0 Safety Tests — Critical gap coverage for:
  1. Stalled assessment timeout (patient goes silent during deterioration assessment)
  2. Phase transition recovery (patient stuck in a phase beyond SLA)
  3. Per-patient rate limiting (message flood protection)

All agents run with llm_client=None (deterministic fallback mode).
"""

import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from medforce.gateway.agents.monitoring_agent import (
    MonitoringAgent,
    ASSESSMENT_TIMEOUT_HOURS,
    PHASE_STALE_THRESHOLDS,
)
from medforce.gateway.agents.base_agent import AgentResult
from medforce.gateway.channels import (
    AgentResponse,
    DeliveryResult,
    DispatcherRegistry,
)
from medforce.gateway.diary import (
    ClinicalSubPhase,
    CommunicationPlan,
    DeteriorationAssessment,
    DeteriorationQuestion,
    MonitoringEntry,
    PatientDiary,
    Phase,
    RiskLevel,
    ScheduledQuestion,
    SlotOption,
    DiaryNotFoundError,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.gateway import Gateway, RATE_LIMIT_MAX_MESSAGES, RATE_LIMIT_WINDOW_SECONDS


# ── Shared Helpers ──


def _seed_monitoring_diary(patient_id: str = "PT-P0") -> PatientDiary:
    """Create a diary that's in the monitoring phase with a communication plan."""
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
    diary.booking.slot_selected = SlotOption(
        date="2026-03-15", time="09:00", provider="Dr. Available"
    )
    diary.monitoring.monitoring_active = True
    diary.monitoring.appointment_date = "2026-03-15"
    diary.monitoring.baseline = {"bilirubin": 2.0, "ALT": 45}
    diary.monitoring.communication_plan = CommunicationPlan(
        risk_level="medium",
        total_messages=4,
        check_in_days=[14, 30, 60, 90],
        questions=[
            ScheduledQuestion(question="How are you?", day=14, priority=1, category="general"),
        ],
        generated=True,
    )
    diary.header.current_phase = Phase.MONITORING
    diary.header.phase_entered_at = datetime.now(timezone.utc)
    return diary


def _heartbeat(patient_id: str, days: int = 14) -> EventEnvelope:
    return EventEnvelope.handoff(
        event_type=EventType.HEARTBEAT,
        patient_id=patient_id,
        source_agent="scheduler",
        payload={"days_since_appointment": days, "channel": "websocket"},
    )


def _user_msg(patient_id: str, text: str) -> EventEnvelope:
    return EventEnvelope.user_message(patient_id, text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P0 #1: Stalled Assessment Timeout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStalledAssessmentTimeout:
    """Tests for stalled deterioration assessment timeout on heartbeat."""

    @pytest.fixture
    def agent(self):
        return MonitoringAgent(llm_client=None)

    @pytest.mark.asyncio
    async def test_fresh_assessment_not_timed_out(self, agent):
        """An assessment started recently should NOT be force-completed."""
        diary = _seed_monitoring_diary()
        diary.monitoring.deterioration_assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["swelling"],
            trigger_message="My legs are swelling",
            started=datetime.now(timezone.utc),
            questions=[
                DeteriorationQuestion(
                    question="Can you describe the swelling?", category="description"
                ),
            ],
        )

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        # Should NOT have force-completed — normal heartbeat processing
        assessment = diary.monitoring.deterioration_assessment
        assert not assessment.assessment_complete
        assert assessment.severity is None

    @pytest.mark.asyncio
    async def test_stalled_assessment_force_completed(self, agent):
        """An assessment older than ASSESSMENT_TIMEOUT_HOURS should be force-completed."""
        diary = _seed_monitoring_diary()
        stale_time = datetime.now(timezone.utc) - timedelta(hours=ASSESSMENT_TIMEOUT_HOURS + 1)
        diary.monitoring.deterioration_assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["swelling"],
            trigger_message="My legs are swelling",
            started=stale_time,
            questions=[
                DeteriorationQuestion(
                    question="Can you describe the swelling?",
                    answer="They're very puffy and painful",
                    category="description",
                ),
                DeteriorationQuestion(
                    question="Any new symptoms?",
                    category="new_symptoms",
                ),
            ],
        )

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        assessment = diary.monitoring.deterioration_assessment
        assert assessment.assessment_complete
        assert assessment.severity is not None
        # Should have escalated conservatively
        assert assessment.severity in ("moderate", "severe", "emergency")
        assert "timed out" in (assessment.reasoning or "").lower()
        # Should have a response to the patient
        assert len(result.responses) > 0
        assert any("haven't heard back" in r.message for r in result.responses)

    @pytest.mark.asyncio
    async def test_stalled_assessment_no_answers_defaults_moderate(self, agent):
        """Assessment with 0 answers after timeout should default to moderate."""
        diary = _seed_monitoring_diary()
        stale_time = datetime.now(timezone.utc) - timedelta(hours=ASSESSMENT_TIMEOUT_HOURS + 10)
        diary.monitoring.deterioration_assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["worse"],
            trigger_message="Getting worse",
            started=stale_time,
            questions=[
                DeteriorationQuestion(
                    question="Can you describe what's worse?",
                    category="description",
                ),
            ],
        )

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        assessment = diary.monitoring.deterioration_assessment
        assert assessment.assessment_complete
        assert assessment.severity == "moderate"
        # Should emit DETERIORATION_ALERT
        alert_types = [e.event_type for e in result.emitted_events]
        assert EventType.DETERIORATION_ALERT in alert_types

    @pytest.mark.asyncio
    async def test_stalled_assessment_severity_escalation(self, agent):
        """When assessment times out with answers, severity should be bumped up."""
        diary = _seed_monitoring_diary()
        stale_time = datetime.now(timezone.utc) - timedelta(hours=ASSESSMENT_TIMEOUT_HOURS + 5)
        diary.monitoring.deterioration_assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["worse"],
            trigger_message="I've been getting worse",
            started=stale_time,
            questions=[
                DeteriorationQuestion(
                    question="Can you describe?",
                    answer="more pain",
                    category="description",
                ),
                DeteriorationQuestion(
                    question="New symptoms?",
                    answer="no just the same",
                    category="new_symptoms",
                ),
            ],
        )

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        assessment = diary.monitoring.deterioration_assessment
        assert assessment.assessment_complete
        # Fallback scorer would rate "worse" + "more pain" as moderate,
        # then escalation bumps to severe
        assert assessment.severity in ("moderate", "severe")
        assert "escalated" in (assessment.reasoning or "").lower()

    @pytest.mark.asyncio
    async def test_completed_assessment_not_retrigered(self, agent):
        """A completed assessment should not be re-triggered on heartbeat."""
        diary = _seed_monitoring_diary()
        diary.monitoring.deterioration_assessment = DeteriorationAssessment(
            active=True,
            assessment_complete=True,
            severity="moderate",
            recommendation="bring_forward",
            detected_symptoms=["swelling"],
            trigger_message="My legs are swelling",
            started=datetime.now(timezone.utc) - timedelta(hours=100),
        )

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        # Should NOT re-trigger — assessment was already complete
        assessment = diary.monitoring.deterioration_assessment
        assert assessment.severity == "moderate"  # unchanged
        assert assessment.assessment_complete

    @pytest.mark.asyncio
    async def test_no_assessment_no_timeout(self, agent):
        """When there's no active assessment, heartbeat should proceed normally."""
        diary = _seed_monitoring_diary()

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        # Normal heartbeat — should have responses (check-in or milestone)
        assessment = diary.monitoring.deterioration_assessment
        assert not assessment.active
        assert not assessment.assessment_complete

    @pytest.mark.asyncio
    async def test_timeout_creates_monitoring_entry(self, agent):
        """Force-completing should log an assessment_timeout entry."""
        diary = _seed_monitoring_diary()
        stale_time = datetime.now(timezone.utc) - timedelta(hours=ASSESSMENT_TIMEOUT_HOURS + 1)
        diary.monitoring.deterioration_assessment = DeteriorationAssessment(
            active=True,
            detected_symptoms=["worse"],
            trigger_message="Feeling worse",
            started=stale_time,
            questions=[],
        )

        event = _heartbeat("PT-P0")
        await agent.process(event, diary)

        timeout_entries = [
            e for e in diary.monitoring.entries if e.type == "assessment_timeout"
        ]
        assert len(timeout_entries) == 1
        assert "force-completed" in timeout_entries[0].action.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P0 #2: Phase Transition Recovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPhaseTransitionRecovery:
    """Tests for detecting patients stuck in a phase beyond SLA."""

    @pytest.fixture
    def agent(self):
        return MonitoringAgent(llm_client=None)

    @pytest.mark.asyncio
    async def test_intake_stuck_sends_nudge(self, agent):
        """Patient stuck in intake beyond 72h should receive a nudge."""
        diary = _seed_monitoring_diary()
        diary.header.current_phase = Phase.INTAKE
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(hours=73)

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        msg = result.responses[0].message.lower()
        assert "details" in msg or "continue" in msg
        # Check it logged the staleness
        stale_entries = [e for e in diary.monitoring.entries if "phase_stale" in e.type]
        assert len(stale_entries) == 1

    @pytest.mark.asyncio
    async def test_clinical_stuck_sends_nudge(self, agent):
        """Patient stuck in clinical beyond 72h should receive a nudge."""
        diary = _seed_monitoring_diary()
        diary.header.current_phase = Phase.CLINICAL
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(hours=73)

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        msg = result.responses[0].message.lower()
        assert "assessment" in msg or "questions" in msg

    @pytest.mark.asyncio
    async def test_booking_stuck_sends_nudge(self, agent):
        """Patient stuck in booking beyond 48h should receive a nudge."""
        diary = _seed_monitoring_diary()
        diary.header.current_phase = Phase.BOOKING
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(hours=49)

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        assert len(result.responses) > 0
        msg = result.responses[0].message.lower()
        assert "slot" in msg or "appointment" in msg

    @pytest.mark.asyncio
    async def test_recently_entered_phase_no_nudge(self, agent):
        """Patient who just entered a phase should NOT get a nudge."""
        diary = _seed_monitoring_diary()
        diary.header.current_phase = Phase.INTAKE
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(hours=1)

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        # Should not be a staleness nudge
        stale_entries = [e for e in diary.monitoring.entries if "phase_stale" in e.type]
        assert len(stale_entries) == 0

    @pytest.mark.asyncio
    async def test_monitoring_phase_never_stales(self, agent):
        """Monitoring phase is long-lived and should never trigger staleness."""
        diary = _seed_monitoring_diary()
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(days=365)

        event = _heartbeat("PT-P0")
        result = await agent.process(event, diary)

        stale_entries = [e for e in diary.monitoring.entries if "phase_stale" in e.type]
        assert len(stale_entries) == 0

    @pytest.mark.asyncio
    async def test_staleness_nudge_not_repeated(self, agent):
        """Staleness nudge should only fire once per phase (no spam)."""
        diary = _seed_monitoring_diary()
        diary.header.current_phase = Phase.INTAKE
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(hours=73)

        event = _heartbeat("PT-P0")
        # First heartbeat — should fire nudge
        result1 = await agent.process(event, diary)
        assert len(result1.responses) > 0

        # Second heartbeat — should NOT fire again
        event2 = _heartbeat("PT-P0")
        result2 = await agent.process(event2, diary)

        stale_entries = [e for e in diary.monitoring.entries if "phase_stale" in e.type]
        assert len(stale_entries) == 1  # still just one

    @pytest.mark.asyncio
    async def test_staleness_adds_alert(self, agent):
        """Phase staleness should add to alerts_fired."""
        diary = _seed_monitoring_diary()
        diary.header.current_phase = Phase.CLINICAL
        diary.header.phase_entered_at = datetime.now(timezone.utc) - timedelta(hours=80)

        event = _heartbeat("PT-P0")
        await agent.process(event, diary)

        assert any("phase stale" in a.lower() for a in diary.monitoring.alerts_fired)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P0 #3: Per-Patient Rate Limiting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MockDiaryStore:
    """In-memory diary store for gateway tests."""

    def __init__(self):
        self._diaries: dict[str, tuple[PatientDiary, int]] = {}

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
        self._diaries[patient_id] = (diary, generation)


class EchoAgent:
    """Simple agent that echoes back for gateway-level tests."""
    agent_name = "echo"

    async def process(self, event, diary):
        return AgentResult(
            updated_diary=diary,
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel="websocket",
                    message=f"Echo: {event.payload.get('text', '')}",
                    metadata={"patient_id": event.patient_id},
                )
            ],
        )


class TestRateLimiting:
    """Tests for per-patient message rate limiting at the Gateway level."""

    @pytest.fixture
    def diary_store(self):
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-RATE")
        store.seed("PT-RATE", diary)
        diary2 = PatientDiary.create_new("PT-RATE2")
        store.seed("PT-RATE2", diary2)
        return store

    @pytest.fixture
    def dispatcher_registry(self):
        registry = DispatcherRegistry()
        mock_dispatcher = MagicMock()
        mock_dispatcher.channel_name = "websocket"
        mock_dispatcher.send = AsyncMock(
            return_value=DeliveryResult(
                success=True, channel="websocket", recipient="patient"
            )
        )
        registry.register(mock_dispatcher)
        return registry

    @pytest.fixture
    def gateway(self, diary_store, dispatcher_registry):
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        agent = EchoAgent()
        gw.register_agent("intake", agent)
        gw.register_agent("clinical", agent)
        gw.register_agent("booking", agent)
        gw.register_agent("monitoring", agent)
        return gw

    @pytest.mark.asyncio
    async def test_under_limit_succeeds(self, gateway):
        """Messages under the rate limit should go through."""
        for i in range(RATE_LIMIT_MAX_MESSAGES):
            event = _user_msg("PT-RATE", f"Message {i}")
            result = await gateway.process_event(event)
            assert result is not None
            # Should be an echo, not a rate limit response
            assert any("Echo" in r.message for r in result.responses)

    @pytest.mark.asyncio
    async def test_over_limit_blocked(self, gateway):
        """Messages over the rate limit should be blocked with friendly response."""
        # Send exactly the limit
        for i in range(RATE_LIMIT_MAX_MESSAGES):
            event = _user_msg("PT-RATE", f"Message {i}")
            await gateway.process_event(event)

        # The next message should be rate limited
        event = _user_msg("PT-RATE", "One more")
        result = await gateway.process_event(event)
        assert result is not None
        assert any("wait" in r.message.lower() for r in result.responses)
        assert any(
            r.metadata.get("rate_limited") for r in result.responses
        )

    @pytest.mark.asyncio
    async def test_rate_limit_per_patient(self, gateway):
        """Rate limiting should be per-patient, not global."""
        # Exhaust rate limit for PT-RATE
        for i in range(RATE_LIMIT_MAX_MESSAGES):
            event = _user_msg("PT-RATE", f"Message {i}")
            await gateway.process_event(event)

        # PT-RATE2 should still work
        event = _user_msg("PT-RATE2", "Hello")
        result = await gateway.process_event(event)
        assert result is not None
        assert any("Echo" in r.message for r in result.responses)

    @pytest.mark.asyncio
    async def test_heartbeat_not_rate_limited(self, gateway):
        """Internal events (heartbeats) should bypass rate limiting."""
        # Exhaust rate limit with user messages
        for i in range(RATE_LIMIT_MAX_MESSAGES + 2):
            event = _user_msg("PT-RATE", f"Spam {i}")
            await gateway.process_event(event)

        # Heartbeat should still go through (routed to monitoring)
        hb = _heartbeat("PT-RATE")
        result = await gateway.process_event(hb)
        # Should not be rate limited (no "wait" message)
        if result and result.responses:
            assert not any("wait" in r.message.lower() for r in result.responses)

    @pytest.mark.asyncio
    async def test_rate_limit_logged(self, gateway):
        """Rate-limited events should appear in the event log."""
        for i in range(RATE_LIMIT_MAX_MESSAGES + 1):
            event = _user_msg("PT-RATE", f"Msg {i}")
            await gateway.process_event(event)

        log = gateway.get_event_log(patient_id="PT-RATE")
        rate_limited = [e for e in log if e["status"] == "RATE_LIMITED"]
        assert len(rate_limited) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P0 Integration: Phase Transition Tracking in Gateway
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PhaseAdvancingAgent:
    """Agent that advances the diary phase for testing phase tracking."""
    agent_name = "advancer"

    def __init__(self, target_phase: Phase):
        self._target_phase = target_phase

    async def process(self, event, diary):
        diary.header.current_phase = self._target_phase
        return AgentResult(
            updated_diary=diary,
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel="websocket",
                    message=f"Advanced to {self._target_phase.value}",
                    metadata={"patient_id": event.patient_id},
                )
            ],
        )


class TestPhaseTransitionTracking:
    """Tests that the Gateway stamps phase_entered_at on phase changes."""

    @pytest.fixture
    def diary_store(self):
        store = MockDiaryStore()
        diary = PatientDiary.create_new("PT-PHASE")
        store.seed("PT-PHASE", diary)
        return store

    @pytest.fixture
    def dispatcher_registry(self):
        registry = DispatcherRegistry()
        mock_dispatcher = MagicMock()
        mock_dispatcher.channel_name = "websocket"
        mock_dispatcher.send = AsyncMock(
            return_value=DeliveryResult(
                success=True, channel="websocket", recipient="patient"
            )
        )
        registry.register(mock_dispatcher)
        return registry

    @pytest.mark.asyncio
    async def test_phase_change_updates_entered_at(self, diary_store, dispatcher_registry):
        """When agent changes phase, Gateway should stamp phase_entered_at."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        gw.register_agent("intake", PhaseAdvancingAgent(Phase.CLINICAL))

        before = datetime.now(timezone.utc)
        event = _user_msg("PT-PHASE", "Hello")
        result = await gw.process_event(event)
        after = datetime.now(timezone.utc)

        diary = result.updated_diary
        assert diary.header.current_phase == Phase.CLINICAL
        assert diary.header.phase_entered_at >= before
        assert diary.header.phase_entered_at <= after

    @pytest.mark.asyncio
    async def test_no_phase_change_preserves_entered_at(self, diary_store, dispatcher_registry):
        """When phase doesn't change, phase_entered_at should not be updated."""
        gw = Gateway(
            diary_store=diary_store,
            dispatcher_registry=dispatcher_registry,
        )
        # This agent keeps phase as INTAKE (same as default)
        gw.register_agent("intake", PhaseAdvancingAgent(Phase.INTAKE))

        diary = PatientDiary.create_new("PT-PHASE2")
        original_entered = datetime.now(timezone.utc) - timedelta(hours=10)
        diary.header.phase_entered_at = original_entered
        diary_store.seed("PT-PHASE2", diary)

        event = _user_msg("PT-PHASE2", "Hello")
        result = await gw.process_event(event)

        # phase_entered_at should NOT have been updated
        assert result.updated_diary.header.phase_entered_at == original_entered

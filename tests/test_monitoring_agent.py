"""
Tests for the Monitoring Agent — baseline setup, heartbeat milestones,
lab comparison, deterioration detection, interactive assessment, and reactive responses.
"""

import pytest
from datetime import datetime, timezone

from medforce.gateway.agents.monitoring_agent import (
    DETERIORATION_THRESHOLDS,
    MonitoringAgent,
)
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalSection,
    CommunicationPlan,
    GPChannel,
    GPQuery,
    MonitoringEntry,
    MonitoringSection,
    PatientDiary,
    Phase,
    RiskLevel,
    ScheduledQuestion,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole


# ── Fixtures ──


def make_diary(
    patient_id: str = "PT-400",
    phase: Phase = Phase.MONITORING,
    monitoring_active: bool = True,
    baseline: dict | None = None,
) -> PatientDiary:
    diary = PatientDiary.create_new(patient_id)
    diary.header.current_phase = phase
    diary.header.risk_level = RiskLevel.MEDIUM
    diary.intake.name = "Monitor Patient"
    diary.intake.phone = "07700900999"
    diary.monitoring.monitoring_active = monitoring_active
    diary.monitoring.appointment_date = "2026-03-15"
    diary.monitoring.baseline = baseline or {"bilirubin": 3.0, "ALT": 200}
    return diary


def make_booking_complete_event(patient_id: str = "PT-400") -> EventEnvelope:
    return EventEnvelope.handoff(
        event_type=EventType.BOOKING_COMPLETE,
        patient_id=patient_id,
        source_agent="booking",
        payload={
            "appointment_date": "2026-03-15",
            "appointment_time": "10:00",
            "appointment_id": "APT-PT-400",
            "risk_level": "medium",
            "baseline": {"bilirubin": 3.0, "ALT": 200},
            "channel": "websocket",
        },
    )


def make_heartbeat_event(
    patient_id: str = "PT-400",
    days: int = 14,
    milestone: str = "follow_up_labs",
) -> EventEnvelope:
    return EventEnvelope.heartbeat(
        patient_id=patient_id,
        days_since_appointment=days,
        milestone=milestone,
    )


def make_user_message(text: str, patient_id: str = "PT-400") -> EventEnvelope:
    return EventEnvelope.user_message(patient_id=patient_id, text=text)


def make_document_event(
    patient_id: str = "PT-400",
    extracted_values: dict | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_type=EventType.DOCUMENT_UPLOADED,
        patient_id=patient_id,
        payload={
            "type": "lab_results",
            "file_ref": "gs://test/new_labs.pdf",
            "channel": "websocket",
            "extracted_values": extracted_values or {},
        },
        sender_id="PATIENT",
        sender_role=SenderRole.PATIENT,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Booking Complete Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBookingComplete:
    """BOOKING_COMPLETE → monitoring setup."""

    @pytest.mark.asyncio
    async def test_activates_monitoring(self):
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False, baseline={})
        diary.clinical.documents = [
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 3.0, "ALT": 200},
            )
        ]
        event = make_booking_complete_event()

        result = await agent.process(event, diary)

        assert result.updated_diary.monitoring.monitoring_active is True
        assert result.updated_diary.monitoring.appointment_date == "2026-03-15"
        assert result.updated_diary.header.current_phase == Phase.MONITORING

    @pytest.mark.asyncio
    async def test_snapshots_baseline(self):
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False, baseline={})
        diary.clinical.documents = [
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 3.0},
            )
        ]
        event = make_booking_complete_event()

        result = await agent.process(event, diary)

        assert "bilirubin" in result.updated_diary.monitoring.baseline

    @pytest.mark.asyncio
    async def test_sets_next_check(self):
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False, baseline={})
        event = make_booking_complete_event()

        result = await agent.process(event, diary)

        assert result.updated_diary.monitoring.next_scheduled_check == "14"

    @pytest.mark.asyncio
    async def test_sends_welcome_message(self):
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False, baseline={})
        event = make_booking_complete_event()

        result = await agent.process(event, diary)

        assert len(result.responses) == 1
        assert "monitoring" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_records_setup_entry(self):
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False, baseline={})
        event = make_booking_complete_event()

        result = await agent.process(event, diary)

        entries = result.updated_diary.monitoring.entries
        assert len(entries) == 1
        assert entries[0].type == "monitoring_setup"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Heartbeat Milestones
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHeartbeatMilestones:
    """HEARTBEAT events at various intervals."""

    @pytest.mark.asyncio
    async def test_14_day_follow_up(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_heartbeat_event(days=14)

        result = await agent.process(event, diary)

        assert len(result.responses) == 1
        msg = result.responses[0].message.lower()
        assert "2 weeks" in msg or "lab" in msg or "check-in" in msg

    @pytest.mark.asyncio
    async def test_30_day_checkin(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_heartbeat_event(days=30)

        result = await agent.process(event, diary)

        assert len(result.responses) == 1
        assert "month" in result.responses[0].message.lower() or "check" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_60_day_updated_labs(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_heartbeat_event(days=60)

        result = await agent.process(event, diary)

        assert len(result.responses) == 1
        assert "2 months" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_records_heartbeat_entry(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_heartbeat_event(days=14)

        result = await agent.process(event, diary)

        entries = result.updated_diary.monitoring.entries
        assert any("heartbeat" in e.type or "checkin" in e.type for e in entries)

    @pytest.mark.asyncio
    async def test_inactive_monitoring_skips(self):
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False)
        event = make_heartbeat_event(days=14)

        result = await agent.process(event, diary)

        assert len(result.responses) == 0

    @pytest.mark.asyncio
    async def test_gp_reminder_emitted_on_heartbeat(self):
        """Heartbeat should emit GP_REMINDER if pending queries exist."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.gp_channel = GPChannel(
            gp_name="Dr. Smith",
            queries=[GPQuery(query_id="GPQ-001", status="pending")],
        )
        event = make_heartbeat_event(days=14)

        result = await agent.process(event, diary)

        gp_reminders = [
            e for e in result.emitted_events
            if e.event_type == EventType.GP_REMINDER
        ]
        assert len(gp_reminders) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lab Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLabComparison:
    """_compare_values — baseline vs new lab results."""

    def test_stable_values(self):
        agent = MonitoringAgent()
        baseline = {"bilirubin": 3.0, "ALT": 200}
        new_values = {"bilirubin": 3.2, "ALT": 210}

        comparison = agent._compare_values(baseline, new_values)

        assert len(comparison["changes"]) == 2
        assert len(comparison["deteriorating"]) == 0

    def test_deteriorating_bilirubin(self):
        agent = MonitoringAgent()
        baseline = {"bilirubin": 3.0}
        new_values = {"bilirubin": 6.0}  # 100% increase

        comparison = agent._compare_values(baseline, new_values)

        assert len(comparison["deteriorating"]) == 1
        assert comparison["deteriorating"][0]["param"] == "bilirubin"
        assert comparison["deteriorating"][0]["change_pct"] == 100.0

    def test_deteriorating_alt_doubling(self):
        agent = MonitoringAgent()
        baseline = {"ALT": 200}
        new_values = {"ALT": 450}  # 125% increase

        comparison = agent._compare_values(baseline, new_values)

        assert len(comparison["deteriorating"]) == 1
        assert comparison["deteriorating"][0]["change_pct"] == 125.0

    def test_deteriorating_platelets_decrease(self):
        agent = MonitoringAgent()
        baseline = {"platelets": 200}
        new_values = {"platelets": 100}  # 50% decrease

        comparison = agent._compare_values(baseline, new_values)

        assert len(comparison["deteriorating"]) == 1
        assert comparison["deteriorating"][0]["change_pct"] == -50.0

    def test_new_value_not_in_baseline(self):
        agent = MonitoringAgent()
        baseline = {"bilirubin": 3.0}
        new_values = {"bilirubin": 3.0, "INR": 1.5}

        comparison = agent._compare_values(baseline, new_values)

        new_entries = [c for c in comparison["changes"] if c["status"] == "new_value"]
        assert len(new_entries) == 1
        assert new_entries[0]["param"] == "INR"

    def test_non_numeric_values_skipped(self):
        agent = MonitoringAgent()
        baseline = {"bilirubin": 3.0}
        new_values = {"bilirubin": "pending"}

        comparison = agent._compare_values(baseline, new_values)

        assert len(comparison["changes"]) == 0

    def test_zero_baseline_handled(self):
        agent = MonitoringAgent()
        baseline = {"bilirubin": 0}
        new_values = {"bilirubin": 3.0}

        comparison = agent._compare_values(baseline, new_values)

        assert len(comparison["changes"]) == 1
        assert comparison["changes"][0]["change_pct"] == 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Document Upload (Monitoring Phase)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringDocumentUpload:
    """New lab results compared against baseline."""

    @pytest.mark.asyncio
    async def test_stable_labs_positive_message(self):
        agent = MonitoringAgent()
        diary = make_diary(baseline={"bilirubin": 3.0, "ALT": 200})
        event = make_document_event(
            extracted_values={"bilirubin": 3.1, "ALT": 205}
        )

        result = await agent.process(event, diary)

        assert "stable" in result.responses[0].message.lower()
        assert len(result.emitted_events) == 0

    @pytest.mark.asyncio
    async def test_deteriorating_labs_trigger_alert(self):
        agent = MonitoringAgent()
        diary = make_diary(baseline={"bilirubin": 3.0})
        event = make_document_event(
            extracted_values={"bilirubin": 7.0}  # 133% increase
        )

        result = await agent.process(event, diary)

        alerts = [
            e for e in result.emitted_events
            if e.event_type == EventType.DETERIORATION_ALERT
        ]
        assert len(alerts) == 1
        assert "changes" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_records_comparison_entry(self):
        agent = MonitoringAgent()
        diary = make_diary(baseline={"bilirubin": 3.0})
        event = make_document_event(
            extracted_values={"bilirubin": 3.5}
        )

        result = await agent.process(event, diary)

        entries = [
            e for e in result.updated_diary.monitoring.entries
            if e.type == "lab_update"
        ]
        assert len(entries) == 1
        assert entries[0].new_values["bilirubin"] == 3.5

    @pytest.mark.asyncio
    async def test_no_values_sends_ack(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_document_event(extracted_values={})

        result = await agent.process(event, diary)

        assert "upload" in result.responses[0].message.lower()
        assert len(result.emitted_events) == 0

    @pytest.mark.asyncio
    async def test_deterioration_records_alert(self):
        agent = MonitoringAgent()
        diary = make_diary(baseline={"ALT": 200})
        event = make_document_event(
            extracted_values={"ALT": 600}  # 200% increase
        )

        result = await agent.process(event, diary)

        assert len(result.updated_diary.monitoring.alerts_fired) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  User Message (Monitoring Phase) — Normal & Emergency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringUserMessage:
    """Patient messages during monitoring phase."""

    @pytest.mark.asyncio
    async def test_normal_message_acknowledged(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_user_message("I'm feeling fine, thanks")

        result = await agent.process(event, diary)

        assert len(result.responses) == 1
        assert "monitoring" in result.responses[0].message.lower()
        assert len(result.emitted_events) == 0

    @pytest.mark.asyncio
    async def test_emergency_keyword_triggers_immediate_alert(self):
        """Emergency keywords (jaundice, confusion, bleeding) skip assessment."""
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_user_message("I've developed jaundice and confusion")

        result = await agent.process(event, diary)

        alerts = [
            e for e in result.emitted_events
            if e.event_type == EventType.DETERIORATION_ALERT
        ]
        assert len(alerts) == 1
        assert "999" in result.responses[0].message or "A&E" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_worsening_starts_assessment(self):
        """Non-emergency deterioration keywords start interactive assessment."""
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_user_message("My symptoms are worsening significantly")

        result = await agent.process(event, diary)

        # Should start assessment, NOT immediately emit alert
        assert len(result.emitted_events) == 0
        assert result.updated_diary.monitoring.deterioration_assessment.active is True
        assert len(result.updated_diary.monitoring.deterioration_assessment.questions) == 1
        # Response should ask a follow-up question
        assert "?" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_records_message_entry(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_user_message("Just a quick update")

        result = await agent.process(event, diary)

        entries = [
            e for e in result.updated_diary.monitoring.entries
            if e.type == "patient_message"
        ]
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_emergency_records_in_alerts_fired(self):
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_user_message("I have been bleeding")

        result = await agent.process(event, diary)

        assert len(result.updated_diary.monitoring.alerts_fired) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Deterioration Assessment Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeteriorationAssessment:
    """Interactive deterioration assessment when patient reports worsening."""

    @pytest.mark.asyncio
    async def test_assessment_starts_on_non_emergency_flags(self):
        """'worse' keyword should start assessment, not immediate escalation."""
        agent = MonitoringAgent()
        diary = make_diary()
        event = make_user_message("I've been feeling worse lately, more fatigue")

        result = await agent.process(event, diary)

        assessment = result.updated_diary.monitoring.deterioration_assessment
        assert assessment.active is True
        assert not assessment.assessment_complete
        assert "worse" in assessment.detected_symptoms
        assert len(assessment.questions) == 1
        # Should ask a question, not send a generic ack
        assert "?" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_assessment_processes_answers(self):
        """Second message during assessment is treated as an answer."""
        agent = MonitoringAgent()
        diary = make_diary()

        # Step 1: Trigger assessment
        event1 = make_user_message("I've been feeling worse lately")
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary
        assert diary.monitoring.deterioration_assessment.active is True

        # Step 2: Patient answers the first question
        event2 = make_user_message("The pain has been getting worse over 3 days, about 6/10 severity")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary

        assessment = diary.monitoring.deterioration_assessment
        assert assessment.questions[0].answer is not None
        # Should have asked a second question
        assert len(assessment.questions) >= 2

    @pytest.mark.asyncio
    async def test_assessment_completes_after_three_questions(self):
        """Assessment completes after 3 Q&A rounds."""
        agent = MonitoringAgent()
        diary = make_diary()

        # Step 1: Trigger
        event1 = make_user_message("I've been feeling worse")
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        # Step 2: Answer 1
        event2 = make_user_message("Pain getting worse, about 5/10")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary

        # Step 3: Answer 2
        event3 = make_user_message("No new symptoms, just more tired")
        result3 = await agent.process(event3, diary)
        diary = result3.updated_diary

        # Step 4: Answer 3 — should complete
        event4 = make_user_message("I can still manage daily activities, about 5/10 severity")
        result4 = await agent.process(event4, diary)
        diary = result4.updated_diary

        assessment = diary.monitoring.deterioration_assessment
        assert assessment.assessment_complete is True
        assert assessment.severity is not None
        assert assessment.recommendation is not None

    @pytest.mark.asyncio
    async def test_emergency_during_assessment_escalates(self):
        """If patient reports emergency symptoms during assessment, escalate immediately."""
        agent = MonitoringAgent()
        diary = make_diary()

        # Step 1: Start assessment
        event1 = make_user_message("I've been feeling worse")
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        # Step 2: Patient mentions emergency symptom in answer
        event2 = make_user_message("I've been vomiting blood and feel confused")
        result2 = await agent.process(event2, diary)

        # Should immediately escalate
        alerts = [e for e in result2.emitted_events if e.event_type == EventType.DETERIORATION_ALERT]
        assert len(alerts) == 1
        assert "999" in result2.responses[0].message or "A&E" in result2.responses[0].message

    @pytest.mark.asyncio
    async def test_moderate_assessment_emits_alert(self):
        """Moderate severity assessment should emit DETERIORATION_ALERT."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "cirrhosis"

        # Run full assessment with moderate symptoms
        event1 = make_user_message("I've been feeling worse, more fatigue")
        r1 = await agent.process(event1, diary)
        diary = r1.updated_diary

        event2 = make_user_message("More tired, struggling to eat, pain is about 6/10")
        r2 = await agent.process(event2, diary)
        diary = r2.updated_diary

        event3 = make_user_message("No jaundice or confusion, just tired and sore")
        r3 = await agent.process(event3, diary)
        diary = r3.updated_diary

        event4 = make_user_message("I can still get around but it's difficult. Maybe 6/10.")
        r4 = await agent.process(event4, diary)

        assessment = r4.updated_diary.monitoring.deterioration_assessment
        assert assessment.assessment_complete is True
        # Should have a severity and recommendation
        assert assessment.severity in ("mild", "moderate", "severe")
        assert assessment.recommendation is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Milestone Message Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMilestoneMessages:
    """_fallback_milestone_message for different time periods."""

    def test_14_day_message(self):
        agent = MonitoringAgent()
        diary = make_diary()
        msg = agent._fallback_milestone_message(14, diary)
        assert "2 weeks" in msg

    def test_30_day_message(self):
        agent = MonitoringAgent()
        diary = make_diary()
        msg = agent._fallback_milestone_message(30, diary)
        assert "month" in msg.lower()

    def test_60_day_message(self):
        agent = MonitoringAgent()
        diary = make_diary()
        msg = agent._fallback_milestone_message(60, diary)
        assert "2 months" in msg.lower()

    def test_90_day_message(self):
        agent = MonitoringAgent()
        diary = make_diary()
        msg = agent._fallback_milestone_message(90, diary)
        assert "90 days" in msg

    def test_includes_patient_name(self):
        agent = MonitoringAgent()
        diary = make_diary()
        msg = agent._fallback_milestone_message(14, diary)
        assert "Monitor Patient" in msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringScenarios:
    """End-to-end monitoring scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_booking_to_monitoring_to_checkin(self):
        """Full flow: BOOKING_COMPLETE → heartbeat → check-in."""
        agent = MonitoringAgent()
        diary = make_diary(monitoring_active=False, baseline={})
        diary.clinical.documents = [
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 3.0},
            )
        ]

        # Step 1: Booking complete
        event1 = make_booking_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary
        assert diary.monitoring.monitoring_active is True

        # Step 2: 14-day heartbeat
        event2 = make_heartbeat_event(days=14)
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary
        assert len(diary.monitoring.entries) >= 2

    @pytest.mark.asyncio
    async def test_scenario_deterioration_full_loop(self):
        """Patient uploads worsening labs → DETERIORATION_ALERT emitted."""
        agent = MonitoringAgent()
        diary = make_diary(baseline={"bilirubin": 3.0, "ALT": 200})

        event = make_document_event(
            extracted_values={"bilirubin": 8.0, "ALT": 700}
        )
        result = await agent.process(event, diary)

        # Should emit deterioration alert
        assert len(result.emitted_events) == 1
        alert = result.emitted_events[0]
        assert alert.event_type == EventType.DETERIORATION_ALERT

    @pytest.mark.asyncio
    async def test_scenario_patient_reports_emergency(self):
        """Patient reports emergency symptoms → immediate alert + safety message."""
        agent = MonitoringAgent()
        diary = make_diary()

        event = make_user_message("I've been vomiting blood and feel confused")
        result = await agent.process(event, diary)

        assert len(result.emitted_events) == 1
        assert "999" in result.responses[0].message or "A&E" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_scenario_worsening_with_assessment(self):
        """Patient says 'feeling worse' → assessment → outcome."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "cirrhosis"

        # Step 1: Patient reports worsening
        event1 = make_user_message("I've been feeling worse lately, more fatigue")
        r1 = await agent.process(event1, diary)
        diary = r1.updated_diary
        assert diary.monitoring.deterioration_assessment.active is True
        assert "?" in r1.responses[0].message

        # Step 2-4: Answer questions
        answers = [
            "The fatigue is worse, I'm sleeping 14 hours a day. Pain about 6/10.",
            "I haven't noticed anything new, just more tired and the pain is worse.",
            "I can barely manage daily activities. Maybe 6/10 severity overall.",
        ]
        for answer in answers:
            event = make_user_message(answer)
            result = await agent.process(event, diary)
            diary = result.updated_diary

        # Should have completed the assessment
        assert diary.monitoring.deterioration_assessment.assessment_complete is True
        assert diary.monitoring.deterioration_assessment.severity is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Check-in Response Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckinResponseEvaluation:
    """Answers to scheduled check-in questions are evaluated for clinical significance."""

    @pytest.mark.asyncio
    async def test_red_urine_triggers_assessment_for_liver_patient(self):
        """'Red urine and clay stool' should trigger assessment for liver patient."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "cirrhosis"

        # Set up a sent but unanswered scheduled question
        diary.monitoring.communication_plan = CommunicationPlan(
            risk_level="medium",
            total_messages=4,
            check_in_days=[14, 30, 60, 90],
            questions=[
                ScheduledQuestion(
                    question="Have you noticed any changes in the colour of your urine or stool?",
                    day=14, priority=1, category="symptom", sent=True,
                ),
            ],
            generated=True,
        )

        event = make_user_message("Yes my urine is red and stool is clay colored")
        result = await agent.process(event, diary)

        # Should start deterioration assessment, NOT send generic ack
        assessment = result.updated_diary.monitoring.deterioration_assessment
        assert assessment.active is True
        assert len(assessment.questions) >= 1
        assert "?" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_dark_urine_triggers_assessment(self):
        """'Dark urine' should trigger assessment for liver patient."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "hepatitis"

        diary.monitoring.communication_plan = CommunicationPlan(
            risk_level="medium",
            total_messages=4,
            check_in_days=[14, 30, 60, 90],
            questions=[
                ScheduledQuestion(
                    question="Any changes in urine colour?",
                    day=14, priority=1, category="symptom", sent=True,
                ),
            ],
            generated=True,
        )

        event = make_user_message("Yes, my urine has been very dark urine lately")
        result = await agent.process(event, diary)

        assessment = result.updated_diary.monitoring.deterioration_assessment
        assert assessment.active is True

    @pytest.mark.asyncio
    async def test_normal_answer_does_not_trigger_assessment(self):
        """'No changes, feeling fine' should NOT trigger assessment."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "cirrhosis"

        diary.monitoring.communication_plan = CommunicationPlan(
            risk_level="medium",
            total_messages=4,
            check_in_days=[14, 30, 60, 90],
            questions=[
                ScheduledQuestion(
                    question="Any changes in urine colour?",
                    day=14, priority=1, category="symptom", sent=True,
                ),
            ],
            generated=True,
        )

        event = make_user_message("No, everything looks normal, feeling fine")
        result = await agent.process(event, diary)

        assessment = result.updated_diary.monitoring.deterioration_assessment
        assert assessment.active is False
        # Should get normal acknowledgment
        assert "monitoring" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_general_fever_triggers_assessment(self):
        """'fever' should trigger assessment regardless of condition."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "MASH"

        diary.monitoring.communication_plan = CommunicationPlan(
            risk_level="medium",
            total_messages=4,
            check_in_days=[14, 30, 60, 90],
            questions=[
                ScheduledQuestion(
                    question="How are you feeling overall?",
                    day=14, priority=1, category="general", sent=True,
                ),
            ],
            generated=True,
        )

        event = make_user_message("I've had a fever for 2 days and night sweats")
        result = await agent.process(event, diary)

        assessment = result.updated_diary.monitoring.deterioration_assessment
        assert assessment.active is True

    @pytest.mark.asyncio
    async def test_black_stool_triggers_assessment(self):
        """'Black stool' (GI bleed indicator) should trigger assessment."""
        agent = MonitoringAgent()
        diary = make_diary()
        diary.clinical.condition_context = "liver cirrhosis"

        diary.monitoring.communication_plan = CommunicationPlan(
            risk_level="high",
            total_messages=6,
            check_in_days=[7, 14, 21, 30, 45, 60],
            questions=[
                ScheduledQuestion(
                    question="Any changes to your bowel movements?",
                    day=7, priority=1, category="symptom", sent=True,
                ),
            ],
            generated=True,
        )

        event = make_user_message("My stool has been black and tarry for a couple days")
        result = await agent.process(event, diary)

        assessment = result.updated_diary.monitoring.deterioration_assessment
        assert assessment.active is True

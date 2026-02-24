"""
Tests for the Heartbeat Scheduler — registration, milestone detection,
GP reminder CRON, startup recovery.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from medforce.gateway.heartbeat import (
    MILESTONE_DAYS,
    HeartbeatScheduler,
)
from medforce.gateway.diary import (
    GPChannel,
    GPQuery,
    MonitoringEntry,
    MonitoringSection,
    PatientDiary,
    Phase,
)


# ── Mock Diary Store ──


class MockDiaryStore:
    def __init__(self, diaries: dict | None = None):
        self._diaries = diaries or {}

    def load(self, patient_id):
        if patient_id not in self._diaries:
            raise Exception(f"Not found: {patient_id}")
        diary, gen = self._diaries[patient_id]
        return diary, gen

    def list_monitoring_patients(self):
        return [
            pid for pid, (diary, _) in self._diaries.items()
            if diary.monitoring.monitoring_active
        ]


def make_monitoring_diary(
    patient_id: str = "PT-HB-001",
    appointment_date: str = "2026-01-01",
    entries: list | None = None,
) -> PatientDiary:
    diary = PatientDiary.create_new(patient_id)
    diary.header.current_phase = Phase.MONITORING
    diary.monitoring.monitoring_active = True
    diary.monitoring.appointment_date = appointment_date
    diary.monitoring.baseline = {"bilirubin": 3.0}
    if entries:
        diary.monitoring.entries = entries
    return diary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRegistration:
    """Register/unregister patients."""

    def test_register_patient(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        scheduler.register("PT-001", "2026-03-15")
        assert "PT-001" in scheduler.monitored_patients
        assert scheduler.monitored_count == 1

    def test_unregister_patient(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        scheduler.register("PT-001")
        scheduler.unregister("PT-001")
        assert "PT-001" not in scheduler.monitored_patients

    def test_unregister_nonexistent(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        scheduler.unregister("PT-NONE")  # Should not raise
        assert scheduler.monitored_count == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Milestone Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMilestoneDetection:
    """_is_milestone_due logic."""

    def test_14_day_milestone_fires(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        diary = make_monitoring_diary()
        result = scheduler._is_milestone_due(14, diary)
        assert result == "heartbeat_14d"

    def test_30_day_milestone_fires(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        diary = make_monitoring_diary()
        result = scheduler._is_milestone_due(30, diary)
        # 14d not yet fired, so 14d fires first
        assert result == "heartbeat_14d"

    def test_30_day_fires_after_14_done(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        diary = make_monitoring_diary(
            entries=[MonitoringEntry(date="2026-01-15", type="heartbeat_14d")]
        )
        result = scheduler._is_milestone_due(30, diary)
        assert result == "heartbeat_30d"

    def test_no_milestone_at_day_5(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        diary = make_monitoring_diary()
        result = scheduler._is_milestone_due(5, diary)
        assert result == ""

    def test_already_fired_does_not_repeat(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=MockDiaryStore()
        )
        diary = make_monitoring_diary(
            entries=[
                MonitoringEntry(date="2026-01-15", type="heartbeat_14d"),
                MonitoringEntry(date="2026-01-31", type="heartbeat_30d"),
                MonitoringEntry(date="2026-03-02", type="heartbeat_60d"),
                MonitoringEntry(date="2026-04-01", type="heartbeat_90d"),
            ]
        )
        result = scheduler._is_milestone_due(100, diary)
        assert result == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Days Since Booking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDaysSinceBooking:
    """_days_since_booking utility."""

    def test_calculates_days(self):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        result = HeartbeatScheduler._days_since_booking(yesterday)
        assert result == 10

    def test_invalid_date_returns_zero(self):
        result = HeartbeatScheduler._days_since_booking("not-a-date")
        assert result == 0

    def test_future_date_negative(self):
        future = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
        result = HeartbeatScheduler._days_since_booking(future)
        assert result < 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hours Since
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHoursSince:
    """_hours_since utility."""

    def test_hours_from_datetime(self):
        two_days_ago = datetime.now(timezone.utc) - timedelta(hours=50)
        result = HeartbeatScheduler._hours_since(two_days_ago)
        assert 49 < result < 51

    def test_hours_from_string(self):
        two_days_ago = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
        result = HeartbeatScheduler._hours_since(two_days_ago)
        assert 49 < result < 51

    def test_invalid_string_returns_zero(self):
        result = HeartbeatScheduler._hours_since("not-a-date")
        assert result == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Startup Recovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStartupRecovery:
    """_recover_on_startup scans GCS for monitored patients."""

    @pytest.mark.asyncio
    async def test_recovers_monitored_patients(self):
        diary1 = make_monitoring_diary("PT-REC-001")
        diary2 = make_monitoring_diary("PT-REC-002")
        store = MockDiaryStore({
            "PT-REC-001": (diary1, 1),
            "PT-REC-002": (diary2, 1),
        })

        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=store
        )
        await scheduler._recover_on_startup()

        assert "PT-REC-001" in scheduler.monitored_patients
        assert "PT-REC-002" in scheduler.monitored_patients

    @pytest.mark.asyncio
    async def test_recovery_handles_errors(self):
        store = MockDiaryStore()
        store.list_monitoring_patients = MagicMock(side_effect=Exception("GCS down"))

        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=store
        )
        await scheduler._recover_on_startup()

        # Should not crash
        assert scheduler.monitored_count == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Check Patient
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckPatient:
    """_check_patient fires heartbeats and GP reminders."""

    @pytest.mark.asyncio
    async def test_fires_heartbeat_at_milestone(self):
        # Set appointment date to 14 days ago
        appt_date = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
        diary = make_monitoring_diary(appointment_date=appt_date)
        store = MockDiaryStore({"PT-HB-001": (diary, 1)})

        processor = AsyncMock()
        scheduler = HeartbeatScheduler(
            processor=processor, diary_store=store
        )
        scheduler.register("PT-HB-001", appt_date)

        await scheduler._check_patient("PT-HB-001")

        # Should have called processor with a HEARTBEAT event
        assert processor.call_count >= 1
        call_args = processor.call_args_list[0]
        event = call_args[0][0]
        assert event.event_type.value == "HEARTBEAT"

    @pytest.mark.asyncio
    async def test_fires_gp_reminder_after_48h(self):
        diary = make_monitoring_diary()
        diary.gp_channel = GPChannel(
            gp_name="Dr. Test",
            queries=[
                GPQuery(
                    query_id="GPQ-TEST",
                    status="pending",
                    sent=datetime.now(timezone.utc) - timedelta(hours=50),
                ),
            ],
        )
        store = MockDiaryStore({"PT-HB-001": (diary, 1)})

        processor = AsyncMock()
        scheduler = HeartbeatScheduler(
            processor=processor, diary_store=store
        )
        scheduler.register("PT-HB-001")

        await scheduler._check_patient("PT-HB-001")

        # Should have called processor with GP_REMINDER
        gp_reminder_calls = [
            call for call in processor.call_args_list
            if call[0][0].event_type.value == "GP_REMINDER"
        ]
        assert len(gp_reminder_calls) >= 1

    @pytest.mark.asyncio
    async def test_unregisters_inactive_patient(self):
        diary = make_monitoring_diary()
        diary.monitoring.monitoring_active = False
        store = MockDiaryStore({"PT-HB-001": (diary, 1)})

        scheduler = HeartbeatScheduler(
            processor=AsyncMock(), diary_store=store
        )
        scheduler.register("PT-HB-001")

        await scheduler._check_patient("PT-HB-001")

        assert "PT-HB-001" not in scheduler.monitored_patients


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Start / Stop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStartStop:
    """Lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(),
            diary_store=MockDiaryStore(),
            check_interval=3600,
        )
        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_double_start_warns(self):
        scheduler = HeartbeatScheduler(
            processor=AsyncMock(),
            diary_store=MockDiaryStore(),
        )
        await scheduler.start()
        await scheduler.start()  # Should not crash
        await scheduler.stop()

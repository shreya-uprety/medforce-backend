"""
Heartbeat Scheduler — Background loop that fires HEARTBEAT events
for monitored patients at milestone intervals.

Features:
  - In-process asyncio loop (no external dependencies)
  - Recovery on restart: scans GCS for monitoring_active patients
  - GP reminder CRON: fires GP_REMINDER for pending queries >48h
  - Register/unregister patients dynamically

Scaling path: swap to Google Cloud Scheduler hitting /api/gateway/emit.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable

from medforce.gateway.events import EventEnvelope, EventType

logger = logging.getLogger("gateway.heartbeat")

# Milestone days when heartbeats should fire
MILESTONE_DAYS = [14, 30, 60, 90]

# How often the loop checks (in seconds)
CHECK_INTERVAL = 3600  # 1 hour


class HeartbeatScheduler:
    """
    Background loop that fires HEARTBEAT events for monitored patients.

    Usage:
        scheduler = HeartbeatScheduler(
            processor=gateway.process_event,
            diary_store=diary_store,
        )
        await scheduler.start()

    On shutdown:
        await scheduler.stop()
    """

    def __init__(
        self,
        processor: Callable[[EventEnvelope], Awaitable[Any]],
        diary_store: Any,
        check_interval: int = CHECK_INTERVAL,
    ) -> None:
        self._processor = processor
        self._diary_store = diary_store
        self._check_interval = check_interval
        self._monitored: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def monitored_patients(self) -> list[str]:
        """List of currently monitored patient IDs."""
        return list(self._monitored.keys())

    @property
    def monitored_count(self) -> int:
        return len(self._monitored)

    async def start(self) -> None:
        """Start the heartbeat loop and recover monitored patients."""
        if self._running:
            logger.warning("HeartbeatScheduler already running")
            return

        self._running = True
        asyncio.create_task(self._recover_on_startup())  # background, non-blocking
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "HeartbeatScheduler started — monitoring %d patients",
            len(self._monitored),
        )

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("HeartbeatScheduler stopped")

    def register(self, patient_id: str, appointment_date: str | None = None) -> None:
        """Register a patient for monitoring heartbeats."""
        self._monitored[patient_id] = {
            "registered_at": datetime.now(timezone.utc),
            "appointment_date": appointment_date,
            "last_heartbeat": None,
        }
        logger.info("Registered patient %s for monitoring", patient_id)

    def unregister(self, patient_id: str) -> None:
        """Remove a patient from monitoring."""
        self._monitored.pop(patient_id, None)
        logger.info("Unregistered patient %s from monitoring", patient_id)

    # ── Internal ──

    async def _recover_on_startup(self) -> None:
        """Scan GCS for patients with monitoring_active=True."""
        try:
            patient_ids = self._diary_store.list_monitoring_patients()
            for pid in patient_ids:
                try:
                    diary, _ = self._diary_store.load(pid)
                    self.register(pid, diary.monitoring.appointment_date)
                except Exception as exc:
                    logger.warning(
                        "Failed to recover monitoring for %s: %s", pid, exc
                    )
            logger.info(
                "Recovered %d monitored patients on startup",
                len(patient_ids),
            )
        except Exception as exc:
            logger.error("Monitoring recovery failed: %s", exc)

    async def _heartbeat_loop(self) -> None:
        """Main loop — fires heartbeats and GP reminders periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                if not self._running:
                    break

                await self._check_all_patients()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat loop error: %s", exc, exc_info=True)

    async def _check_all_patients(self) -> None:
        """Check all monitored patients for milestones and GP reminders."""
        # Snapshot the keys to avoid mutation during iteration
        patient_ids = list(self._monitored.keys())

        for pid in patient_ids:
            try:
                await self._check_patient(pid)
            except Exception as exc:
                logger.warning(
                    "Error checking patient %s: %s", pid, exc
                )

    async def _check_patient(self, patient_id: str) -> None:
        """Check a single patient for milestone heartbeats and GP reminders."""
        try:
            diary, _ = self._diary_store.load(patient_id)
        except Exception:
            return

        # Skip if monitoring is no longer active
        if not diary.monitoring.monitoring_active:
            self.unregister(patient_id)
            return

        # Check milestone
        appointment_date = diary.monitoring.appointment_date
        if appointment_date:
            days_since = self._days_since_booking(appointment_date)
            milestone = self._is_milestone_due(days_since, diary)

            if milestone:
                event = EventEnvelope.heartbeat(
                    patient_id=patient_id,
                    days_since_appointment=days_since,
                    milestone=milestone,
                )
                try:
                    await self._processor(event)
                    info = self._monitored.get(patient_id)
                    if info:
                        info["last_heartbeat"] = datetime.now(timezone.utc)
                except Exception as exc:
                    logger.error(
                        "Failed to process heartbeat for %s: %s",
                        patient_id, exc,
                    )

        # Check GP reminders (pending queries older than 48h)
        if diary.gp_channel.has_pending_queries():
            for query in diary.gp_channel.get_pending_queries():
                hours_since_sent = self._hours_since(query.sent)
                if hours_since_sent > 48 and query.reminder_sent is None:
                    event = EventEnvelope.handoff(
                        event_type=EventType.GP_REMINDER,
                        patient_id=patient_id,
                        source_agent="heartbeat_scheduler",
                        payload={"channel": "websocket"},
                    )
                    try:
                        await self._processor(event)
                    except Exception as exc:
                        logger.warning(
                            "Failed to send GP reminder for %s: %s",
                            patient_id, exc,
                        )

    def _is_milestone_due(self, days_since: int, diary: Any) -> str:
        """Check if a milestone heartbeat should fire."""
        for milestone_day in MILESTONE_DAYS:
            if days_since >= milestone_day:
                # Check if this milestone was already fired
                milestone_key = f"heartbeat_{milestone_day}d"
                already_fired = any(
                    e.type == milestone_key
                    for e in diary.monitoring.entries
                )
                if not already_fired:
                    return milestone_key
        return ""

    @staticmethod
    def _days_since_booking(appointment_date: str) -> int:
        """Calculate days since the appointment was booked."""
        try:
            booked = datetime.strptime(appointment_date, "%Y-%m-%d")
            booked = booked.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - booked
            return delta.days
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _hours_since(dt: datetime | str) -> float:
        """Calculate hours since a datetime."""
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except (ValueError, TypeError):
                return 0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return delta.total_seconds() / 3600

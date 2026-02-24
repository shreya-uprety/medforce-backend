"""
Per-Patient Event Queue — serialises events for each patient.

One asyncio.Queue per active patient.  Events are processed FIFO,
one at a time.  Cross-patient queues run in parallel.

Idle queues are cleaned up after a configurable timeout.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from medforce.gateway.events import EventEnvelope

logger = logging.getLogger("gateway.queue")

# Type for the callback the queue calls to process each event
EventProcessor = Callable[[EventEnvelope], Awaitable[Any]]


class PatientQueueManager:
    """
    Manages one asyncio.Queue per patient_id.

    Usage:
        mgr = PatientQueueManager(processor=gateway.process_single_event)
        await mgr.enqueue(event)   # puts event in the right patient queue

    The manager automatically spawns a worker task for each patient on
    first event, and tears it down after idle_timeout_seconds of inactivity.
    """

    def __init__(
        self,
        processor: EventProcessor,
        idle_timeout_seconds: int = 1800,  # 30 minutes
        event_timeout_seconds: int = 60,   # max time for a single event
    ) -> None:
        self._processor = processor
        self._idle_timeout = idle_timeout_seconds
        self._event_timeout = event_timeout_seconds

        self._queues: dict[str, asyncio.Queue[EventEnvelope]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._last_activity: dict[str, datetime] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    # ── Public API ──

    async def start(self) -> None:
        """Start the cleanup background loop."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("PatientQueueManager started (idle timeout=%ds)", self._idle_timeout)

    async def stop(self) -> None:
        """Gracefully stop all workers and cleanup."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for pid in list(self._workers.keys()):
            await self._destroy_queue(pid)

        logger.info("PatientQueueManager stopped")

    async def enqueue(self, event: EventEnvelope) -> None:
        """Add an event to the patient's queue.  Creates queue if needed."""
        pid = event.patient_id

        if pid not in self._queues:
            self._create_queue(pid)

        self._last_activity[pid] = datetime.now(timezone.utc)
        await self._queues[pid].put(event)
        logger.debug("Enqueued %s for patient %s (depth=%d)",
                      event.event_type.value, pid, self._queues[pid].qsize())

    @property
    def active_patients(self) -> list[str]:
        """List of patient IDs with active queues."""
        return list(self._queues.keys())

    @property
    def active_count(self) -> int:
        return len(self._queues)

    def queue_depth(self, patient_id: str) -> int:
        """Number of pending events for a patient.  Returns 0 if no queue."""
        q = self._queues.get(patient_id)
        return q.qsize() if q else 0

    # ── Internal ──

    def _create_queue(self, patient_id: str) -> None:
        q: asyncio.Queue[EventEnvelope] = asyncio.Queue()
        self._queues[patient_id] = q
        self._last_activity[patient_id] = datetime.now(timezone.utc)
        worker = asyncio.create_task(self._worker_loop(patient_id))
        self._workers[patient_id] = worker
        logger.debug("Created queue + worker for patient %s", patient_id)

    async def _worker_loop(self, patient_id: str) -> None:
        """Process events for a single patient, one at a time."""
        q = self._queues.get(patient_id)
        if q is None:
            return

        while self._running or not q.empty():
            try:
                event = await asyncio.wait_for(q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue
            except asyncio.CancelledError:
                break

            import time as _time
            try:
                self._last_activity[patient_id] = datetime.now(timezone.utc)
                logger.info(
                    "Processing %s for patient %s (queue depth=%d)",
                    event.event_type.value, patient_id, q.qsize(),
                )
                t0 = _time.monotonic()
                # Do NOT use asyncio.wait_for — cancelling a to_thread
                # coroutine leaves zombie threads that hold GCS connections,
                # causing cascading timeouts. Let events run to completion;
                # individual GCS calls have their own HTTP timeouts.
                await self._processor(event)
                elapsed = _time.monotonic() - t0
                logger.info(
                    "Event %s for %s processed in %.2fs",
                    event.event_type.value, patient_id, elapsed,
                )
                if elapsed > 30:
                    logger.warning(
                        "Slow event: %s for %s took %.1fs",
                        event.event_type.value, patient_id, elapsed,
                    )
            except Exception as exc:
                logger.error(
                    "Error processing %s for patient %s: %s",
                    event.event_type.value, patient_id, exc,
                    exc_info=True,
                )
            finally:
                q.task_done()

    async def _destroy_queue(self, patient_id: str) -> None:
        worker = self._workers.pop(patient_id, None)
        if worker and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._queues.pop(patient_id, None)
        self._last_activity.pop(patient_id, None)
        logger.debug("Destroyed queue for patient %s", patient_id)

    async def _cleanup_loop(self) -> None:
        """Periodically destroy idle queues."""
        while self._running:
            try:
                await asyncio.sleep(60)  # check every minute
                now = datetime.now(timezone.utc)
                idle_patients = []
                for pid, last in list(self._last_activity.items()):
                    elapsed = (now - last).total_seconds()
                    q = self._queues.get(pid)
                    if elapsed > self._idle_timeout and (q is None or q.empty()):
                        idle_patients.append(pid)

                for pid in idle_patients:
                    logger.info("Cleaning up idle queue for patient %s", pid)
                    await self._destroy_queue(pid)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Queue cleanup error: %s", exc)

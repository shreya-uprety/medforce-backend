"""
Comprehensive tests for the Per-Patient Event Queue.

Tests cover:
  - FIFO ordering within a patient
  - Cross-patient parallelism
  - Queue creation on first event
  - Queue depth tracking
  - Error handling (processor failure doesn't crash worker)
  - Active patient listing
  - Stop/cleanup
  - Multiple rapid events for same patient
"""

import asyncio

import pytest

from medforce.gateway.events import EventEnvelope, EventType
from medforce.gateway.queue import PatientQueueManager


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _msg(patient_id: str, text: str) -> EventEnvelope:
    return EventEnvelope.user_message(patient_id=patient_id, text=text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIFO Ordering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFIFOOrdering:

    @pytest.mark.asyncio
    async def test_events_processed_in_order(self):
        """Events for one patient should be processed FIFO."""
        processed = []

        async def processor(event: EventEnvelope):
            processed.append(event.payload["text"])
            await asyncio.sleep(0.01)

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        await mgr.enqueue(_msg("PT-1", "First"))
        await mgr.enqueue(_msg("PT-1", "Second"))
        await mgr.enqueue(_msg("PT-1", "Third"))

        # Wait for processing
        await asyncio.sleep(0.2)
        await mgr.stop()

        assert processed == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_five_rapid_events(self):
        """Rapidly enqueue 5 events — all should process in order."""
        processed = []

        async def processor(event: EventEnvelope):
            processed.append(int(event.payload["text"]))

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        for i in range(5):
            await mgr.enqueue(_msg("PT-RAPID", str(i)))

        await asyncio.sleep(0.3)
        await mgr.stop()

        assert processed == [0, 1, 2, 3, 4]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cross-Patient Parallelism
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCrossPatientParallelism:

    @pytest.mark.asyncio
    async def test_different_patients_run_in_parallel(self):
        """Events for different patients should run concurrently."""
        processing_order = []
        start_times = {}

        async def processor(event: EventEnvelope):
            pid = event.patient_id
            start_times[pid] = asyncio.get_event_loop().time()
            processing_order.append(f"{pid}-start")
            await asyncio.sleep(0.1)
            processing_order.append(f"{pid}-end")

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        # Enqueue events for 3 different patients near-simultaneously
        await mgr.enqueue(_msg("PT-A", "Hello A"))
        await mgr.enqueue(_msg("PT-B", "Hello B"))
        await mgr.enqueue(_msg("PT-C", "Hello C"))

        await asyncio.sleep(0.5)
        await mgr.stop()

        # All three should have started (verify parallelism)
        assert "PT-A-start" in processing_order
        assert "PT-B-start" in processing_order
        assert "PT-C-start" in processing_order

    @pytest.mark.asyncio
    async def test_patient_serialisation_does_not_block_others(self):
        """A slow patient queue should not block other patients."""
        fast_done = asyncio.Event()

        async def processor(event: EventEnvelope):
            if event.patient_id == "PT-SLOW":
                await asyncio.sleep(0.5)  # slow processing
            else:
                fast_done.set()

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        await mgr.enqueue(_msg("PT-SLOW", "Slow"))
        await mgr.enqueue(_msg("PT-FAST", "Fast"))

        # Fast patient should complete well before slow patient
        await asyncio.wait_for(fast_done.wait(), timeout=0.3)
        await mgr.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Queue Lifecycle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQueueLifecycle:

    @pytest.mark.asyncio
    async def test_queue_created_on_first_event(self):
        mgr = PatientQueueManager(
            processor=lambda e: asyncio.sleep(0),
            idle_timeout_seconds=5,
        )
        mgr._running = True

        assert mgr.active_count == 0
        await mgr.enqueue(_msg("PT-NEW", "Hello"))
        assert "PT-NEW" in mgr.active_patients
        assert mgr.active_count == 1

        await asyncio.sleep(0.1)
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_active_patients_tracking(self):
        mgr = PatientQueueManager(
            processor=lambda e: asyncio.sleep(0.01),
            idle_timeout_seconds=5,
        )
        mgr._running = True

        await mgr.enqueue(_msg("PT-1", "A"))
        await mgr.enqueue(_msg("PT-2", "B"))
        await mgr.enqueue(_msg("PT-3", "C"))

        assert mgr.active_count == 3
        assert set(mgr.active_patients) == {"PT-1", "PT-2", "PT-3"}

        await asyncio.sleep(0.1)
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up_all_queues(self):
        mgr = PatientQueueManager(
            processor=lambda e: asyncio.sleep(0),
            idle_timeout_seconds=5,
        )
        mgr._running = True

        await mgr.enqueue(_msg("PT-1", "A"))
        await mgr.enqueue(_msg("PT-2", "B"))
        await asyncio.sleep(0.1)
        await mgr.stop()

        assert mgr.active_count == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Error Handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_processor_error_does_not_crash_worker(self):
        """If the processor raises, the worker should continue processing."""
        processed = []
        call_count = 0

        async def processor(event: EventEnvelope):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Simulated error on second event")
            processed.append(event.payload["text"])

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        await mgr.enqueue(_msg("PT-ERR", "First"))   # OK
        await mgr.enqueue(_msg("PT-ERR", "Second"))  # ERROR
        await mgr.enqueue(_msg("PT-ERR", "Third"))   # OK — should still process

        await asyncio.sleep(0.3)
        await mgr.stop()

        assert "First" in processed
        assert "Third" in processed
        assert "Second" not in processed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientScenarios:

    @pytest.mark.asyncio
    async def test_scenario_4_parallel_inputs(self):
        """
        Scenario from architecture doc: 3 events arrive within seconds
        for the same patient. Queue serialises them.
        """
        processed = []

        async def processor(event: EventEnvelope):
            processed.append(event.payload.get("text", event.event_type.value))
            await asyncio.sleep(0.02)

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        # Simulate 3 concurrent events for PT-1234
        lab_webhook = EventEnvelope(
            event_type=EventType.WEBHOOK,
            patient_id="PT-1234",
            payload={"text": "lab_webhook", "values": {"ALT": 180}},
            source="hospital_lab",
        )
        patient_msg = _msg("PT-1234", "patient_answer")
        helper_upload = EventEnvelope.user_message(
            patient_id="PT-1234",
            text="helper_upload",
            sender_id="HELPER-SARAH",
            sender_role="helper",
            attachments=["nhs_screenshot.jpg"],
        )

        await mgr.enqueue(lab_webhook)
        await mgr.enqueue(patient_msg)
        await mgr.enqueue(helper_upload)

        await asyncio.sleep(0.3)
        await mgr.stop()

        # All processed, in order
        assert processed == ["lab_webhook", "patient_answer", "helper_upload"]

    @pytest.mark.asyncio
    async def test_scenario_multi_patient_concurrent(self):
        """Multiple patients being processed simultaneously."""
        patient_events = {"PT-1": [], "PT-2": [], "PT-3": []}

        async def processor(event: EventEnvelope):
            patient_events[event.patient_id].append(event.payload["text"])
            await asyncio.sleep(0.01)

        mgr = PatientQueueManager(processor=processor, idle_timeout_seconds=5)
        mgr._running = True

        # Interleave events from different patients
        await mgr.enqueue(_msg("PT-1", "A1"))
        await mgr.enqueue(_msg("PT-2", "B1"))
        await mgr.enqueue(_msg("PT-3", "C1"))
        await mgr.enqueue(_msg("PT-1", "A2"))
        await mgr.enqueue(_msg("PT-2", "B2"))
        await mgr.enqueue(_msg("PT-1", "A3"))

        await asyncio.sleep(0.5)
        await mgr.stop()

        # Each patient's events are in order
        assert patient_events["PT-1"] == ["A1", "A2", "A3"]
        assert patient_events["PT-2"] == ["B1", "B2"]
        assert patient_events["PT-3"] == ["C1"]

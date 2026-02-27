"""
Gateway — Central deterministic event router.

No AI, no LLM — pure dict lookups and if/else.  The Gateway:
  1. Receives an EventEnvelope
  2. Loads the patient diary
  3. Routes to the correct agent (Strategy A or B)
  4. Saves the updated diary
  5. Dispatches responses via DispatcherRegistry
  6. Loops back any emitted events (recursive, with circuit breaker)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import (
    AgentResponse,
    DispatcherRegistry,
)
from medforce.gateway.diary import (
    ConversationEntry,
    CrossPhaseState,
    DiaryConcurrencyError,
    DiaryNotFoundError,
    DiaryStore,
    PatientDiary,
    Phase,
)
from medforce.gateway.events import (
    EventEnvelope,
    EventType,
    SenderRole,
)
from medforce.gateway.permissions import PermissionChecker, PermissionResult

logger = logging.getLogger("gateway.core")

# Maximum chained events from a single trigger (circuit breaker)
MAX_CHAIN_DEPTH = 10

# P0: Per-patient rate limiting
RATE_LIMIT_WINDOW_SECONDS = 60  # sliding window
RATE_LIMIT_MAX_MESSAGES = 15  # max USER_MESSAGE events per window

# P2: Input size limits
MAX_MESSAGE_LENGTH = 10_000  # characters — truncate beyond this


class Gateway:
    """
    Central event router. Deliberately dumb — deterministic lookup only.

    Strategy A (explicit routing):
      Handoff events have a hardcoded target agent in EXPLICIT_ROUTES.

    Strategy B (phase-based routing):
      External events (user messages, documents, webhooks) are routed
      to whichever agent owns the patient's current diary phase.
    """

    # Cross-phase keyword lists for content routing
    CLINICAL_KEYWORDS = [
        "allerg", "medication", "medicine", "taking", "prescribed",
        "symptom", "pain", "hurts", "bleeding", "dizzy", "nausea",
        "vomit", "fever", "swelling", "rash", "breathing",
        "diagnosed", "condition", "surgery", "operation",
        "side effect", "reaction", "intolerant",
    ]

    INTAKE_KEYWORDS = [
        "next of kin", "next-of-kin", "emergency contact",
        "my address", "moved to", "new phone", "new email",
        "my gp", "gp is", "changed my name", "nhs number",
    ]

    # Strategy A: event_type → agent name
    EXPLICIT_ROUTES: dict[EventType, str] = {
        EventType.INTAKE_COMPLETE: "clinical",
        EventType.INTAKE_DATA_PROVIDED: "clinical",
        EventType.CLINICAL_COMPLETE: "booking",
        EventType.BOOKING_COMPLETE: "monitoring",
        EventType.NEEDS_INTAKE_DATA: "intake",
        EventType.HEARTBEAT: "monitoring",
        EventType.DETERIORATION_ALERT: "clinical",
        EventType.RESCHEDULE_REQUEST: "booking",
        EventType.GP_QUERY: "gp_comms",
        EventType.GP_RESPONSE: "clinical",
        EventType.GP_REMINDER: "gp_comms",
        EventType.HELPER_REGISTRATION: "helper_manager",
        EventType.HELPER_VERIFIED: "helper_manager",
        EventType.AGENT_ERROR: "error_handler",
        EventType.INTAKE_FORM_SUBMITTED: "intake",
    }

    # Strategy B: phase → agent name
    PHASE_ROUTES: dict[str, str | None] = {
        Phase.INTAKE.value: "intake",
        Phase.CLINICAL.value: "clinical",
        Phase.BOOKING.value: "booking",
        Phase.MONITORING.value: "monitoring",
        Phase.CLOSED.value: None,  # closed patients — log only
    }

    def __init__(
        self,
        *,
        diary_store: DiaryStore,
        dispatcher_registry: DispatcherRegistry,
        permission_checker: PermissionChecker | None = None,
    ) -> None:
        self._diary_store = diary_store
        self._dispatchers = dispatcher_registry
        self._permissions = permission_checker or PermissionChecker()
        self._agents: dict[str, BaseAgent] = {}
        self._event_log: list[dict[str, Any]] = []  # in-memory event log for debugging
        self._processed_events: dict[str, OrderedDict[str, bool]] = {}  # patient_id → {event_id: True}
        # Per-patient diary cache — safe because events per patient are sequential
        self._diary_cache: dict[str, tuple[PatientDiary, int]] = {}  # patient_id → (diary, generation)
        # Background tasks (fire-and-forget chat persistence etc.)
        self._bg_tasks: set[asyncio.Task] = set()
        # P0: Per-patient rate limiting — patient_id → list of timestamps
        self._rate_limiter: dict[str, list[float]] = {}
        # P2: Dead Letter Queue — failed events stored for ops review
        self._dead_letter_queue: list[dict[str, Any]] = []
        # P2: Observability metrics
        self._metrics: dict[str, Any] = {
            "events_processed": 0,
            "events_failed": 0,
            "events_rate_limited": 0,
            "agent_processing_times": {},   # agent_name → list[float]
            "patients_per_phase": {},       # phase → count (snapshot)
            "diary_save_failures": 0,
        }

    # ── Agent Registration ──

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist agent by name."""
        self._agents[name] = agent
        agent.agent_name = name
        logger.info("Registered agent: %s", name)

    def get_agent(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    @property
    def registered_agents(self) -> list[str]:
        return list(self._agents.keys())

    # ── Main Entry Point ──

    async def process_event(self, event: EventEnvelope) -> AgentResult | None:
        """
        Process a single event through the Gateway.

        This is the main entry point. It:
          1. Loads or creates the patient diary
          2. Checks permissions
          3. Routes to the correct agent
          4. Saves the updated diary
          5. Dispatches responses
          6. Loops back emitted events (recursive)

        Returns the AgentResult from the primary agent, or None if the
        event was rejected or no agent handled it.
        """
        chain_depth = getattr(event, "_chain_depth", 0)
        logger.info(
            "process_event entered: %s for patient %s (chain_depth=%d)",
            event.event_type.value, event.patient_id, chain_depth,
        )

        # Idempotency guard — skip duplicate events per patient
        patient_seen = self._processed_events.setdefault(
            event.patient_id, OrderedDict()
        )
        if event.event_id in patient_seen:
            logger.info(
                "Duplicate event %s for patient %s — skipping",
                event.event_id, event.patient_id,
            )
            self._log_event(event, "DUPLICATE", None)
            return None
        patient_seen[event.event_id] = True
        # Cap at 100 per patient with FIFO eviction
        while len(patient_seen) > 100:
            patient_seen.popitem(last=False)

        # P0: Rate limiting for user messages (skip internal/agent events)
        if (
            chain_depth == 0
            and event.event_type == EventType.USER_MESSAGE
            and self._is_rate_limited(event.patient_id)
        ):
            logger.warning(
                "Rate limit exceeded for patient %s — dropping USER_MESSAGE",
                event.patient_id,
            )
            self._log_event(event, "RATE_LIMITED", None)
            self._metrics["events_rate_limited"] += 1
            rate_response = AgentResponse(
                recipient=event.sender_id or "patient",
                channel=event.payload.get("channel", "websocket"),
                message=(
                    "You're sending messages quite quickly. Please wait a moment "
                    "before sending another message — we want to make sure each "
                    "one is properly processed."
                ),
                metadata={"patient_id": event.patient_id, "rate_limited": True},
            )
            # Dispatch the rate-limit response so the patient actually sees it
            await self._dispatchers.dispatch_all([rate_response])
            return AgentResult(
                updated_diary=PatientDiary.create_new(event.patient_id),
                responses=[rate_response],
            )

        if chain_depth >= MAX_CHAIN_DEPTH:
            logger.error(
                "Circuit breaker: max chain depth %d reached for patient %s "
                "(event %s). Dropping event.",
                MAX_CHAIN_DEPTH,
                event.patient_id,
                event.event_type.value,
            )
            self._log_event(event, "CIRCUIT_BREAKER", None)
            return None

        # 1. Load or create diary
        t0 = time.monotonic()
        diary, generation = await self._load_or_create_diary(event)
        logger.info("  [timing] diary load: %.2fs", time.monotonic() - t0)

        # 1b. Cross-phase timeout safety — auto-clear stale cross-phase state
        if diary.cross_phase_state.active and diary.cross_phase_state.started:
            elapsed = (datetime.now(timezone.utc) - diary.cross_phase_state.started).total_seconds()
            if elapsed > 600:  # 10 minutes
                logger.info(
                    "Cross-phase state timed out for patient %s (%.0fs) — clearing",
                    event.patient_id, elapsed,
                )
                diary.cross_phase_state = CrossPhaseState()

        # 2. Check permissions
        perm_result = self._check_permissions(event, diary)
        if not perm_result.allowed:
            logger.warning(
                "Permission denied for %s (%s) on patient %s: %s",
                event.sender_id,
                event.sender_role.value if isinstance(event.sender_role, SenderRole) else event.sender_role,
                event.patient_id,
                perm_result.reason,
            )
            self._log_event(event, "PERMISSION_DENIED", perm_result.reason)

            # Return a rejection response
            rejection = AgentResponse(
                recipient=event.sender_id,
                channel=event.payload.get("channel", "websocket"),
                message=(
                    "Sorry, you don't have permission to perform this action. "
                    "Please ask the patient to grant you the required access."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(
                updated_diary=diary,
                responses=[rejection],
            )

        # 3. Pre-detect cross-phase content BEFORE primary agent
        cross_phase_detected = False
        xphase_targets: list[str] = []
        xphase_from_phase = diary.header.current_phase.value  # capture BEFORE agent may change it
        if chain_depth == 0 and event.event_type == EventType.USER_MESSAGE:
            text_for_xphase = event.payload.get("text", "")
            current_phase_val = diary.header.current_phase.value
            xphase_targets = self._detect_cross_phase_targets(text_for_xphase, current_phase_val)
            if xphase_targets:
                cross_phase_detected = True
                event.payload["_has_cross_phase_content"] = True
                event.payload["_cross_phase_targets"] = xphase_targets

        # 3b. Cross-phase state redirect — when awaiting a follow-up response,
        #     route the USER_MESSAGE directly to the cross-phase agent
        if (
            event.event_type == EventType.USER_MESSAGE
            and diary.cross_phase_state.active
            and diary.cross_phase_state.awaiting_response
        ):
            target_agent_name = diary.cross_phase_state.target_agent
            event.payload["_cross_phase_followup"] = True
            event.payload["_pending_phase"] = diary.cross_phase_state.pending_phase
            # Don't also emit cross-phase events for the follow-up response
            cross_phase_detected = False
            xphase_targets = []
        else:
            target_agent_name = self._resolve_target(event, diary)

        if target_agent_name is None:
            logger.info(
                "No agent target for event %s (patient %s, phase %s) — logged only",
                event.event_type.value,
                event.patient_id,
                diary.header.current_phase.value,
            )
            self._log_event(event, "NO_TARGET", None)
            return AgentResult(updated_diary=diary)

        agent = self._agents.get(target_agent_name)
        if agent is None:
            logger.warning(
                "Agent '%s' not registered — cannot process %s for patient %s",
                target_agent_name,
                event.event_type.value,
                event.patient_id,
            )
            self._log_event(event, "AGENT_NOT_FOUND", target_agent_name)
            return AgentResult(updated_diary=diary)

        # 4. P4: Truncate oversized user messages to prevent abuse
        if event.event_type == EventType.USER_MESSAGE:
            text = event.payload.get("text", "")
            if len(text) > MAX_MESSAGE_LENGTH:
                logger.warning(
                    "Truncating oversized message for patient %s (%d chars → %d)",
                    event.patient_id, len(text), MAX_MESSAGE_LENGTH,
                )
                event.payload["text"] = text[:MAX_MESSAGE_LENGTH]

        # 4b. Log the inbound conversation entry
        # Determine the chat channel for this event: explicit override > phase-based
        source_chat_channel = event.payload.get("_source_chat_channel")
        if event.event_type == EventType.USER_MESSAGE:
            role = event.sender_role.value if isinstance(event.sender_role, SenderRole) else event.sender_role
            inbound_chat_channel = (
                source_chat_channel
                or ("monitoring"
                    if diary.header.current_phase == Phase.MONITORING
                    else "pre_consultation")
            )
            diary.add_conversation(
                ConversationEntry(
                    direction=f"{role.upper()}→AGENT",
                    channel=event.payload.get("channel", ""),
                    message=event.payload.get("text", ""),
                    chat_channel=inbound_chat_channel,
                )
            )

        # 5. Process the event
        logger.info(
            "Routing %s → %s (patient %s, phase %s)",
            event.event_type.value,
            target_agent_name,
            event.patient_id,
            diary.header.current_phase.value,
        )
        self._log_event(event, "ROUTED", target_agent_name)

        # Capture phase before processing for P0 phase-transition tracking
        phase_before = diary.header.current_phase

        try:
            t1 = time.monotonic()
            result = await agent.process(event, diary)
            elapsed = time.monotonic() - t1
            logger.info("  [timing] agent %s process: %.2fs", target_agent_name, elapsed)

            # P2: Track metrics
            self._metrics["events_processed"] += 1
            agent_times = self._metrics["agent_processing_times"].setdefault(target_agent_name, [])
            agent_times.append(elapsed)
            if len(agent_times) > 200:
                self._metrics["agent_processing_times"][target_agent_name] = agent_times[-100:]
        except Exception as exc:
            logger.error(
                "Agent '%s' error processing %s for patient %s: %s",
                target_agent_name,
                event.event_type.value,
                event.patient_id,
                exc,
                exc_info=True,
            )
            self._log_event(event, "AGENT_ERROR", str(exc))
            self._metrics["events_failed"] += 1

            # P2: Add to dead letter queue for ops replay
            self._add_to_dlq(event, target_agent_name, exc)

            error_response = AgentResponse(
                recipient=event.sender_id or "patient",
                channel=event.payload.get("channel", "websocket"),
                message=(
                    "We're sorry, we encountered a temporary issue processing "
                    "your request. Please try again in a moment. If the problem "
                    "persists, our team has been notified and will follow up."
                ),
                metadata={"patient_id": event.patient_id, "error": True},
            )
            return AgentResult(updated_diary=diary, responses=[error_response])

        # 5b. Cross-phase content routing — emit CROSS_PHASE_DATA events
        #     using the pre-detected targets (step 3) so the primary agent
        #     already saw the _has_cross_phase_content flag.
        #     Skip if the primary agent already responded (e.g. monitoring
        #     started a deterioration assessment — don't also fire cross-phase).
        primary_responded = bool(result.responses)
        if cross_phase_detected and xphase_targets and not primary_responded:
            text_for_xphase = event.payload.get("text", "")
            for tgt_agent in xphase_targets:
                xphase_event = EventEnvelope.handoff(
                    event_type=EventType.CROSS_PHASE_DATA,
                    patient_id=event.patient_id,
                    source_agent="gateway",
                    payload={
                        "_target_agent": tgt_agent,
                        "text": text_for_xphase,
                        "from_phase": xphase_from_phase,
                        "channel": event.payload.get("channel", "websocket"),
                    },
                    correlation_id=event.correlation_id,
                )
                result.emitted_events.append(xphase_event)
                logger.info(
                    "Cross-phase data detected: %s → %s for patient %s",
                    xphase_from_phase, tgt_agent, event.patient_id,
                )

        # 6. Stamp chat_channel on outbound responses and log as conversation entries
        #    If the event carries a _source_chat_channel (propagated from a parent
        #    event in the monitoring chat), honour it even when the phase has changed
        #    (e.g. booking agent sets phase=BOOKING during reschedule).
        outbound_chat_channel = (
            source_chat_channel
            or (
                "monitoring"
                if target_agent_name == "monitoring"
                or result.updated_diary.header.current_phase == Phase.MONITORING
                else "pre_consultation"
            )
        )
        for resp in result.responses:
            resp.metadata.setdefault("chat_channel", outbound_chat_channel)
            result.updated_diary.add_conversation(
                ConversationEntry(
                    direction=f"AGENT→{resp.recipient.upper()}",
                    channel=resp.channel,
                    message=resp.message[:200] if resp.message else "",
                    chat_channel=resp.metadata.get("chat_channel", outbound_chat_channel),
                )
            )

        # 6b. P0: Stamp phase_entered_at if the phase changed
        if result.updated_diary.header.current_phase != phase_before:
            result.updated_diary.header.phase_entered_at = datetime.now(timezone.utc)
            logger.info(
                "Phase transition: %s → %s for patient %s",
                phase_before.value,
                result.updated_diary.header.current_phase.value,
                event.patient_id,
            )

        # 6c. Eagerly update the diary cache BEFORE dispatching responses.
        #      This ensures that any API consumer polling the diary after
        #      receiving a response sees the latest agent-updated state,
        #      even while the GCS save (step 8) is still in flight.
        #      The generation is stale until step 8 updates it, but the
        #      per-patient queue guarantees no concurrent event uses it.
        self._diary_cache[event.patient_id] = (
            result.updated_diary.model_copy(deep=True),
            generation,
        )

        # 7. Dispatch responses IMMEDIATELY (before diary save) so patients
        #    don't wait for GCS round-trips.
        if result.responses:
            delivery_results = await self._dispatchers.dispatch_all(result.responses)
            for dr in delivery_results:
                if not dr.success:
                    logger.warning(
                        "Delivery failed for %s on %s: %s",
                        dr.recipient,
                        dr.channel,
                        dr.error,
                    )

        # 8. Cache diary immediately so reads return fresh data instantly,
        #    then persist to GCS in background (fire-and-forget).
        #    This eliminates the 30-90s GCS save blocking the response pipeline.
        self._diary_cache[event.patient_id] = (
            result.updated_diary.model_copy(deep=True),
            generation,  # will be updated by background save
        )

        async def _save_diary_bg(pid, diary_copy, gen):
            backoffs = [0.1, 0.3, 0.9]
            for attempt in range(len(backoffs) + 1):
                try:
                    t2 = time.monotonic()
                    new_gen = await asyncio.to_thread(
                        self._diary_store.save, pid, diary_copy, gen,
                    )
                    logger.info("  [timing] diary save: %.2fs", time.monotonic() - t2)
                    # Update ONLY the generation in the cache — the diary data
                    # in the cache may already be newer (updated by a subsequent
                    # event processed while this bg save was in flight).
                    # Overwriting with diary_copy would revert to stale state.
                    cached = self._diary_cache.get(pid)
                    if cached is not None:
                        self._diary_cache[pid] = (cached[0], new_gen)
                    else:
                        self._diary_cache[pid] = (diary_copy, new_gen)
                    return
                except DiaryConcurrencyError:
                    # Do NOT clear the diary cache — it has the latest agent-
                    # updated state which is MORE current than what's in GCS.
                    # Clearing it causes the next event to load stale state
                    # from GCS, leading to duplicate agent actions (e.g. booking
                    # agent re-offering slots after the user already selected).
                    if attempt < len(backoffs):
                        logger.warning(
                            "Diary concurrency conflict for patient %s (attempt %d) — retrying",
                            pid, attempt + 1,
                        )
                        try:
                            # Reload generation from GCS for the retry
                            _, gen = await asyncio.to_thread(
                                self._diary_store.load, pid
                            )
                        except Exception:
                            pass
                        await asyncio.sleep(backoffs[attempt])
                    else:
                        logger.error(
                            "Diary save failed after %d retries for %s (concurrency)",
                            len(backoffs), pid,
                        )
                        self._metrics["diary_save_failures"] += 1
                except Exception as exc:
                    if attempt < len(backoffs):
                        logger.warning(
                            "Diary save failed for %s (attempt %d): %s — retrying",
                            pid, attempt + 1, exc,
                        )
                        await asyncio.sleep(backoffs[attempt])
                    else:
                        logger.error(
                            "Failed to save diary for %s after %d retries: %s",
                            pid, len(backoffs) + 1, exc,
                        )
                        self._metrics["diary_save_failures"] += 1

        save_task = asyncio.create_task(
            _save_diary_bg(
                event.patient_id,
                result.updated_diary.model_copy(deep=True),
                generation,
            )
        )
        self._bg_tasks.add(save_task)
        save_task.add_done_callback(self._bg_tasks.discard)

        # 9. Persist chat history to patient_data in GCS
        #    Fire-and-forget so it doesn't block the patient queue.
        if event.event_type == EventType.USER_MESSAGE or result.responses:
            async def _persist_bg(pid, diary_copy):
                try:
                    await asyncio.to_thread(
                        self._persist_chat_history, pid, diary_copy
                    )
                except Exception as exc:
                    logger.warning(
                        "Chat persistence failed for patient %s: %s", pid, exc,
                    )
            task = asyncio.create_task(
                _persist_bg(event.patient_id, result.updated_diary)
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

        # 10. Loop back emitted events (recursive with increased chain depth)
        #     Propagate _source_chat_channel so child events stay in the same chat.
        #     Only propagate "monitoring" — pre_consultation is the default and
        #     should not override agent-based routing (e.g. BOOKING_COMPLETE → monitoring).
        for emitted in result.emitted_events:
            emitted._chain_depth = chain_depth + 1
            if (
                outbound_chat_channel == "monitoring"
                and "_source_chat_channel" not in emitted.payload
            ):
                emitted.payload["_source_chat_channel"] = "monitoring"
            logger.info(
                "Looping back emitted event %s for patient %s (depth=%d)",
                emitted.event_type.value,
                emitted.patient_id,
                emitted._chain_depth,
            )
            await self.process_event(emitted)

        return result

    # ── Routing Logic ──

    def _resolve_target(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> str | None:
        """Determine which agent should handle this event."""

        # Special handling: CROSS_PHASE_DATA routes to its target agent
        if event.event_type == EventType.CROSS_PHASE_DATA:
            return event.payload.get("_target_agent")

        # CROSS_PHASE_REPROMPT routes back to the pending phase's agent
        if event.event_type == EventType.CROSS_PHASE_REPROMPT:
            pending = event.payload.get("_pending_phase")
            return self.PHASE_ROUTES.get(pending)

        # Strategy A: explicit routing for handoff / internal events
        if event.is_explicit_route():
            return self.EXPLICIT_ROUTES.get(event.event_type)

        # Strategy B: phase-based routing for external events
        if event.is_phase_route():
            phase = diary.header.current_phase.value
            return self.PHASE_ROUTES.get(phase)

        # Fallback — shouldn't happen if EventType is well-defined
        logger.warning(
            "Event %s doesn't match any routing strategy",
            event.event_type.value,
        )
        return None

    def _detect_cross_phase_targets(
        self, text: str, current_phase: str
    ) -> list[str]:
        """
        Fast keyword matching to detect cross-phase content.
        Returns list of agent names that should ALSO receive this data.
        Excludes the current phase's agent (no self-routing).
        """
        text_lower = text.lower()
        targets = []

        # Check clinical keywords
        current_agent = self.PHASE_ROUTES.get(current_phase)
        if current_agent != "clinical":
            if any(kw in text_lower for kw in self.CLINICAL_KEYWORDS):
                targets.append("clinical")

        # Check intake keywords
        if current_agent != "intake":
            if any(kw in text_lower for kw in self.INTAKE_KEYWORDS):
                targets.append("intake")

        return targets

    # ── Patient Data Persistence ──

    def _persist_chat_history(
        self, patient_id: str, diary: PatientDiary
    ) -> None:
        """
        Persist the conversation log split by chat_channel:
          - patient_data/{patient_id}/pre_consultation_chat.json
          - patient_data/{patient_id}/monitoring_chat.json
        """
        import json as _json

        def _build_conversation(entries):
            conversation = []
            for entry in entries:
                sender = "admin"  # agent responses
                if "PATIENT" in entry.direction or "HELPER" in entry.direction:
                    if "→AGENT" in entry.direction:
                        sender = "patient"
                conversation.append({
                    "sender": sender,
                    "message": entry.message,
                    "channel": entry.channel,
                    "timestamp": entry.timestamp.isoformat(),
                })
            return conversation

        try:
            pre_consult_entries = diary.get_conversation("pre_consultation")
            monitoring_entries = diary.get_conversation("monitoring")

            # Always write pre-consultation chat
            pre_data = {"conversation": _build_conversation(pre_consult_entries)}
            self._diary_store._gcs.create_file_from_string(
                _json.dumps(pre_data, indent=2),
                f"patient_data/{patient_id}/pre_consultation_chat.json",
                content_type="application/json",
            )

            # Write monitoring chat if there are any monitoring entries
            if monitoring_entries:
                mon_data = {"conversation": _build_conversation(monitoring_entries)}
                self._diary_store._gcs.create_file_from_string(
                    _json.dumps(mon_data, indent=2),
                    f"patient_data/{patient_id}/monitoring_chat.json",
                    content_type="application/json",
                )
        except Exception as exc:
            logger.warning(
                "Failed to persist chat history for patient %s: %s",
                patient_id,
                exc,
            )

    # ── Helpers ──

    async def _load_or_create_diary(
        self, event: EventEnvelope
    ) -> tuple[PatientDiary, int]:
        """Load existing diary or create a new one for first contact."""
        # Check in-memory cache first (safe — per-patient queue is sequential)
        cached = self._diary_cache.get(event.patient_id)
        if cached is not None:
            diary, generation = cached
            logger.info("  diary cache hit for patient %s", event.patient_id)
            return diary.model_copy(deep=True), generation

        try:
            t0 = time.monotonic()
            diary, generation = await asyncio.to_thread(
                self._diary_store.load, event.patient_id
            )
            logger.info("  [timing] diary load: %.2fs", time.monotonic() - t0)
            # Cache for subsequent loads
            self._diary_cache[event.patient_id] = (
                diary.model_copy(deep=True), generation,
            )
            return diary, generation
        except DiaryNotFoundError:
            logger.info(
                "No diary found for patient %s — creating in-memory (GCS save deferred)",
                event.patient_id,
            )
            # Create diary in-memory immediately; GCS persistence happens
            # in the background save after the agent processes the event.
            diary = PatientDiary.create_new(
                event.patient_id, event.correlation_id
            )
            generation = None  # type: ignore[assignment]
            self._diary_cache[event.patient_id] = (
                diary.model_copy(deep=True), generation,
            )
            return diary, generation

    def _check_permissions(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> PermissionResult:
        """Look up sender permissions and check access."""
        sender_role = event.sender_role
        sender_permissions: list[str] = []

        role_val = sender_role.value if isinstance(sender_role, SenderRole) else sender_role

        if role_val == "patient":
            sender_permissions = ["full_access"]
        elif role_val == "helper":
            helper = diary.helper_registry.get_helper(event.sender_id)
            if helper and helper.verified:
                sender_permissions = list(helper.permissions)
            else:
                return PermissionResult(
                    allowed=False,
                    reason="helper_not_verified" if helper else "helper_not_found",
                )
        elif role_val == "gp":
            sender_permissions = [
                "view_status",
                "upload_documents",
                "respond_to_queries",
            ]

        return self._permissions.check(
            sender_role=sender_role,
            sender_permissions=sender_permissions,
            event=event,
            diary_phase=diary.header.current_phase.value,
        )

    def _log_event(
        self, event: EventEnvelope, status: str, detail: Any
    ) -> None:
        """Append to in-memory event log for debugging / test harness."""
        self._event_log.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "patient_id": event.patient_id,
                "sender_id": event.sender_id,
                "sender_role": (
                    event.sender_role.value
                    if isinstance(event.sender_role, SenderRole)
                    else event.sender_role
                ),
                "status": status,
                "detail": detail,
                "timestamp": event.timestamp.isoformat(),
            }
        )
        # Keep log bounded
        if len(self._event_log) > 1000:
            self._event_log = self._event_log[-500:]

    def get_event_log(
        self, patient_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Retrieve event log entries, optionally filtered by patient."""
        if patient_id:
            entries = [e for e in self._event_log if e["patient_id"] == patient_id]
        else:
            entries = list(self._event_log)
        return entries[-limit:]

    # ── P2: Dead Letter Queue ──

    def _add_to_dlq(
        self, event: EventEnvelope, agent_name: str, error: Exception
    ) -> None:
        """Add a failed event to the dead letter queue for ops review."""
        import traceback

        entry = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "patient_id": event.patient_id,
            "agent": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "payload": event.payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._dead_letter_queue.append(entry)
        # Cap at 500 entries
        if len(self._dead_letter_queue) > 500:
            self._dead_letter_queue = self._dead_letter_queue[-250:]
        logger.info("Event %s added to DLQ (agent=%s, error=%s)", event.event_id, agent_name, type(error).__name__)

    def get_dlq(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve dead letter queue entries for ops review."""
        return self._dead_letter_queue[-limit:]

    def replay_dlq_event(self, index: int) -> dict[str, Any] | None:
        """Get a DLQ entry by index for manual replay."""
        if 0 <= index < len(self._dead_letter_queue):
            return self._dead_letter_queue[index]
        return None

    # ── P2: Observability & Metrics ──

    def get_metrics(self) -> dict[str, Any]:
        """Return current gateway metrics for observability."""
        metrics = dict(self._metrics)
        # Compute agent processing time summaries
        summaries = {}
        for agent_name, times in metrics.get("agent_processing_times", {}).items():
            if times:
                summaries[agent_name] = {
                    "count": len(times),
                    "avg_ms": round(sum(times) / len(times) * 1000, 1),
                    "max_ms": round(max(times) * 1000, 1),
                    "min_ms": round(min(times) * 1000, 1),
                }
        metrics["agent_processing_summaries"] = summaries
        metrics["dlq_size"] = len(self._dead_letter_queue)
        return metrics

    def health_check(self) -> dict[str, Any]:
        """P2: Health check — verify agents, diary store, and channels."""
        checks: dict[str, Any] = {
            "agents_registered": len(self._agents) > 0,
            "agent_names": list(self._agents.keys()),
            "channels_registered": len(self._dispatchers._dispatchers) > 0,
            "channel_names": self._dispatchers.registered_channels,
            "diary_store_available": self._diary_store is not None,
            "events_processed": self._metrics["events_processed"],
            "events_failed": self._metrics["events_failed"],
            "dlq_size": len(self._dead_letter_queue),
        }
        # Overall status
        checks["healthy"] = (
            checks["agents_registered"]
            and checks["diary_store_available"]
        )
        return checks

    # ── P0: Rate Limiting ──

    def _is_rate_limited(self, patient_id: str) -> bool:
        """
        Sliding window rate limiter. Returns True if the patient has
        exceeded RATE_LIMIT_MAX_MESSAGES in the last RATE_LIMIT_WINDOW_SECONDS.
        """
        now = time.monotonic()
        timestamps = self._rate_limiter.get(patient_id, [])

        # Evict expired entries
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= RATE_LIMIT_MAX_MESSAGES:
            self._rate_limiter[patient_id] = timestamps
            return True

        timestamps.append(now)
        self._rate_limiter[patient_id] = timestamps
        return False

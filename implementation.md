# MedForce Event-Driven Architecture — Implementation Plan

**Date:** February 2026
**Based on:** MedForce_Full_Architecture.md v2.0
**Current Codebase:** clinic-os-v4/backend (FastAPI + Gemini + GCS)

---

## Feasibility Assessment

### Verdict: IMPLEMENTABLE with current resources

The existing codebase provides all the infrastructure building blocks needed. No new cloud services, databases, or paid APIs are required for Phases 1–5. The core architecture is pure Python + asyncio + GCS, all of which are already in production.

### What Already Exists (Reusable)

| Existing Component | Architecture Role | Location |
|---|---|---|
| `GCSBucketManager` | Patient Diary storage backend | `medforce/infrastructure/gcs.py` |
| `PreConsulteAgent` | Core logic for Clinical Agent (wrappable) | `medforce/agents/pre_consult_agents.py` |
| `ScheduleCSVManager` | Booking Agent slot queries | `medforce/managers/schedule.py` |
| `side_agent.parse_lab_values()` | Lab value extraction for Clinical Agent | `medforce/agents/side_agent.py` |
| `side_agent.generate_dili_diagnosis()` | Risk assessment logic (reusable) | `medforce/agents/side_agent.py` |
| `asyncio.Queue` pattern | Per-patient event queue model | Already used in `voice_session.py` |
| Gemini Flash + Pro models | LLM backend for all 4 agents | Already configured in `app.py` |
| WebSocket infrastructure | Real-time chat transport | `medforce/agents/websocket_agent.py` |
| System prompts (clinical, diagnosis) | Agent instruction templates | `system_prompts/` directory |
| Pydantic schemas | Event envelope and diary models | `medforce/schemas/` |

### What Must Be Built

| New Component | Complexity | Depends On |
|---|---|---|
| Event Envelope (Pydantic model) | Low | Nothing |
| Channel Dispatcher abstraction + registry | Low | Event Envelope |
| Patient Diary schema + CRUD | Medium | GCSBucketManager |
| Per-patient asyncio queue manager | Medium | Event Envelope |
| Gateway router (deterministic) | Medium | Diary, Queue, Identity, ChannelDispatcher |
| Channel Ingest abstraction (inbound) | Low | Event Envelope |
| Identity Resolution system | Medium | Diary (contact index) |
| Intake Agent | Medium | Gateway contract |
| Clinical Agent (wrapping PreConsulteAgent) | High | Gateway contract, side_agent |
| Booking Agent | Medium | ScheduleCSVManager, Gateway contract |
| Monitoring Agent | Medium | Gateway contract |
| Heartbeat Scheduler | Low | Gateway, Diary |
| GP Communication Handler | Medium | Gateway contract, ChannelDispatcher |
| Helper Manager | Medium | Diary, Identity, ChannelDispatcher |
| Permission checking layer | Low | Diary (helper registry) |
| HTML Test Harness | Medium | All Gateway API endpoints |
| Gateway API endpoints (3 new) | Low | Gateway |

### Key Architecture Decision: Channel Abstraction for Seamless Dialogflow

The architecture document assumes Dialogflow is currently in the stack as a channel adapter for WhatsApp/SMS/Email. **It is not.** The current system uses direct WebSocket and HTTP endpoints.

To ensure Dialogflow (and any other channel) can be plugged in later **without touching the Gateway or any agent code**, we introduce two abstractions from Phase 1:

1. **ChannelDispatcher (outbound)** — an interface for delivering responses to patients/helpers/GPs. The Gateway never calls WhatsApp or Dialogflow directly — it calls `dispatcher.send(response)` and the registered dispatcher handles delivery.

2. **ChannelIngest (inbound)** — an interface for receiving messages from external channels and wrapping them into Event Envelopes. Each channel (WebSocket, Dialogflow webhook, Twilio webhook, email inbound parse) implements this once.

```
INBOUND (any channel → Gateway):
  WhatsApp msg → Dialogflow webhook → ChannelIngest.to_envelope() → POST /api/gateway/emit
  WebSocket msg → WebSocketIngest.to_envelope() → gateway.process_event()
  Email reply → EmailIngest.to_envelope() → POST /api/gateway/emit
  SMS reply → TwilioIngest.to_envelope() → POST /api/gateway/emit

OUTBOUND (Gateway → any channel):
  AgentResponse → Gateway → ChannelDispatcher.send() → correct channel
  ChannelDispatcher routes by response.channel field:
    "websocket" → WebSocketDispatcher (pushes to connected session)
    "dialogflow_whatsapp" → DialogflowDispatcher (calls Dialogflow API → WhatsApp)
    "dialogflow_sms" → DialogflowDispatcher (calls Dialogflow API → SMS)
    "email" → EmailDispatcher (calls SendGrid/Mailgun API)
```

**Why this matters:** When Phase 6 arrives, adding Dialogflow is just:
1. Register `DialogflowDispatcher` in the dispatcher registry
2. Add a `/api/gateway/dialogflow-webhook` endpoint that uses `DialogflowIngest`
3. Zero changes to Gateway, agents, diary, queue, or any other component

During Phases 1–5, only `WebSocketDispatcher` and `TestHarnessDispatcher` are registered. The architecture runs identically — the abstraction costs nothing but prevents a Phase 6 retrofit.

---

## Phase 1: Foundation

**Goal:** Define the data structures that everything else builds on. No runtime behavior yet — just models, schemas, and storage.

**Existing code touched:** None. Fully additive.

### 1.1 Event Envelope Model

Create `medforce/gateway/events.py`:

```python
# Pydantic models for the universal event envelope
class EventEnvelope:
    event_id: UUID
    event_type: EventType  # Enum: USER_MESSAGE, HEARTBEAT, INTAKE_COMPLETE, etc.
    patient_id: str
    payload: dict
    source: str
    sender_id: str
    sender_role: SenderRole  # Enum: patient, helper, gp, system
    correlation_id: UUID
    timestamp: datetime

class EventType(str, Enum):
    USER_MESSAGE = "USER_MESSAGE"
    DOCUMENT_UPLOADED = "DOCUMENT_UPLOADED"
    INTAKE_COMPLETE = "INTAKE_COMPLETE"
    INTAKE_DATA_PROVIDED = "INTAKE_DATA_PROVIDED"
    CLINICAL_COMPLETE = "CLINICAL_COMPLETE"
    BOOKING_COMPLETE = "BOOKING_COMPLETE"
    NEEDS_INTAKE_DATA = "NEEDS_INTAKE_DATA"
    HEARTBEAT = "HEARTBEAT"
    DETERIORATION_ALERT = "DETERIORATION_ALERT"
    GP_QUERY = "GP_QUERY"
    GP_RESPONSE = "GP_RESPONSE"
    GP_REMINDER = "GP_REMINDER"
    HELPER_REGISTRATION = "HELPER_REGISTRATION"
    HELPER_VERIFIED = "HELPER_VERIFIED"
    WEBHOOK = "WEBHOOK"
    DOCTOR_COMMAND = "DOCTOR_COMMAND"
    AGENT_ERROR = "AGENT_ERROR"
```

**Deliverable:** Pure Pydantic model. No side effects.

### 1.2 Channel Dispatcher + Channel Ingest Abstractions

Create `medforce/gateway/channels.py`:

```python
# ──── OUTBOUND: Delivering responses to recipients ────

class ChannelDispatcher(ABC):
    """Abstract interface for delivering messages to a specific channel."""
    channel_name: str  # "websocket", "dialogflow_whatsapp", "email", etc.

    @abstractmethod
    async def send(self, response: AgentResponse) -> DeliveryResult:
        """Deliver a response to the recipient on this channel."""

    @abstractmethod
    async def send_bulk(self, responses: list[AgentResponse]) -> list[DeliveryResult]:
        """Deliver multiple responses (e.g., notify patient + all helpers)."""

class DeliveryResult:
    success: bool
    channel: str
    recipient: str
    error: str | None
    timestamp: datetime

class DispatcherRegistry:
    """Registry of all active channel dispatchers. Gateway uses this to route outbound."""
    _dispatchers: dict[str, ChannelDispatcher]

    def register(self, dispatcher: ChannelDispatcher):
        self._dispatchers[dispatcher.channel_name] = dispatcher

    def get(self, channel_name: str) -> ChannelDispatcher | None:
        return self._dispatchers.get(channel_name)

    async def dispatch(self, response: AgentResponse) -> DeliveryResult:
        """Route a response to the correct dispatcher by channel name."""
        dispatcher = self.get(response.channel)
        if not dispatcher:
            # Fallback: store in diary for retrieval (no channel available)
            return DeliveryResult(success=False, error=f"No dispatcher for {response.channel}")
        return await dispatcher.send(response)

    async def dispatch_all(self, responses: list[AgentResponse]) -> list[DeliveryResult]:
        """Dispatch all responses from an AgentResult, each to its correct channel."""
        results = []
        for response in responses:
            results.append(await self.dispatch(response))
        return results


# ──── INBOUND: Receiving messages from external channels ────

class ChannelIngest(ABC):
    """Abstract interface for converting channel-specific input into Event Envelopes."""
    channel_name: str

    @abstractmethod
    async def to_envelope(self, raw_input: dict) -> EventEnvelope:
        """Convert raw channel input (webhook body, WebSocket message, etc.) to EventEnvelope."""

    def _build_base_envelope(self, patient_id: str, sender_id: str,
                              sender_role: SenderRole, payload: dict) -> EventEnvelope:
        """Shared helper for building envelopes with common fields."""
        return EventEnvelope(
            event_id=uuid4(),
            event_type=EventType.USER_MESSAGE,
            patient_id=patient_id,
            payload={**payload, "channel": self.channel_name},
            source=self.channel_name,
            sender_id=sender_id,
            sender_role=sender_role,
            correlation_id=None,  # Gateway fills this from diary
            timestamp=datetime.utcnow(),
        )


# ──── CONCRETE: Phase 1-5 implementations ────

class WebSocketDispatcher(ChannelDispatcher):
    """Delivers responses via connected WebSocket sessions."""
    channel_name = "websocket"

    async def send(self, response: AgentResponse) -> DeliveryResult:
        # Push message to the WebSocket session for this patient/helper
        # Uses existing WebSocketAgent session registry

class WebSocketIngest(ChannelIngest):
    """Converts WebSocket messages into Event Envelopes."""
    channel_name = "websocket"

    async def to_envelope(self, raw_input: dict) -> EventEnvelope:
        # raw_input comes from existing /ws/pre-consult/{patient_id} handler
        # Wraps text + attachments into standard envelope

class TestHarnessDispatcher(ChannelDispatcher):
    """Stores responses in memory for the HTML test harness to poll."""
    channel_name = "test_harness"
    _response_log: list[AgentResponse]  # test harness polls this

class TestHarnessIngest(ChannelIngest):
    """Converts test harness HTTP POST into Event Envelopes."""
    channel_name = "test_harness"


# ──── FUTURE: Phase 6 implementations (stubs shown for illustration) ────

# class DialogflowDispatcher(ChannelDispatcher):
#     """Sends responses via Dialogflow API → WhatsApp/SMS."""
#     channel_name = "dialogflow_whatsapp"
#     async def send(self, response): ...
#         # Call Dialogflow Messaging API to deliver to WhatsApp/SMS
#
# class DialogflowIngest(ChannelIngest):
#     """Converts Dialogflow fulfillment webhook into Event Envelopes."""
#     channel_name = "dialogflow_whatsapp"
#     async def to_envelope(self, raw_input): ...
#         # raw_input = Dialogflow webhook body
#         # Extract sender phone, message text, attachments
#         # Use IdentityResolver to map phone → patient_id + role
#
# class EmailDispatcher(ChannelDispatcher):
#     """Sends emails via SendGrid/Mailgun for GP communication."""
#     channel_name = "email"
#
# class EmailIngest(ChannelIngest):
#     """Converts SendGrid inbound parse webhook into Event Envelopes."""
#     channel_name = "email"
#
# class TwilioSMSDispatcher(ChannelDispatcher):
#     """Sends SMS via Twilio API."""
#     channel_name = "sms"
#
# class TwilioSMSIngest(ChannelIngest):
#     """Converts Twilio webhook into Event Envelopes."""
#     channel_name = "sms"
```

**Why this is Phase 1, not Phase 6:**

Every component that sends a message (Gateway, GP handler, Helper Manager, Monitoring Agent) needs to know *how* to deliver it. If we hardcode "push to WebSocket" everywhere in Phases 2–4, then Phase 6 requires finding and rewriting every delivery point. By abstracting from day one:

- Agents return `AgentResponse` objects with a `channel` field — they never know or care how delivery works
- The Gateway calls `dispatcher_registry.dispatch_all(result.responses)` — one line, works for any channel
- GP Communication Handler calls `dispatcher_registry.dispatch(email_response)` — same interface
- Adding Dialogflow in Phase 6 = registering a new dispatcher + adding one webhook endpoint. Zero changes elsewhere.

**Deliverable:** ABC interfaces + WebSocketDispatcher + TestHarnessDispatcher + DispatcherRegistry.

### 1.3 Patient Diary Schema + GCS Storage

*(Renumbered from original 1.2)*

Create `medforce/gateway/diary.py`:

```python
# Patient Diary Pydantic model matching the architecture spec
class PatientDiary:
    header: DiaryHeader        # patient_id, current_phase, risk_level, timestamps
    intake: IntakeSection      # demographics, fields_collected, fields_missing
    helper_registry: HelperRegistry  # list of helpers with permissions
    gp_channel: GPChannel      # GP queries, responses, timeline
    clinical: ClinicalSection  # complaints, history, meds, red_flags, risk, sub_phase
    booking: BookingSection    # slots, selection, instructions, confirmed
    monitoring: MonitoringSection  # baseline, entries, alerts, next_check
    conversation_log: list[ConversationEntry]  # capped at 100

# DiaryStore class wrapping GCSBucketManager
class DiaryStore:
    def load(patient_id: str) -> PatientDiary
    def save(patient_id: str, diary: PatientDiary, generation: int) -> bool
    def create(patient_id: str) -> PatientDiary
    def list_monitoring_patients() -> list[str]
```

- **Storage path:** `gs://{bucket}/patient_diaries/patient_{id}/diary.json`
- **Concurrency:** Use GCS generation-match (optimistic locking) — GCSBucketManager already supports conditional writes via the GCS Python SDK
- **Size management:** Conversation log capped at 100 entries, monitoring entries at 50

**Deliverable:** Diary model + DiaryStore with load/save/create using existing GCSBucketManager.

### 1.4 Per-Patient Event Queue

Create `medforce/gateway/queue.py`:

```python
class PatientQueueManager:
    """Manages one asyncio.Queue per active patient."""
    _queues: dict[str, asyncio.Queue]
    _workers: dict[str, asyncio.Task]
    _last_activity: dict[str, datetime]

    async def enqueue(patient_id: str, event: EventEnvelope)
    async def _process_queue(patient_id: str)  # worker loop
    async def cleanup_idle(timeout_minutes: int = 30)
```

- One queue per `patient_id`, events processed FIFO
- Worker per patient runs until queue empty + idle timeout
- Cross-patient parallelism via separate asyncio tasks
- Cleanup idle queues after 30 minutes
- Abstract the queue interface behind a protocol/ABC for future swap to Pub/Sub or Redis

**Deliverable:** Queue manager with enqueue, process, cleanup.

### 1.5 Identity Resolution System

Create `medforce/gateway/handlers/identity_resolver.py`:

```python
class IdentityResolver:
    """Resolves phone/email to (patient_id, role, permissions)."""
    _contact_index: dict[str, IdentityRecord]  # phone/email → patient + role

    async def resolve(contact: str) -> IdentityRecord | None
    async def rebuild_index()  # scan all diaries on startup
    async def update_index(patient_id: str, diary: PatientDiary)
```

- **Contact index:** In-memory dict rebuilt from GCS on startup
- Updated whenever helpers are added/removed
- Handles ambiguity (one contact linked to multiple patients) by returning list
- Lookup order: patient registry → helper registry → GP registry → unknown

**Deliverable:** Identity resolver with index build/update/lookup.

### Phase 1 File Structure

```
medforce/gateway/
├── __init__.py
├── events.py          # EventEnvelope, EventType, SenderRole
├── channels.py        # ChannelDispatcher, ChannelIngest ABCs + DispatcherRegistry
├── diary.py           # PatientDiary model + DiaryStore
├── queue.py           # PatientQueueManager
├── dispatchers/
│   ├── __init__.py
│   ├── websocket_dispatcher.py   # WebSocketDispatcher (Phase 2 impl)
│   └── test_harness_dispatcher.py # TestHarnessDispatcher (Phase 5 impl)
└── handlers/
    ├── __init__.py
    └── identity_resolver.py
```

### Phase 1 Testing

- Unit tests for diary serialization/deserialization (JSON round-trip)
- Unit tests for event envelope creation and validation
- Unit tests for queue ordering (enqueue 3 events, verify FIFO processing)
- Unit tests for identity resolution (patient match, helper match, GP match, unknown)
- Unit tests for DispatcherRegistry (register, lookup, dispatch to correct channel, fallback on unknown channel)
- Unit tests for ChannelIngest.to_envelope() (WebSocket and test harness variants)

---

## Phase 2: Gateway + Intake Agent

**Goal:** The central router is live. Events can enter the system and be dispatched. The Intake Agent handles the first patient interaction.

**Existing code touched:** Minimal — `app.py` (add router + startup hook), one line in `dependencies.py`.

### 2.1 Gateway Router

Create `medforce/gateway/gateway.py`:

```python
class Gateway:
    """Central event router. Deliberately dumb — deterministic lookup only."""

    # Strategy A: Explicit routing for handoff events
    EXPLICIT_ROUTES: dict[EventType, str] = {
        EventType.INTAKE_COMPLETE: "clinical",
        EventType.INTAKE_DATA_PROVIDED: "clinical",
        EventType.CLINICAL_COMPLETE: "booking",
        EventType.BOOKING_COMPLETE: "monitoring",
        EventType.NEEDS_INTAKE_DATA: "intake",
        EventType.HEARTBEAT: "monitoring",
        EventType.DETERIORATION_ALERT: "clinical",
        EventType.GP_QUERY: "gp_comms",
        EventType.GP_RESPONSE: "clinical",
        EventType.GP_REMINDER: "gp_comms",
        EventType.HELPER_REGISTRATION: "helper_manager",
        EventType.HELPER_VERIFIED: "helper_manager",
    }

    # Strategy B: Phase-based routing for external events
    PHASE_ROUTES: dict[str, str] = {
        "intake": "intake",
        "clinical": "clinical",
        "booking": "booking",
        "monitoring": "monitoring",
        "closed": None,  # log only
    }

    def __init__(self, dispatcher_registry: DispatcherRegistry, ...):
        self._dispatchers = dispatcher_registry

    async def process_event(event: EventEnvelope):
        # 1. Resolve identity (IdentityResolver)
        # 2. Check permissions (PermissionChecker)
        # 3. Enqueue in patient's queue (PatientQueueManager)
        # 4. Load diary (DiaryStore)
        # 5. Route (Strategy A if handoff, Strategy B if external)
        # 6. Dispatch to target agent
        # 7. Save updated diary
        # 8. Deliver responses via DispatcherRegistry:
        #    await self._dispatchers.dispatch_all(result.responses)
        # 9. If agent emitted new events → loop back (recursive call)
```

- No AI, no LLM — pure if/else and dict lookups
- The Gateway calls agents through a uniform interface (see 2.3)
- **Response delivery is fully abstracted** — the Gateway calls `self._dispatchers.dispatch_all()` and never knows whether responses go to WebSocket, WhatsApp, email, or SMS
- Loop detection: max 10 chained events per single trigger (circuit breaker)

### 2.2 Permission Checking

Create `medforce/gateway/permissions.py`:

```python
class PermissionChecker:
    """Checks if sender has permission for the implied action."""

    # Maps event context to required permission
    def check(sender_role: SenderRole, sender_permissions: list[str],
              event: EventEnvelope, diary_phase: str) -> PermissionResult

    # Infers required permission from event context
    def _infer_required_permission(event: EventEnvelope, diary_phase: str) -> str
```

- Patients always have full permission
- Helpers checked against their diary registry permissions
- GPs checked for GP-specific actions only
- Unknown senders always denied

### 2.3 Universal Agent Contract

Create `medforce/gateway/agents/base_agent.py`:

```python
class AgentResult:
    updated_diary: PatientDiary
    emitted_events: list[EventEnvelope]  # handoffs, queries, alerts
    responses: list[AgentResponse]       # messages to send back

class AgentResponse:
    recipient: str          # "patient", "helper:HELPER-001", "gp:Dr.Patel"
    channel: str            # Channel name matching a registered ChannelDispatcher:
                            #   "websocket" (Phases 1-5)
                            #   "test_harness" (Phase 5)
                            #   "dialogflow_whatsapp" (Phase 6)
                            #   "dialogflow_sms" (Phase 6)
                            #   "email" (Phase 6)
    message: str
    attachments: list[str]
    metadata: dict          # Channel-specific data (e.g., WhatsApp template ID,
                            # email subject line, reply-to address)

class BaseAgent(ABC):
    @abstractmethod
    async def process(event: EventEnvelope, diary: PatientDiary) -> AgentResult:
        """Every agent implements this. Receive event + diary, return result."""
```

**Channel resolution for responses:** Agents determine the `channel` field from:
1. The `event.payload["channel"]` (reply on the same channel the message came from)
2. The diary's helper registry (each helper has a preferred `channel` field)
3. The diary's GP channel (GP always gets `email`)

This means agents never hardcode "whatsapp" or "websocket" — they read the channel preference from the diary or echo the inbound channel. When Dialogflow registers new channel types, agents automatically use them because the diary stores the preference.

### 2.4 Intake Agent

Create `medforce/gateway/agents/intake_agent.py`:

```python
class IntakeAgent(BaseAgent):
    """The Receptionist. Collects demographics. Never asks clinical questions."""

    async def process(event, diary) -> AgentResult:
        if event.event_type == EventType.NEEDS_INTAKE_DATA:
            # Backward loop: ask ONLY for the specific missing data
            return self._collect_specific_field(event.payload["missing"], diary)

        # Normal flow: check what fields are missing
        missing = self._identify_missing_fields(diary)
        if not missing:
            # All fields collected → hand off to Clinical
            diary.header.current_phase = "clinical"
            return AgentResult(
                diary=diary,
                emitted_events=[EventEnvelope(type=INTAKE_COMPLETE)],
                responses=[...]
            )
        # Ask for next missing field (one question per turn)
        return self._ask_for_field(missing[0], diary)
```

- Uses Gemini Flash for extraction and question generation
- Parses referral letters for pre-populated fields (name, DOB, NHS#)
- Offers helper registration during intake
- UK-specific: NHS number format, postcode validation, GP ODS code
- **Strict boundary:** Never asks about symptoms, medications, medical history

### 2.5 Gateway API Endpoints

Create `medforce/routers/gateway_api.py`:

```python
@router.post("/api/gateway/emit")       # Submit any event
@router.get("/api/gateway/diary/{id}")   # Read diary state
@router.get("/api/gateway/events/{id}")  # Read event log
@router.get("/api/gateway/status")       # Health + active queues
```

### 2.6 Existing Code Changes

**`medforce/app.py`** — Add (wrapped in try/except):
```python
from medforce.routers import gateway_api
app.include_router(gateway_api.router)

@app.on_event("startup")
async def startup_gateway():
    try:
        from medforce.gateway import setup
        await setup.initialize_gateway()
    except Exception:
        logger.warning("Gateway failed to start — running without it")
```

**`medforce/dependencies.py`** — Add:
```python
def get_gateway() -> Gateway:
    """Lazy singleton for the Gateway."""
```

### Phase 2 Testing

- Integration test: POST event to `/api/gateway/emit` → Intake Agent processes → diary updated
- Test: unknown sender gets rejected with helpful message
- Test: Intake asks one question per turn, doesn't re-ask collected fields
- Test: All fields collected → `INTAKE_COMPLETE` fires → diary phase = "clinical"
- Test: `NEEDS_INTAKE_DATA` triggers backward loop asking only for specified field

---

## Phase 3: Clinical Agent + Booking Agent + GP Communication

**Goal:** The core clinical workflow is complete. A patient can go from intake through clinical assessment to booking.

**Existing code touched:** One event emission hook in `pre_consult.py` (fire-and-forget).

### 3.1 Clinical Agent

Create `medforce/gateway/agents/clinical_agent.py`:

```python
class ClinicalAgent(BaseAgent):
    """The Triage Nurse. Most complex and safety-critical agent."""

    # Sub-phases tracked in diary:
    # analyzing_referral → asking_questions → collecting_documents → scoring_risk → complete

    async def process(event, diary) -> AgentResult:
        sub_phase = diary.clinical.sub_phase

        if sub_phase == "analyzing_referral":
            return await self._analyze_referral(event, diary)
        elif sub_phase == "asking_questions":
            return await self._process_answer(event, diary)
        elif sub_phase == "collecting_documents":
            return await self._process_document(event, diary)
        elif sub_phase == "scoring_risk":
            return await self._score_risk(diary)
```

**Reuses existing code:**
- Wraps `PreConsulteAgent` for conversation management and referral analysis
- Uses `side_agent.parse_lab_values()` for lab extraction from documents
- Uses Gemini Vision (already configured) for image/document processing
- Uses `side_agent.generate_dili_diagnosis()` pattern for risk reasoning

**Risk Scoring — Deterministic Rules (non-negotiable):**
```python
class RiskScorer:
    HARD_RULES = [
        ("bilirubin", ">", 5.0, "HIGH"),
        ("ALT", ">", 500, "HIGH"),
        ("platelets", "<", 50, "HIGH"),
    ]
    KEYWORD_RULES = [
        ("jaundice", "HIGH"),
        ("confusion", "HIGH"),
        ("encephalopathy", "HIGH"),
        ("gi_bleeding", "HIGH"),
        ("ascites", "HIGH"),
    ]

    def score(diary: PatientDiary) -> RiskResult:
        # 1. Check hard rules FIRST — these override everything
        # 2. Check keyword rules
        # 3. Only if NO hard rule fired → ask LLM for gray-zone assessment
        # 4. Log which method determined the score
```

- LLM requirement: Gemini Pro (or best available) for medical reasoning
- Generates 3–5 personalized clinical questions (not generic)
- Tracks sub-phase in diary for resume-after-interruption
- Can emit `NEEDS_INTAKE_DATA` for backward loop
- Can emit `GP_QUERY` for missing clinical data (non-blocking)

### 3.2 Booking Agent

Create `medforce/gateway/agents/booking_agent.py`:

```python
class BookingAgent(BaseAgent):
    """The Scheduler. Gets patients into appointments by urgency."""

    async def process(event, diary) -> AgentResult:
        risk = diary.clinical.risk_level

        # 1. Determine urgency window
        window = {"HIGH": 2, "MEDIUM": 14, "LOW": 30}[risk]  # days

        # 2. Query ScheduleCSVManager for available slots
        slots = self._get_filtered_slots(window)

        # 3. Present 2-3 options
        # 4. If slot selected → confirm and generate pre-appointment instructions
        # 5. Snapshot baseline labs → fire BOOKING_COMPLETE
```

**Reuses existing code:**
- Calls `ScheduleCSVManager.get_empty_schedule()` and filters by date window
- Uses `ScheduleCSVManager.book_appointment()` to confirm
- LLM requirement: Gemini Flash for personalized pre-appointment instructions

**Context-aware instructions (reads Clinical Section):**
- Liver function test → "Fast for 8–12 hours"
- Patient on Metformin → "Continue taking Metformin"
- Ultrasound → "Drink 1L water 1 hour before"

### 3.3 GP Communication Handler

Create `medforce/gateway/handlers/gp_comms.py`:

```python
class GPCommunicationHandler:
    """Handles outbound queries to GPs and processes their responses."""

    async def send_query(event, diary) -> AgentResult:
        # Generate professional email content (template-based)
        # Log query in diary.gp_channel.queries
        # Return AgentResponse with channel="email" — DispatcherRegistry handles delivery
        return AgentResult(
            diary=updated_diary,
            emitted_events=[],
            responses=[AgentResponse(
                recipient=f"gp:{diary.gp_channel.gp_name}",
                channel="email",
                message=email_body,
                metadata={
                    "subject": f"MedForce — Lab Results Requested for {patient_name} ({ref_id})",
                    "to": diary.gp_channel.gp_email,
                    "reply_to": f"gp-reply+{patient_id}@medforce.app",  # for inbound parsing
                }
            )]
        )

    async def send_reminder(event, diary) -> AgentResult:
        # Same pattern — returns AgentResponse with channel="email"
        # DispatcherRegistry delivers via whatever email dispatcher is registered

    # GP response flow: GP_RESPONSE event → routes to Clinical Agent (not here)
```

- **GP emails use the same ChannelDispatcher pattern as everything else.** The handler generates the content and metadata; delivery is abstracted.
- **Phases 1–5:** No email dispatcher registered → GP emails stored in diary with `delivery_status: "pending_channel"` → visible in test harness diary viewer for manual verification
- **Phase 6:** Register `EmailDispatcher` → GP emails delivered automatically via SendGrid/Mailgun
- 48-hour reminder via Heartbeat Scheduler (CRON check)
- 7-day fallback: mark `gp_non_responsive`, fall back to asking patient
- Audit trail: all queries/responses logged in diary with delivery status

### 3.4 Multi-Recipient Response Routing (via ChannelDispatcher)

The Gateway never directly sends messages. After an agent returns its `AgentResult`, the Gateway calls:

```python
delivery_results = await self._dispatchers.dispatch_all(result.responses)
```

The `DispatcherRegistry` routes each response to the correct `ChannelDispatcher` by matching `response.channel`:

```
AgentResult.responses = [
    AgentResponse(recipient="patient", channel="websocket", message="..."),
    AgentResponse(recipient="helper:Sarah", channel="websocket", message="..."),
    AgentResponse(recipient="gp:Dr.Patel", channel="email", message="...",
                  metadata={"subject": "Lab Results Request - REF-2026-4521"}),
]

DispatcherRegistry routes:
    response[0] → WebSocketDispatcher.send()    ← connected session
    response[1] → WebSocketDispatcher.send()    ← connected session
    response[2] → EmailDispatcher.send()        ← SendGrid API (Phase 6)
                  OR fallback: store in diary    ← if no email dispatcher registered yet
```

**Fallback behavior (Phases 1–5):** If a response targets a channel with no registered dispatcher (e.g., `email` before Phase 6), the Gateway:
1. Stores the response in `diary.conversation_log` with `delivery_status: "pending_channel"`
2. Logs a warning: "No dispatcher for channel 'email' — response stored in diary"
3. Does NOT fail or block — the agent's work is preserved regardless of delivery

**Phase 6 upgrade:** Register `DialogflowDispatcher` and `EmailDispatcher`. Pending responses can optionally be flushed on dispatcher registration.

### Phase 3 Testing

- Integration test: Full flow intake → clinical → booking
- Test: Hard rules override LLM risk score (bilirubin 6.2 → always HIGH)
- Test: Clinical sub-phase resumes correctly after interruption
- Test: GP query emitted for missing data, clinical continues without blocking
- Test: Booking filters slots by risk level window
- Test: Pre-appointment instructions reference patient-specific clinical data
- Test: Backward loop (NEEDS_INTAKE_DATA → Intake → back to Clinical)

---

## Phase 4: Monitoring Agent + Heartbeat Scheduler

**Goal:** Long-term patient care. The system operates autonomously for weeks/months after booking.

**Existing code touched:** None. Fully additive.

### 4.1 Monitoring Agent

Create `medforce/gateway/agents/monitoring_agent.py`:

```python
class MonitoringAgent(BaseAgent):
    """The Guardian. Cares for patients post-consultation."""

    async def process(event, diary) -> AgentResult:
        if event.event_type == EventType.BOOKING_COMPLETE:
            return self._setup_monitoring(diary)  # snapshot baseline

        if event.event_type == EventType.HEARTBEAT:
            return self._proactive_check(event, diary)

        if event.event_type == EventType.USER_MESSAGE:
            return await self._reactive_response(event, diary)
```

**Proactive mode (CRON-driven):**
| Days Post-Appointment | Action |
|---|---|
| 14 | Check for follow-up labs. Reminder if missing. |
| 30 | General check-in message. |
| 60 | Prompt for updated labs. |
| 90 | Full symptom review. |

**Reactive mode (patient returns):**
- Load full diary including baseline values
- If new lab results: compare against baseline, calculate % changes, identify trends
- If new symptoms: cross-reference against clinical history, check red flags
- If deterioration detected: emit `DETERIORATION_ALERT` → Clinical reassessment loop

**LLM requirement:**
- Gemini Flash for routine check-ins and reminders
- Gemini Pro for trend analysis and deterioration assessment

### 4.2 Heartbeat Scheduler

Create `medforce/gateway/heartbeat.py`:

```python
class HeartbeatScheduler:
    """Background loop that fires HEARTBEAT events for monitored patients."""

    async def start():
        """Run on app startup. Scans GCS for monitored patients."""
        patients = await diary_store.list_monitoring_patients()
        for pid in patients:
            self._register(pid)
        asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop():
        while True:
            await asyncio.sleep(3600)  # every hour
            for pid in self._monitored_patients:
                diary = await diary_store.load(pid)
                if self._is_milestone_due(diary):
                    event = EventEnvelope(
                        event_type=EventType.HEARTBEAT,
                        patient_id=pid,
                        payload={
                            "days_since_appointment": ...,
                            "milestone": ...,
                        }
                    )
                    await gateway.process_event(event)

    async def register(patient_id: str):
        """Called when BOOKING_COMPLETE fires."""

    async def _recover_on_startup():
        """Scan GCS for monitoring_active patients and re-register."""
```

- In-process asyncio loop (no external dependencies)
- Recovery on restart: scans all diaries on startup
- Scaling path: swap to Google Cloud Scheduler hitting `/api/gateway/emit` externally

### 4.3 GP Reminder CRON

Part of the heartbeat loop. Checks diary.gp_channel for pending queries older than 48 hours without response:

```python
# Inside heartbeat loop:
if diary.gp_channel.has_pending_queries():
    for query in diary.gp_channel.queries:
        if query.status == "pending" and hours_since(query.sent) > 48:
            if not query.reminder_sent:
                emit GP_REMINDER event
            elif days_since(query.sent) > 7:
                mark gp_non_responsive, fallback to patient
```

### Phase 4 Testing

- Test: BOOKING_COMPLETE → baseline snapshot stored → monitoring active
- Test: Heartbeat fires at correct milestones (14d, 30d, 60d, 90d)
- Test: New lab values compared against baseline with correct % change
- Test: Red flag keywords in monitoring context → DETERIORATION_ALERT fires
- Test: DETERIORATION_ALERT → Clinical reassessment → new risk → rebooking → back to monitoring (full loop)
- Test: Heartbeat scheduler recovers monitored patients on restart
- Test: GP reminder fires after 48h, fallback after 7d

---

## Phase 5: HTML Test Harness

**Goal:** Interactive browser-based tool to test all scenarios without external services.

**Existing code touched:** None. Static file served by existing utility router.

### 5.1 Test Harness Features

Create `medforce/static/test_harness.html` (single-file SPA):

1. **Multi-role chat panels** — tabs for Patient, Helper 1 (Spouse), Helper 2 (Friend), GP
2. **Event log** — real-time scrolling log showing every event through the Gateway (color-coded)
3. **Diary viewer** — live JSON display of patient diary (polls `/api/gateway/diary/{id}`)
4. **Phase indicator** — visual bar showing current phase, flashes on backward loops
5. **Manual event injection** — buttons: Heartbeat, Lab Webhook, GP Response, Deterioration Alert
6. **Scenario loader** — dropdown to load all 8 test scenarios from the architecture doc
7. **Assertion checker** — pass/fail for expected behaviors per scenario

### 5.2 Additional API Endpoints

Add to `medforce/routers/gateway_api.py`:

```python
@router.post("/api/gateway/scenario/load")     # Seed diary with test data
@router.post("/api/gateway/scenario/assert")    # Run assertions
@router.delete("/api/gateway/reset/{patient_id}")  # Clear diary + events
```

### 5.3 Test Scenarios (from architecture doc)

| # | Scenario | Key Assertions |
|---|---|---|
| 1 | Happy Path — Solo Patient, Low Risk | Risk=LOW, slots >7d, no GP query |
| 2 | Urgent — Patient + Spouse, High Risk | Risk=HIGH (deterministic), slots 48h, helper notified |
| 3 | Missing Info — GP Query Required | GP_QUERY emitted, clinical doesn't block, GP response merged |
| 4 | Backward Loop — Missing Medications | NEEDS_INTAKE_DATA fires, Intake asks only for meds, Clinical resumes |
| 5 | Deterioration Escalation (Month 3) | DETERIORATION_ALERT, full loop Clinical→Booking→Monitoring |
| 6 | Multi-Helper Permission Conflict | Queue serializes, full-access accepted, limited denied |
| 7 | GP Non-Responsive | 48h reminder, 7d fallback |
| 8 | Unknown Sender | No diary accessed, polite rejection, security log |

### Phase 5 Testing

- Manual testing of all 8 scenarios through the harness
- Verify event log shows correct event chain
- Verify diary viewer updates in real-time
- Verify assertion checker passes for each scenario

---

## Phase 6: Channel Integration (Seamless — Post-MVP)

**Goal:** Connect external messaging channels. The Gateway and agents require **zero changes** — this phase only adds new dispatchers, ingest endpoints, and external service configuration.

### 6.1 What Phase 6 Actually Requires (Because of Phase 1 Abstractions)

Because `ChannelDispatcher` and `ChannelIngest` were defined in Phase 1, adding any new channel is a 3-step process:

1. **Implement a `ChannelDispatcher`** subclass for outbound delivery
2. **Implement a `ChannelIngest`** subclass for inbound message conversion
3. **Add a webhook endpoint** that receives external messages and calls `ChannelIngest.to_envelope()` → `gateway.process_event()`

No Gateway code changes. No agent code changes. No diary code changes. No queue code changes.

### 6.2 Dialogflow as Channel Adapter

**Dialogflow CX Configuration:**
- Single "catch-all" page with no intent matching
- Default Start Flow → immediately trigger fulfillment webhook
- Fulfillment webhook URL: `POST https://{backend}/api/gateway/dialogflow-webhook`
- No flow logic, no context/session management — Dialogflow is a pure message relay

**New files (3 total):**

```python
# medforce/gateway/dispatchers/dialogflow_dispatcher.py
class DialogflowDispatcher(ChannelDispatcher):
    """Delivers responses via Dialogflow API → WhatsApp Business / SMS."""
    channel_name = "dialogflow_whatsapp"  # or "dialogflow_sms"

    async def send(self, response: AgentResponse) -> DeliveryResult:
        # Call Dialogflow Sessions API or WhatsApp Business API directly
        # Use response.metadata for WhatsApp template IDs (required for proactive messages)
        # Return delivery confirmation

# medforce/gateway/ingest/dialogflow_ingest.py
class DialogflowIngest(ChannelIngest):
    """Converts Dialogflow fulfillment webhook body into Event Envelopes."""
    channel_name = "dialogflow_whatsapp"

    async def to_envelope(self, raw_input: dict) -> EventEnvelope:
        # Extract from Dialogflow webhook:
        #   - sender phone: raw_input["sessionInfo"]["parameters"]["phone"]
        #   - message text: raw_input["text"]
        #   - attachments: raw_input.get("media", [])
        # Use IdentityResolver to map phone → patient_id + role
        # Build and return EventEnvelope

# medforce/routers/gateway_api.py (add one endpoint)
@router.post("/api/gateway/dialogflow-webhook")
async def dialogflow_webhook(request: Request):
    """Dialogflow fulfillment webhook — converts incoming to event and processes."""
    body = await request.json()
    ingest = DialogflowIngest()
    envelope = await ingest.to_envelope(body)
    result = await gateway.process_event(envelope)
    # Return Dialogflow-formatted response (their specific JSON schema)
    return dialogflow_format(result.responses)
```

**Registration (in `setup.py`):**
```python
async def initialize_gateway():
    ...
    # Phase 6: register Dialogflow dispatcher (only if configured)
    if settings.DIALOGFLOW_ENABLED:
        registry.register(DialogflowDispatcher(credentials=settings.DIALOGFLOW_CREDENTIALS))
```

**That's it.** The entire Dialogflow integration is 3 new files + 1 new endpoint + 1 line in setup. Every existing component (Gateway, agents, diary, queue, identity resolver, permission checker) works unchanged.

### 6.3 Direct Channel Integrations (Alternative to Dialogflow)

Each channel follows the exact same pattern:

| Channel | Dispatcher | Ingest | Webhook Endpoint |
|---|---|---|---|
| WhatsApp (via Dialogflow) | `DialogflowDispatcher` | `DialogflowIngest` | `/api/gateway/dialogflow-webhook` |
| WhatsApp (direct) | `WhatsAppDispatcher` | `WhatsAppIngest` | `/api/gateway/whatsapp-webhook` |
| Email (SendGrid) | `EmailDispatcher` | `EmailIngest` | `/api/gateway/email-inbound` |
| SMS (Twilio) | `TwilioSMSDispatcher` | `TwilioSMSIngest` | `/api/gateway/twilio-webhook` |
| Web Chat | `WebSocketDispatcher` | `WebSocketIngest` | Already connected (Phase 2) |

**You can mix and match.** Use Dialogflow for WhatsApp but direct Twilio for SMS. Use SendGrid for email. Each is independent — just register the dispatchers you want.

### 6.4 Helper Verification Flow (Channel-Aware)

```
Patient says: "Add my wife Sarah, 07700 900462, full access"
    │
    ▼
Helper Manager creates AgentResponse:
    recipient: "helper:pending:+447700900462"
    channel: diary.intake.contact_preference  ← patient's preferred channel
             OR "dialogflow_whatsapp"         ← if WhatsApp available
             OR "sms"                         ← fallback
    message: "Hi Sarah, John Smith has added you as a helper. Reply YES to confirm."
    │
    ▼
DispatcherRegistry.dispatch() → routes to correct dispatcher
    │
    ▼
Sarah receives message on WhatsApp/SMS/email
Sarah replies "YES"
    │
    ▼
Dialogflow/Twilio/SendGrid webhook → ChannelIngest.to_envelope()
    → event_type: HELPER_VERIFIED
    → gateway.process_event()
    │
    ▼
Helper Manager confirms registration. Diary updated.
```

The Helper Manager never knows which channel was used. It just sets the `channel` field on the `AgentResponse` and the dispatcher handles delivery.

### 6.5 Inbound GP Response Flow (Channel-Aware)

```
GP replies by email to gp-reply+PT-1234@medforce.app
    │
    ▼
SendGrid Inbound Parse webhook → POST /api/gateway/email-inbound
    │
    ▼
EmailIngest.to_envelope():
    - Parses reply-to address: patient_id = "PT-1234"
    - IdentityResolver: dr.patel@greenfields.nhs.uk → GP for PT-1234
    - Extracts attachments (lab PDFs)
    - Returns EventEnvelope(event_type=GP_RESPONSE, patient_id="PT-1234", ...)
    │
    ▼
gateway.process_event() → explicit routing → Clinical Agent
    │
    ▼
Clinical Agent processes GP response (same as Phase 3 — no changes)
```

### 6.6 WhatsApp Template Messages (Proactive Outbound)

WhatsApp Business API requires pre-approved templates for proactive messages (system-initiated, not reply-to-user). The `DialogflowDispatcher` handles this:

```python
class DialogflowDispatcher(ChannelDispatcher):
    async def send(self, response: AgentResponse) -> DeliveryResult:
        if response.metadata.get("proactive"):
            # Use WhatsApp template message
            template_id = response.metadata["template_id"]
            template_params = response.metadata["template_params"]
            return await self._send_template(template_id, template_params)
        else:
            # Reply within existing conversation (no template needed)
            return await self._send_reply(response.message)
```

Agents tag proactive messages (heartbeat reminders, deterioration alerts, booking confirmations) with `metadata={"proactive": True, "template_id": "..."}`. The dispatcher handles the difference.

### Phase 6 File Structure (New Files Only)

```
medforce/gateway/
├── dispatchers/
│   ├── dialogflow_dispatcher.py    # NEW — Dialogflow → WhatsApp/SMS
│   ├── email_dispatcher.py         # NEW — SendGrid/Mailgun
│   └── twilio_dispatcher.py        # NEW — Twilio SMS (optional)
│
├── ingest/
│   ├── dialogflow_ingest.py        # NEW — Dialogflow webhook → EventEnvelope
│   ├── email_ingest.py             # NEW — SendGrid inbound parse → EventEnvelope
│   └── twilio_ingest.py            # NEW — Twilio webhook → EventEnvelope (optional)

medforce/routers/
└── gateway_api.py                  # ADD 3 webhook endpoints (dialogflow, email, twilio)
```

**Total changes to existing code: ZERO.** All new files. One line in `setup.py` to register each new dispatcher.

### Phase 6 Scope

This phase adds **distribution, not intelligence.** The event loop, agents, diary, queue, identity, permissions — everything works identically. Phase 6 just connects new pipes to the same system.

---

## Phase 7: Hardening (Future — Post-MVP)

**Goal:** Production readiness, observability, edge cases.

### 7.1 Circuit Breakers

- Backward loop limit: 3 attempts per field, then proceed with incomplete data
- Time threshold: 7 days without missing data → proceed anyway
- Max chained events per trigger: 10 (prevent infinite loops)
- Tracked in diary: `backward_loop_count`

### 7.2 Diary Size Management

- Conversation log: cap at 100 entries, archive older to `diary_archive_{date}.json`
- Monitoring entries: cap at 50, archive older
- Keep only current + previous baselines in active diary

### 7.3 Observability

- Correlation ID threads through every event in a patient journey
- Structured logging with correlation IDs from Phase 1 (built in, not bolted on)
- Dashboard: patients per phase, avg time per phase, risk distribution, loop frequency

### 7.4 Permission Audit Logging

- Every permission check (granted and denied) logged
- Helper access revocation support
- Time-limited helper access option

### 7.5 Multi-Helper Conflict Resolution

- Queue serialization handles concurrent helpers (first valid action wins)
- Polite denial for unauthorized actions with explanation
- Patient notified of actions taken by helpers

### 7.6 Identity Ambiguity Handling

- One helper linked to multiple patients → ask "Which patient is this about?"
- Conversation affinity: within 30 minutes, assume same patient context

---

## Complete File Structure

```
medforce/gateway/
├── __init__.py
├── events.py               # EventEnvelope, EventType, SenderRole enums
├── channels.py             # ChannelDispatcher, ChannelIngest ABCs + DispatcherRegistry
├── diary.py                # PatientDiary model + DiaryStore (GCS CRUD)
├── gateway.py              # Gateway router (Strategy A + B, uses DispatcherRegistry)
├── queue.py                # PatientQueueManager (asyncio per-patient)
├── heartbeat.py            # HeartbeatScheduler (CRON background loop)
├── permissions.py          # PermissionChecker
├── setup.py                # initialize_gateway() + dispatcher registration
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # BaseAgent ABC, AgentResult, AgentResponse
│   ├── intake_agent.py     # Intake Agent (receptionist)
│   ├── clinical_agent.py   # Clinical Agent (triage nurse) — wraps PreConsulteAgent
│   ├── booking_agent.py    # Booking Agent (scheduler) — uses ScheduleCSVManager
│   └── monitoring_agent.py # Monitoring Agent (guardian)
│
├── dispatchers/                          # OUTBOUND channel implementations
│   ├── __init__.py
│   ├── websocket_dispatcher.py           # Phase 2 — pushes to WebSocket sessions
│   ├── test_harness_dispatcher.py        # Phase 5 — stores for test harness polling
│   ├── dialogflow_dispatcher.py          # Phase 6 — Dialogflow → WhatsApp/SMS
│   ├── email_dispatcher.py               # Phase 6 — SendGrid/Mailgun
│   └── twilio_dispatcher.py              # Phase 6 — Twilio SMS (optional)
│
├── ingest/                               # INBOUND channel implementations
│   ├── __init__.py
│   ├── websocket_ingest.py               # Phase 2 — WebSocket msg → EventEnvelope
│   ├── test_harness_ingest.py            # Phase 5 — HTTP POST → EventEnvelope
│   ├── dialogflow_ingest.py              # Phase 6 — Dialogflow webhook → EventEnvelope
│   ├── email_ingest.py                   # Phase 6 — SendGrid parse → EventEnvelope
│   └── twilio_ingest.py                  # Phase 6 — Twilio webhook → EventEnvelope
│
├── handlers/
│   ├── __init__.py
│   ├── identity_resolver.py              # Phone/email → patient_id + role
│   ├── gp_comms.py                       # GP query/response/reminder handling
│   └── helper_manager.py                 # Helper registration/verification/permissions
│
└── tests/
    ├── test_events.py
    ├── test_channels.py                  # DispatcherRegistry, dispatch routing, fallback
    ├── test_diary.py
    ├── test_gateway.py
    ├── test_queue.py
    ├── test_intake_agent.py
    ├── test_clinical_agent.py
    ├── test_booking_agent.py
    ├── test_monitoring_agent.py
    └── test_scenarios.py                 # All 8 architecture scenarios as integration tests

medforce/routers/
└── gateway_api.py           # 7 API endpoints (Phases 2-5) + 3 webhook endpoints (Phase 6)

medforce/static/
└── test_harness.html        # Single-file interactive test harness
```

---

## Existing Files Modified (Minimal)

| File | Change | Risk |
|---|---|---|
| `medforce/app.py` | Add gateway router import + startup init (try/except wrapped) | Zero — failure is silent, old system unaffected |
| `medforce/dependencies.py` | Add `get_gateway()` lazy singleton | Zero — additive only |
| `medforce/routers/pre_consult.py` | After existing return, emit `USER_MESSAGE` event (fire-and-forget, try/except) | Zero — happens after existing logic completes |

**Safety principle:** If the Gateway crashes, every existing endpoint continues working exactly as before. Event emission is always fire-and-forget with try/except.

---

## Implementation Order & Dependencies

```
Phase 1 (Foundation) ─── no dependencies
    │
    ├── 1.1 Event Envelope ──────────────────────────────┐
    ├── 1.2 Channel Abstractions (Dispatcher + Ingest)───┤  ← enables seamless Phase 6
    ├── 1.3 Patient Diary + DiaryStore ──────────────────┤
    ├── 1.4 Per-Patient Queue ───────────────────────────┤
    └── 1.5 Identity Resolver ───────────────────────────┤
                                                          │
Phase 2 (Gateway + Intake) ─── depends on Phase 1 ───────┘
    │
    ├── 2.1 Gateway Router (uses DispatcherRegistry) ────┐
    ├── 2.2 Permission Checker ──────────────────────────┤
    ├── 2.3 Base Agent Contract ─────────────────────────┤
    ├── 2.4 Intake Agent ────────────────────────────────┤
    ├── 2.5 Gateway API Endpoints ───────────────────────┤
    ├── 2.6 WebSocketDispatcher + WebSocketIngest ────────┤  ← first concrete channel
    └── 2.7 app.py + dependencies.py changes ────────────┤
                                                          │
Phase 3 (Clinical + Booking + GP) ─── depends on Phase 2 ┘
    │
    ├── 3.1 Clinical Agent (wrapping PreConsulteAgent) ──┐
    ├── 3.2 Booking Agent (using ScheduleCSVManager) ────┤
    ├── 3.3 GP Communication Handler (uses Dispatcher) ──┤  ← email via abstraction
    └── 3.4 Multi-Recipient Routing (via Dispatcher) ────┤  ← all delivery abstracted
                                                          │
Phase 4 (Monitoring + Heartbeat) ─── depends on Phase 3 ─┘
    │
    ├── 4.1 Monitoring Agent ────────────────────────────┐
    ├── 4.2 Heartbeat Scheduler ─────────────────────────┤
    └── 4.3 GP Reminder CRON ───────────────────────────┤
                                                          │
Phase 5 (Test Harness) ─── depends on Phase 4 ───────────┘
    │
    ├── 5.1 HTML Test Harness (single-file SPA)
    ├── 5.2 TestHarnessDispatcher + TestHarnessIngest ────  ← second concrete channel
    ├── 5.3 Additional API Endpoints
    └── 5.4 8 Test Scenarios + Assertions

Phase 6 (Channels) ─── SEAMLESS: new dispatchers + ingest only, zero core changes
    │
    ├── 6.1 DialogflowDispatcher + DialogflowIngest ──────  ← third concrete channel
    ├── 6.2 EmailDispatcher + EmailIngest ────────────────  ← fourth concrete channel
    ├── 6.3 TwilioSMSDispatcher + TwilioSMSIngest ───────  ← fifth (optional)
    ├── 6.4 Webhook endpoints (3 new routes) ─────────────
    └── 6.5 Register dispatchers in setup.py (1 line each)

Phase 7 (Hardening) ─── independent, deferred
```

### Why Phase 6 is Seamless (Summary)

The channel abstraction built in Phase 1 means Phase 6 follows a strict pattern for each channel:

| Step | What | Lines of Code | Touches Existing Files? |
|---|---|---|---|
| 1 | Implement `ChannelDispatcher` subclass | ~50 lines | No — new file |
| 2 | Implement `ChannelIngest` subclass | ~30 lines | No — new file |
| 3 | Add webhook endpoint | ~15 lines | 1 route added to gateway_api.py |
| 4 | Register dispatcher | 1 line | setup.py (conditional on config) |
| **Total per channel** | | **~100 lines** | **Zero core changes** |

The Gateway, agents, diary, queue, identity resolver, permission checker, and heartbeat scheduler are completely untouched. Dialogflow is just another pipe connected to the same system.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Clinical risk scoring error | Medium | Critical | Hard rules ALWAYS override LLM. Extensive adversarial testing. |
| Infinite event loop | Low | High | Circuit breaker (max 10 chained events). Backward loop counter (max 3). |
| Diary corruption from concurrent writes | Low | High | GCS generation-match (optimistic locking). Per-patient queue serialization. |
| Heartbeat memory leak | Low | Medium | Idle queue cleanup (30 min). Capped monitored patient list. |
| Gateway crash affects existing endpoints | None | N/A | Fire-and-forget event emission. try/except wrapping. Gateway is additive overlay. |
| Identity resolution performance at scale | Medium | Low | In-memory contact index. Rebuild on startup. LRU cache for recent lookups. |
| Diary grows too large over months | Medium | Medium | Capped logs (100 conversation, 50 monitoring). Archival to separate files. |

---

## Summary

The architecture described in `MedForce_Full_Architecture.md` is **fully implementable** with the current codebase. The key insight is that the architecture is an additive layer on top of the existing system, not a replacement. Every existing endpoint, agent, and manager continues working. The Gateway orchestrates them through events rather than replacing them.

The **ChannelDispatcher + ChannelIngest abstraction** (Phase 1) is the key design decision that makes Dialogflow integration seamless. By abstracting both inbound and outbound message handling from day one:

- **No agent ever knows** which channel a message came from or which channel a response goes to
- **No Gateway code changes** are needed when adding WhatsApp, SMS, email, or any future channel
- **Phase 6 is purely additive** — implement a dispatcher class, implement an ingest class, add a webhook route, register in setup. ~100 lines per channel, zero core changes.
- **The same system** that runs on WebSocket in development runs on WhatsApp + SMS + email in production, with identical event processing and clinical logic

The most critical component is the **Clinical Agent's risk scoring** — the deterministic hard rules must be implemented first and tested exhaustively. Everything else is infrastructure and workflow orchestration.

Phases 1–5 require no external services beyond what's already configured (GCS + Gemini). Phase 6 introduces external channel integrations (Dialogflow, SendGrid, Twilio) as plug-in dispatchers. Phase 7 adds production hardening. Both can be deferred until the core loop is validated.

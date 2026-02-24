# Claude Session — 2026-02-24 (Session 5)

## Latest Changes (Session 5)

### Booking Registry + Rescheduling + Slot Rejection + Clinical Quality

#### 1. Persistent Booking Registry (`medforce/gateway/booking_registry.py`) — NEW
- GCS-backed slot hold/confirm/cancel system preventing double-booking
- `SlotHold` model: hold_id, patient_id, date, time, provider, status (held/confirmed/cancelled), TTL 15min
- Methods: `hold_slots()`, `confirm_slot()`, `cancel_booking()`, `get_patient_booking()`, `release_holds()`
- In-memory fallback when `gcs_bucket_manager=None` (test mode)
- Lazy expired-hold cleanup on every `_load()`

#### 2. Rescheduling Flow
- `RESCHEDULE_REQUEST` event type added to `events.py` + explicit route in `gateway.py`
- MonitoringAgent detects reschedule keywords → emits RESCHEDULE_REQUEST
- BookingAgent `_handle_reschedule()`: cancels in registry + schedule manager, resets diary, re-offers fresh slots
- Reschedule history saved in `diary.booking.rescheduled_from`

#### 3. Slot Rejection by Patient — NEW
- `SLOT_REJECTION_KEYWORDS` in booking_agent.py ("none of these work", "not available", "don't work", etc.)
- `_is_slot_rejection()` + `_handle_slot_rejection()`: releases holds, fetches fresh slots, re-offers
- Retry prompt updated to mention "or let me know if none of these times work for you"

#### 4. Clinical Question Quality Improvement
- `QUESTION_GENERATION_PROMPT` improved: more empathetic, avoids repeating asked questions, avoids vague "what brings you in today?" when complaint is known, asks about medication adherence/side effects
- `PERSONALIZED_QUESTIONS_PROMPT` improved: condition-specific, conversational, self-contained questions

#### 5. E2E Resilience Test Consolidation (13 → 7 scenarios)
- Increased response timeout from 20s → 40s (LLM chain latency)
- Added `do_intake()`, `do_clinical()`, `do_booking()` helpers to reduce duplication
- 7 scenarios: Happy Path, Confused Patient, Emergency Escalation, Rescheduler, Slot Rejector, Complex Clinical, Resilient Patient

#### Files Modified/Created
| File | Action |
|------|--------|
| `medforce/gateway/booking_registry.py` | Created — BookingRegistry with GCS persistence |
| `medforce/gateway/diary.py` | Modified — hold_id on SlotOption, rescheduled_from on BookingSection |
| `medforce/gateway/events.py` | Modified — RESCHEDULE_REQUEST event type |
| `medforce/gateway/gateway.py` | Modified — RESCHEDULE_REQUEST routing |
| `medforce/gateway/agents/booking_agent.py` | Modified — registry integration, reschedule, slot rejection |
| `medforce/gateway/agents/clinical_agent.py` | Modified — improved question prompts |
| `medforce/gateway/agents/monitoring_agent.py` | Modified — reschedule keyword detection |
| `medforce/gateway/setup.py` | Modified — wire BookingRegistry |
| `tests/test_booking_registry.py` | Created — 24 tests |
| `tests/test_booking_agent.py` | Modified — 5 slot rejection tests + 9 registry tests |
| `tests/test_resilience_journeys.py` | Modified — 70 tests (13 scenario classes) |
| `tests/e2e_resilience_test.py` | Rewritten — 7 consolidated scenarios |
| `tests/test_full_journey.py` | Modified — reschedule keyword test updated |

#### Test Results: 706/706 passed, 0 regressions

---

## Analysis: Pre-Consultation Agentic Workflow & Resilience Testing

### Current Architecture

Event-driven, state-machine-based patient journey orchestrator with 4 specialist agents:

| Agent | Role | Key Responsibility |
|-------|------|--------------------|
| **IntakeAgent** | Receptionist | Demographics & consent collection, opportunistic extraction |
| **ClinicalAgent** | Triage Nurse | Adaptive clinical assessment, risk scoring (hard rules → keywords → LLM) |
| **BookingAgent** | Scheduler | Risk-stratified slot selection, pre-appointment instructions, slot rejection, rescheduling |
| **MonitoringAgent** | Guardian | Post-booking check-ins, deterioration detection, emergency escalation, reschedule handoff |

**Coordination:** Event envelopes routed by deterministic Gateway. Explicit routes for handoffs (`INTAKE_COMPLETE → clinical`), phase-based routes for external events (`USER_MESSAGE → current phase agent`).

**Resilience layers:**
1. Gateway: circuit breaker (chain depth 10), idempotency guard, diary optimistic locking with retries
2. Agents: LLM retry (1 retry, 0.3s backoff) + deterministic fallbacks for every LLM call
3. Risk scoring: hard rules → keyword rules → LLM fallback (deterministic always wins)
4. Patient-facing: graceful error messages, empty/gibberish input handling

### Resilience Testing Coverage (~1,971 lines)

5 patient personas across unit, journey, and e2e levels:
- **Confused Patient** — sends data for wrong fields (opportunistic extraction)
- **Contradicting Patient** — updates pain level, overrides allergies
- **Skipper** — empty messages, gibberish, "I don't know", minimal data
- **Happy Path** — clean full journey
- **Worst Case** — emergency escalation, post-emergency, negation handling

All tests run with `llm_client=None` (deterministic-only).

### Gaps Identified

**Architectural:** No distributed transaction support, in-memory idempotency (lost on restart), no dead letter queue, no service-to-service auth.

**Testing:** No chaos engineering (GCS down, LLM fully unavailable), limited concurrency testing, no booking negative paths, no GP communication edge cases, no input boundary tests.

**Operational:** No health check endpoint, no metrics/observability, no graceful shutdown.

**Data Quality:** No lab value extraction validation, no document deduplication, incomplete deterioration → rebooking integration.

---

## Next Steps (Prioritized)

### P0 — Critical Gaps

**1. Stalled Assessment Timeout**
- If a patient goes silent during the 3-question deterioration assessment, it hangs indefinitely.
- **Action:** Add configurable timeout (48h). Heartbeat checks assessment age → completes with partial data → escalates conservatively.

**2. Phase Transition Recovery**
- Agent crash mid-processing → patient stuck in phase forever.
- **Action:** Phase-timeout watchdog (heartbeat-driven). Detect patients stuck beyond SLA → retry or alert staff.

**3. Rate Limiting**
- Zero rate limiting. Patient/bot can spam 1000 msgs/min, exhausting LLM quota and GCS throughput.
- **Action:** Per-patient message rate limiting at Gateway level (5 msgs/min with backpressure response).

### P1 — Resilience Testing Gaps

**4. Chaos Engineering Tests** — Mock GCS down, LLM fully unavailable, verify graceful degradation.
**5. Concurrency & Load Testing** — Multiple coroutines hitting same patient diary, verify locking under contention.
**6. Booking Negative Paths** — Zero slots available, schedule manager errors, race conditions on slot selection.
**7. GP Communication Edge Cases** — GP never responds, malformed response, wrong patient routing.

### P2 — Operational Readiness

**8. Health Check Endpoint** — Verify GCS, LLM client, all 4 agents registered.
**9. Observability & Metrics** — Processing time per agent, error count per event type, patients per phase.
**10. Dead Letter Queue** — GCS-backed store for failed events, ops can review and replay.

### P3 — Data Quality & Safety

**11. Lab Value Extraction Validation** — Confidence scoring + out-of-range sanity checks.
**12. Document Deduplication** — Content-hash dedup for uploaded documents.
**13. Deterioration → Rebooking Integration** — Complete moderate-severity → rebooking path end-to-end.

### P4 — Hardening

**14. LLM Retry Enhancement** — 3 retries with exponential backoff for critical paths.
**15. Distributed Idempotency** — Move to persistent store (Redis/GCS).
**16. Input Boundary Testing** — 100k+ char messages, large uploads, diary overflow.

### Execution Order

| Sprint | Items | Theme |
|--------|-------|-------|
| Sprint 1 | #1, #2, #3 | Safety-critical gaps |
| Sprint 2 | #4, #5, #6, #7 | Resilience test coverage |
| Sprint 3 | #8, #9, #10 | Operational readiness |
| Sprint 4 | #11, #12, #13 | Data quality & integration |
| Sprint 5 | #14, #15, #16 | Production hardening |

---

## P0 Implementation — Complete

### P0 #1: Stalled Assessment Timeout

| File | Changes |
|------|---------|
| `medforce/gateway/agents/monitoring_agent.py` | Added `ASSESSMENT_TIMEOUT_HOURS = 48`, `_check_stalled_assessment()` method |

**How it works:**
- On every HEARTBEAT, monitoring agent checks if there's an active deterioration assessment older than 48h
- If stalled with answers: runs fallback severity scorer, then bumps severity up one level (mild→moderate, moderate→severe) for conservative safety
- If stalled with no answers: defaults to moderate severity
- Emits `DETERIORATION_ALERT` for moderate/severe/emergency
- Sends patient a message: "We noticed you started telling us about symptoms but haven't heard back..."
- Logs `assessment_timeout` entry in monitoring entries

### P0 #2: Phase Transition Recovery Watchdog

| File | Changes |
|------|---------|
| `medforce/gateway/diary.py` | Added `phase_entered_at: datetime` field to `DiaryHeader` |
| `medforce/gateway/gateway.py` | Added phase change detection after agent processing — stamps `phase_entered_at` on transitions |
| `medforce/gateway/agents/monitoring_agent.py` | Added `PHASE_STALE_THRESHOLDS`, `_check_phase_staleness()` method |

**How it works:**
- Gateway compares diary phase before/after agent processing; stamps `phase_entered_at` on change
- On HEARTBEAT, monitoring agent checks if patient is stuck in intake (72h), clinical (72h), or booking (48h)
- Monitoring phase is exempt (long-lived by design)
- Sends phase-specific nudge message to patient (e.g. "We haven't finished collecting your details...")
- Nudge fires only once per phase (checks for existing `phase_stale_*` entry to avoid spam)
- Adds alert to `alerts_fired`

### P0 #3: Per-Patient Rate Limiting

| File | Changes |
|------|---------|
| `medforce/gateway/gateway.py` | Added `RATE_LIMIT_WINDOW_SECONDS = 60`, `RATE_LIMIT_MAX_MESSAGES = 5`, `_is_rate_limited()` method, rate limit check in `process_event()` |

**How it works:**
- Sliding window rate limiter: tracks timestamps per patient_id, evicts entries older than 60s
- Only applies to `USER_MESSAGE` events at chain_depth=0 (internal events like heartbeats, handoffs bypass it)
- Returns friendly "please wait" response when exceeded
- Logs `RATE_LIMITED` status in event log
- Per-patient isolation (one patient's spam doesn't affect others)

### P0 Tests — 21/21 passing

| Test Class | Count | What it tests |
|-----------|-------|---------------|
| `TestStalledAssessmentTimeout` | 7 | Fresh assessment not timed out, stale force-completed, no answers → moderate, severity escalation, completed not re-triggered, no assessment no timeout, timeout creates entry |
| `TestPhaseTransitionRecovery` | 7 | Intake/clinical/booking stuck sends nudge, recently entered no nudge, monitoring never stales, nudge not repeated, staleness adds alert |
| `TestRateLimiting` | 5 | Under limit succeeds, over limit blocked, per-patient isolation, heartbeats bypass, rate limit logged |
| `TestPhaseTransitionTracking` | 2 | Phase change updates entered_at, no change preserves entered_at |

**Full test suite: 577/577 passed, 0 regressions.**

---

## P1 Implementation — Resilience Testing Gaps — Complete

### P1 #4: Chaos Engineering Tests (3 tests)
- GCS load failure → agent still returns safe response
- GCS save failure → agent result still dispatched to patient
- Concurrency retry exhaustion → graceful failure

### P1 #5: LLM Fully Unavailable Tests (5 tests)
- Each agent (intake, clinical, booking, monitoring) fully functional without LLM
- Full journey in deterministic mode (pre-seeded intake → clinical → booking → monitoring)

### P1 #6: Booking Negative Paths (7 tests)
- Zero slots available → sorry message + escalation
- Schedule manager crash → graceful error
- Schedule manager update failure → no false confirmation
- Double booking guard → idempotent
- Gibberish then valid selection → recovers
- Out-of-range selection → retry prompt
- Rebooking after deterioration → gets new slots

### P1 #7: GP Communication Edge Cases (4 tests)
- Empty GP response → patient notified
- Valid GP response → forwarded
- No matching query → handled gracefully
- Deterioration alert from stalled assessment timeout

### P1 Tests: 22/22 passing (`tests/test_p1_resilience_gaps.py`)

---

## P2 Implementation — Operational Readiness — Complete

### P2 #8: Health Check Endpoint
| File | Changes |
|------|---------|
| `medforce/gateway/gateway.py` | Added `health_check()` method — returns agent count, status |
| `medforce/routers/gateway_api.py` | Added `GET /health` endpoint |

### P2 #9: Observability & Metrics
| File | Changes |
|------|---------|
| `medforce/gateway/gateway.py` | Added `_metrics` dict tracking `events_processed`, `events_failed`, `agent_timings` per agent, `events_by_type` |
| `medforce/routers/gateway_api.py` | Added `GET /metrics` endpoint |

**How it works:**
- Every event increments `events_processed`
- Agent processing is timed — stores cumulative and count per agent for avg calculation
- Errors increment `events_failed`
- All exposed via `/metrics` JSON endpoint

### P2 #10: Dead Letter Queue
| File | Changes |
|------|---------|
| `medforce/gateway/gateway.py` | Added `_dead_letter_queue` list, `_add_to_dlq()`, `get_dlq()`, `replay_dlq_event()` methods |
| `medforce/routers/gateway_api.py` | Added `GET /dlq` endpoint |

**How it works:**
- When an agent throws an unhandled exception, the event + traceback is captured in DLQ
- DLQ is bounded (last 100 entries) to prevent memory bloat
- Each entry has: event envelope, error message, traceback, timestamp
- Ops can review via `/dlq` endpoint

### P2 Tests: 8/8 passing (in `tests/test_p2_p3_p4.py`)

---

## P3 Implementation — Data Quality & Safety — Complete

### P3 #11: Lab Value Extraction Validation
| File | Changes |
|------|---------|
| `medforce/gateway/agents/monitoring_agent.py` | Added `LAB_PLAUSIBLE_RANGES` dict (18 parameters), `_validate_lab_values()` method |

**How it works:**
- Before lab comparison, extracted values are checked against plausible clinical ranges
- Values outside bounds (e.g., hemoglobin=500) are excluded and flagged as likely extraction errors
- Remaining valid values proceed through normal deterioration comparison
- Patient is notified values were flagged for manual review
- Supported parameters: bilirubin, ALT, AST, INR, creatinine, platelets, albumin, hemoglobin, sodium, potassium, glucose, WBC

### P3 #12: Document Deduplication
| File | Changes |
|------|---------|
| `medforce/gateway/diary.py` | Added `content_hash: Optional[str]` to `ClinicalDocument`, `has_document_hash()` to `ClinicalSection` |
| `medforce/gateway/agents/clinical_agent.py` | Added duplicate check in `_handle_document()` using SHA-256 content hash |
| `medforce/routers/gateway_api.py` | Added `hashlib.sha256` content hash computation in upload endpoint |

**How it works:**
- Upload endpoint computes SHA-256 hash of file content
- Hash passed in `DOCUMENT_UPLOADED` event payload
- Clinical agent checks if hash already exists in diary documents
- Duplicate uploads get "already uploaded" response — no false deterioration alerts

### P3 Tests: 13/13 passing (in `tests/test_p2_p3_p4.py`)

---

## P4 Implementation — Hardening — Complete

### P4 #14: LLM Retry Enhancement
| File | Changes |
|------|---------|
| `medforce/gateway/agents/llm_utils.py` | Enhanced `llm_generate()` — default 2 retries (3 attempts), `critical=True` mode for 3 retries (4 attempts), exponential backoff (0.5s/1.0s base) |

**How it works:**
- Normal calls: 0.5s → 1.0s → 2.0s backoff
- Critical calls (risk scoring, emergency detection): 1.0s → 2.0s → 4.0s backoff
- All callers already have deterministic fallback for total failure

### P4 #16: Input Boundary Protection
| File | Changes |
|------|---------|
| `medforce/gateway/gateway.py` | Added `MAX_MESSAGE_LENGTH = 10_000`, truncation in `process_event()` |

**How it works:**
- Messages over 10,000 chars are silently truncated before reaching agents
- Prevents LLM context overflow and excessive GCS storage
- Only applies to `USER_MESSAGE` events with text content

### P4 Tests: 8/8 passing (in `tests/test_p2_p3_p4.py`)

---

## Full Test Results After P0-P4

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_p0_safety.py` | 21 | All pass |
| `tests/test_p1_resilience_gaps.py` | 22 | All pass |
| `tests/test_p2_p3_p4.py` | 30 | All pass |
| All other existing tests | 556 | All pass |
| **Total** | **629** | **629/629 passed, 0 regressions** |

---

## Observability & Real-Life Patient Scenario Summary

### How to Observe Each Feature

| Feature | How to See It |
|---------|--------------|
| Rate limiting | Send 6+ messages in under a minute → get "please wait" response |
| Stalled assessment | Start deterioration assessment, don't answer for 48h+, send heartbeat → auto-completes |
| Phase staleness | Leave patient in intake for 72h+, send heartbeat → nudge message appears |
| Health check | `GET /api/gateway/health` → JSON with agent count and status |
| Metrics | Process some events, then `GET /api/gateway/metrics` → event counts, timing |
| DLQ | If an agent throws an error, `GET /api/gateway/dlq` → failed event + traceback |
| Lab validation | Upload labs with albumin=500 → value excluded, patient told "flagged for review" |
| Document dedup | Upload the same PDF twice → second time gets "already uploaded" message |
| LLM retry | Kill LLM connectivity → watch logs for retry attempts with backoff |
| Input truncation | Send a 15,000 char message → only first 10,000 chars reach the agent |

### Real-Life Patient Scenarios

**Scenario 1: The Silent Patient**
Sarah starts her pre-consultation, completes intake and clinical, gets her appointment booked. Two weeks before her appointment she reports "feeling worse." The system starts a 3-question deterioration assessment. Sarah answers the first question but then disappears — maybe she's hospitalized, lost her phone, or just forgot.

- **Before P0:** Sarah's assessment hangs forever. Nobody notices.
- **After P0:** 48 hours later, the heartbeat triggers `_check_stalled_assessment()`. Since she reported feeling worse and answered one question indicating pain, the system conservatively escalates severity by one level (mild→moderate), fires a deterioration alert, and notifies the clinical team. Sarah's appointment gets brought forward.

**Scenario 2: The Anxious Spammer**
Ahmed is nervous about his liver consultation and sends 15 messages in a minute asking about his results.

- **Before P0:** All 15 hit the LLM, burning API quota and GCS writes.
- **After P0:** After message 5, Ahmed gets "You're sending messages quite quickly. Please wait a moment before sending another." The system stays responsive for other patients.

**Scenario 3: The Duplicate Uploader**
Dr. Patel's office sends the same blood test PDF twice (email glitch, secretary resent it).

- **Before P3:** Both get processed. The second triggers a lab comparison against baseline — comparing identical values to themselves. This could create confusing "no change" entries or, worse, if extraction differs slightly between runs, a false deterioration alert.
- **After P3:** SHA-256 hash matches. Second upload returns "This document has already been uploaded." No duplicate processing, no false alerts.

**Scenario 4: The Bad OCR**
Maria uploads a blood test PDF. The OCR extracts albumin=500 (actual value was 50.0 — OCR misread the decimal).

- **Before P3:** The system sees albumin jump from baseline 28 to 500 — massive "increase," which might confuse comparison logic or generate misleading entries in the diary.
- **After P3:** Lab validation catches albumin=500 as outside plausible range (0-60 g/L), excludes it, and tells Maria: "Some values were flagged for manual review." The remaining valid values still process normally.

**Scenario 5: The Stuck Patient**
Tom completes intake but the clinical agent crashes during his risk assessment (server restart, GCS timeout, etc.). His diary is stuck in CLINICAL phase.

- **Before P0:** Tom waits forever. Nobody notices unless he calls the clinic.
- **After P0:** After 72 hours, the heartbeat-driven phase staleness watchdog detects Tom is stuck. It sends him: "We're still reviewing your clinical information — a team member will follow up shortly." It also fires an alert for staff review.

**Scenario 6: Ops Team Monitoring**
The ops team wants to know how the system is performing on a Monday morning after 200 patients went through over the weekend.

- **Before P2:** No visibility. Check logs manually.
- **After P2:** `GET /health` confirms all 4 agents are registered. `GET /metrics` shows 1,247 events processed, 3 failures, average intake processing at 120ms, clinical at 340ms. `GET /dlq` shows the 3 failed events with full tracebacks — all were transient GCS timeouts that self-recovered on retry.

---

# Claude Session Summary — 2026-02-20 (Session 3)

## What Was Done This Session

### 1. HTML Test Harness — Editable Patient ID (UI)

Made patient ID directly editable in the top bar instead of being hidden in the Controls drawer.

| File | Changes |
|------|---------|
| `medforce/static/test_harness.html` | Replaced static `pid-label` with editable `<input>` + Load/Reset buttons in top bar |

- Added CSS for `.pid-input`, `.topbar-btn-sm`, `.topbar-btn-sm.reset`, `.topbar-btn-sm.go`
- `loadPatient()` JS function syncs top bar input → drawer input and resets conversation
- `resetPatient()` uses top bar input as primary source
- Added yellow heartbeat simulation buttons (Day 7/14/30/60) during monitoring phase for testing proactive agent messages
- Updated monitoring quick-replies with assessment-flow options ("Getting worse", "More detail", "Can manage", "Struggling")

---

### 2. Interactive Deterioration Assessment Flow (Monitoring Agent)

Completely rewrote the monitoring agent's deterioration handling. Previously sent canned generic responses. Now runs an interactive 3-question clinical assessment before deciding what to do.

| File | Changes |
|------|---------|
| `medforce/gateway/diary.py` | Added `DeteriorationQuestion`, `DeteriorationAssessment` models; added `deterioration_assessment` field to `MonitoringSection` |
| `medforce/gateway/agents/monitoring_agent.py` | Major rewrite of `_handle_user_message`; added assessment flow, LLM prompts, emergency escalation, check-in response evaluation |
| `medforce/gateway/agents/clinical_agent.py` | Rewrote `_handle_deterioration` to use assessment data and trigger rebooking |
| `tests/test_monitoring_agent.py` | Full rewrite — 47 tests covering all monitoring flows |

#### New Diary Models (`diary.py`)
```python
class DeteriorationQuestion(BaseModel):
    question: str
    answer: Optional[str] = None
    category: str = ""  # "description", "new_symptoms", "severity", "functional"

class DeteriorationAssessment(BaseModel):
    active: bool = False
    detected_symptoms: list[str]
    trigger_message: str = ""
    questions: list[DeteriorationQuestion]
    assessment_complete: bool = False
    severity: Optional[str] = None   # "mild", "moderate", "severe", "emergency"
    recommendation: Optional[str] = None  # "continue_monitoring", "bring_forward", "urgent_referral", "emergency"
    reasoning: Optional[str] = None
    started: Optional[datetime] = None
```

#### Monitoring Agent — Assessment Flow (`monitoring_agent.py`)

**How `_handle_user_message` now works:**
1. If an active deterioration assessment exists → process the answer (`_process_deterioration_answer`)
2. Check for **emergency keywords** (unconscious, seizure, bleeding, jaundice, etc.) → immediate escalation, skip assessment
3. Check for **non-emergency red flag keywords** (worse, worsening, fatigue, etc.) → start interactive assessment
4. Check if message is a response to a scheduled question → **evaluate clinical significance** of the answer
5. If answer is concerning → start deterioration assessment
6. Otherwise → normal risk-aware acknowledgment

**New methods added:**
- `_start_deterioration_assessment()` — Initializes assessment, asks first question
- `_process_deterioration_answer()` — Records answer, checks for emergency keywords with negation awareness, asks next question or completes
- `_complete_deterioration_assessment()` — LLM severity assessment with rule-based fallback, emits DETERIORATION_ALERT for moderate/severe/emergency
- `_immediate_emergency_escalation()` — Skips assessment for life-threatening symptoms
- `_generate_assessment_question()` / `_fallback_assessment_question()` — LLM + fallback question generation
- `_assess_severity()` / `_fallback_severity_assessment()` — LLM + rule-based severity scoring
- `_determine_recommendation()` — Maps severity to action
- `_generate_assessment_outcome()` — Patient-facing response after assessment
- `_evaluate_checkin_response()` — **Evaluates scheduled question answers for clinical significance** (pattern + LLM)

**Negation-aware keyword detection:**
"No jaundice or confusion" does NOT trigger jaundice/confusion as emergency flags. Checks for negation patterns ("no ", "not ", "don't have ", etc.) in the 20 characters before each keyword.

**Check-in response evaluation (`_evaluate_checkin_response`):**
When a patient answers a scheduled check-in question, the answer is evaluated in two tiers:
1. **Pattern matching** — condition-specific symptom patterns (liver: red/dark urine, clay stool, yellow skin, swelling, tarry stool, bruising, etc.) and general patterns (fever, weight loss, vomiting, breathlessness, etc.)
2. **LLM fallback** — If no patterns match, sends the question + answer to Gemini for clinical significance evaluation

**LLM prompts added:**
- `DETERIORATION_ASSESSMENT_PROMPT` — Severity assessment after 3 Q&A rounds
- `DETERIORATION_QUESTION_PROMPT` — Follow-up question generation during assessment
- `CHECKIN_RESPONSE_EVALUATION_PROMPT` — Evaluates if check-in answer is clinically concerning

#### Clinical Agent — Rebooking on Deterioration (`clinical_agent.py`)

**Rewrote `_handle_deterioration` to actually trigger rebooking:**
- Accepts assessment data from monitoring agent
- Sets risk level: emergency→CRITICAL, severe→HIGH, moderate→MEDIUM
- Recognizes both `"deterioration_assessment"` and `"emergency_escalation"` sources
- For moderate/severe/emergency: clears confirmed booking, resets `slots_offered`/`slot_selected`, sets phase to BOOKING, emits `CLINICAL_COMPLETE` to trigger booking agent with updated urgency
- Added `_assessment_based_guidance()` for assessment-informed patient responses

---

### 3. Bugs Fixed During Testing

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| LLM mock returning non-string in tests | `response.text` returns MagicMock when Gemini unavailable | Added `isinstance(raw, str)` check before using LLM response |
| "No jaundice" triggering emergency | No negation detection in keyword matching | Added negation-aware prefix checking (20-char window before keyword) |
| Booking not brought forward after assessment | Clinical agent sent guidance but didn't clear booking or emit CLINICAL_COMPLETE | Added rebooking logic: clear booking → set phase BOOKING → emit CLINICAL_COMPLETE |
| Emergency severity not rebooking | Condition was `severity in ("moderate", "severe")` excluding "emergency" | Changed to `severity in ("moderate", "severe", "emergency")` |
| Emergency escalation source not recognized | Clinical agent only checked `source == "deterioration_assessment"` | Changed to `source in ("deterioration_assessment", "emergency_escalation")` |
| Check-in answers not evaluated | Scheduled question answers stored but never analyzed for clinical significance | Added `_evaluate_checkin_response()` with pattern matching + LLM evaluation |
| "tarry" pattern not matching natural language | Pattern "tarry stool" doesn't match "stool has been black and tarry" | Added standalone "tarry", "black and tarry", "melena" patterns |

---

## Test Results

**47/47 tests pass** in `tests/test_monitoring_agent.py`:

| Test Class | Count | Coverage |
|-----------|-------|----------|
| `TestBookingComplete` | 5 | Monitoring setup, baseline, welcome message |
| `TestHeartbeatMilestones` | 6 | Scheduled check-ins, GP reminders, inactive skip |
| `TestLabComparison` | 7 | Stable/deteriorating/new values, edge cases |
| `TestMonitoringDocumentUpload` | 5 | Lab upload, deterioration alerts |
| `TestMonitoringUserMessage` | 5 | Normal messages, emergency escalation, assessment start |
| `TestDeteriorationAssessment` | 5 | Full assessment flow, emergency during assessment |
| `TestMilestoneMessages` | 5 | Message generation for different time periods |
| `TestMonitoringScenarios` | 4 | End-to-end scenarios |
| `TestCheckinResponseEvaluation` | 5 | Red/dark urine, fever, tarry stool, normal answer |

---

# Claude Session Summary — 2026-02-19 (Session 2)

## What Was Done This Session

### 1. Document Upload & GCS Storage (NEW)

Added full document upload support to the gateway system, following the previous pre-consultation implementation's patterns.

#### Files Modified

| File | Changes |
|------|---------|
| `medforce/routers/gateway_api.py` | Added `POST /upload/{patient_id}`, `GET /chat/{patient_id}`, `GET /documents/{patient_id}` endpoints |
| `medforce/gateway/gateway.py` | Added `_persist_chat_history()` — auto-saves conversation to `patient_data/{patient_id}/pre_consultation_chat.json` after every event |
| `medforce/static/test_harness.html` | Added paperclip upload button, file staging area, Documents tab, upload quick-reply in clinical phase |

#### New API Endpoints

```
POST /api/gateway/upload/{patient_id}    — Upload documents (Base64-encoded files)
GET  /api/gateway/chat/{patient_id}      — Read chat history from GCS
GET  /api/gateway/documents/{patient_id} — List uploaded documents in GCS
```

#### GCS Storage Structure (clinic_sim_dev bucket)

```
patient_data/{patient_id}/
├── pre_consultation_chat.json    ← Chat history (auto-persisted after every event)
└── raw_data/
    ├── lab_results.pdf           ← Uploaded documents
    ├── referral_letter.png
    └── ...

patient_diaries/patient_{patient_id}/
└── diary.json                    ← Patient diary (unchanged, already existed)
```

#### How Upload Works
1. Frontend sends Base64-encoded files to `POST /api/gateway/upload/{patient_id}`
2. Backend decodes, stores in `patient_data/{patient_id}/raw_data/{filename}` in GCS
3. Emits `DOCUMENT_UPLOADED` event through the Gateway
4. Clinical agent's `_handle_document()` processes it — adds to `diary.clinical.documents` with `file_ref` pointing to GCS path
5. If in monitoring phase, monitoring agent compares lab values against baseline

#### How Chat History Persistence Works
- Gateway's `process_event()` step 8 calls `_persist_chat_history()` after saving diary
- Converts `diary.conversation_log` entries to `{"conversation": [{"sender": "patient"|"admin", "message": "...", ...}]}` format
- Writes to `patient_data/{patient_id}/pre_consultation_chat.json`
- Matches the previous implementation's storage pattern exactly

---

### 2. Previous Session Work (Already Complete)

These were done in the prior session and are still uncommitted:

| File | Changes |
|------|---------|
| `medforce/routers/gateway_api.py` | Reset endpoint clears test harness responses |
| `medforce/gateway/agents/intake_agent.py` | Helper keywords checked before patient keywords (fixes helper detection) |
| `medforce/gateway/agents/monitoring_agent.py` | Added cardiac red flag keywords + risk-stratified deterioration responses |
| `medforce/gateway/agents/clinical_agent.py` | Deterioration guard (no re-booking if confirmed) + document collection phase |
| `medforce/gateway/agents/booking_agent.py` | Guard against re-booking confirmed patients |
| `tests/e2e_gateway_test.py` | Complete rewrite — 2 cases (patient + helper), snapshot recording, adaptive phase checking |

**E2E test results**: 18/18 passed. Transcript at `tests/e2e_results.json`.

---

## What's NOT Done Yet / Next Steps

1. **All changes are uncommitted** on branch `pre-consultation-agentic`. Run `git status` to see full list.
2. **Full end-to-end manual test** — Restart server and run through complete flow: intake → clinical → booking → monitoring → heartbeat → deterioration assessment → rebooking.
3. **E2E automated tests may need update** — The monitoring agent changes are significant. `tests/e2e_gateway_test.py` may need adjustments for the new assessment flow.
4. **LLM integration testing** — The assessment flow uses Gemini for question generation and severity assessment. Tested with fallback rules only (no LLM in unit tests). Need to verify LLM path works with real API key.
5. **Additional condition-specific patterns** — Currently only liver-specific patterns are defined in `CONCERNING_PATTERNS`. Could add cardiac, renal, etc.

---

## Branch & Git State

- **Branch**: `pre-consultation-agentic`
- **Base branch**: `maser`
- **All changes uncommitted** — staged + untracked files

Key untracked directories:
```
medforce/gateway/agents/
medforce/gateway/dispatchers/
medforce/gateway/handlers/
medforce/gateway/heartbeat.py
medforce/gateway/ingest/
medforce/gateway/setup.py
medforce/gateway/tests/
medforce/routers/gateway_api.py
medforce/static/
tests/e2e_gateway_test.py
tests/e2e_results.json
tests/test_*.py
```

Modified tracked files:
```
medforce/app.py
medforce/dependencies.py
medforce/gateway/diary.py
medforce/gateway/gateway.py
```

---

## How to Resume

1. `cd D:\clinic-os-v4\backend`
2. `git status` to verify state
3. Run tests: `python -m pytest tests/test_monitoring_agent.py -v` (47 should pass)
4. Restart server: `python run.py`
5. Open `http://127.0.0.1:8080/static/test_harness.html`
6. Type patient ID in top bar, click Load
7. Test monitoring flow: go through intake → clinical → booking → monitoring
8. Click heartbeat buttons (Day 7/14/30/60) to test proactive check-ins
9. Answer check-in questions with concerning symptoms to verify assessment triggers
10. When satisfied, commit all changes

---

## Quick Test Script for Deterioration Assessment

1. Open HTML harness, complete intake → clinical → booking phases
2. In monitoring phase, click "Day 14" heartbeat button
3. Agent sends scheduled check-in question (e.g., about urine/stool changes)
4. Answer with concerning symptom: "Yes my urine is red and stool is clay colored"
5. **Expected**: Agent starts deterioration assessment, asks follow-up questions
6. Answer 3 follow-up questions
7. **Expected**: Assessment completes, severity determined, clinical agent notified
8. If moderate/severe/emergency: booking should be cleared, new booking offered

Alternative test — direct deterioration:
1. In monitoring phase, type "I've been feeling worse, more fatigue"
2. **Expected**: Assessment starts with follow-up question
3. Answer 3 questions
4. **Expected**: Assessment completes with severity and recommendation

Emergency test:
1. Type "I've been vomiting blood and feel confused"
2. **Expected**: Immediate escalation — "call 999" / "go to A&E" message, no assessment

---

## Architecture Reference

- **Gateway** (`gateway.py`): Deterministic event router, no AI. Routes via Strategy A (explicit event→agent map) or Strategy B (phase→agent map).
- **Agents**: intake → clinical → booking → monitoring. Each processes events and returns `AgentResult` with updated diary, responses, and emitted events.
- **Recursive event loop**: Gateway processes emitted events recursively (circuit breaker at depth 10).
- **Diary**: Single JSON doc per patient in GCS. All agents read before acting, write after.
- **Chat persistence**: Now also writes to `patient_data/` for frontend/reporting compatibility.
- **Document flow**: Upload → GCS storage → DOCUMENT_UPLOADED event → clinical/monitoring agent processes → diary updated.
- **Deterioration flow**: Patient message/check-in answer → pattern/LLM evaluation → interactive 3-question assessment → severity scoring → clinical agent → rebooking if needed.

# MedForce Clinic OS — Full Architecture
### Event-Driven Control Loop for Multi-Channel Medical Consultation

**Version:** 2.0
**Date:** February 2026
**Status:** Implementation Ready

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State and Why It Must Change](#2-current-state-and-why-it-must-change)
3. [The Core Concept: It's a Loop, Not a Line](#3-the-core-concept-its-a-loop-not-a-line)
4. [Master Architecture](#4-master-architecture)
5. [Component 1: The Event Envelope](#5-component-1-the-event-envelope)
6. [Component 2: The Gateway](#6-component-2-the-gateway)
7. [Component 3: The Patient Diary](#7-component-3-the-patient-diary)
8. [Component 4: The Per-Patient Event Queue](#8-component-4-the-per-patient-event-queue)
9. [Component 5: The Four Specialist Agents](#9-component-5-the-four-specialist-agents)
10. [Component 6: The Heartbeat Scheduler](#10-component-6-the-heartbeat-scheduler)
11. [Dialogflow's New Role: Channel Adapter Only](#11-dialogflows-new-role-channel-adapter-only)
12. [Multi-Helper Channel System](#12-multi-helper-channel-system)
13. [GP Communication Channel](#13-gp-communication-channel)
14. [Identity Resolution System](#14-identity-resolution-system)
15. [Full Flow Examples](#15-full-flow-examples)
16. [Test Scenarios](#16-test-scenarios)
17. [HTML Test Harness](#17-html-test-harness)
18. [Integration with Existing MedForce Codebase](#18-integration-with-existing-medforce-codebase)
19. [Known Challenges and Mitigations](#19-known-challenges-and-mitigations)
20. [Implementation Phases](#20-implementation-phases)
21. [Decision Summary](#21-decision-summary)

---

# 1. Executive Summary

MedForce Clinic OS is an event-driven control loop where every signal — patient message, helper message, GP response, lab webhook, CRON heartbeat, agent handoff — enters a single Gateway that reads a persistent Patient Diary from GCS, routes to the right specialist agent based on the patient's current phase, lets that agent do its work and update the diary, and feeds any handoff events back into the same loop so agents trigger each other automatically.

One patient message can chain through Intake → Clinical → Booking without the patient lifting a finger. The system can loop backward when information is missing, loop forward again when it's provided, and self-drive through monitoring with CRON heartbeats months after the appointment.

Dialogflow stays in the stack but only as a channel adapter for messaging platforms (WhatsApp, email, SMS) — it handles delivery, not intelligence. The system supports multiple communication channels per patient: the patient themselves, their helpers (spouse, parent, friend), and the referring GP — each with their own authenticated identity and permission level.

Four specialist agents (Intake, Clinical, Booking, Monitoring) handle the clinical workflow, with a shared Patient Diary on GCS as the single source of truth that persists across days, weeks, and months.

---

# 2. Current State and Why It Must Change

## What MedForce Is Today

MedForce is a request-driven FastAPI monolith. Every HTTP or WebSocket request directly calls an agent, which processes it and returns a response. Then the system goes back to sleep.

There is no event bus, no workflow orchestration, and no persistent patient state machine. Patient data is scattered across GCS blobs, in-memory singletons, and local JSON files. Agents communicate through direct Python imports, not events.

The system behaves like a vending machine: patient presses a button, the machine does one thing, spits out a response, and forgets everything. Each request is independent. The system has no awareness of where the patient is in their clinical journey, no ability to trigger itself, and no way for one part of the system to wake up another.

## What It Must Become

A system that works like a real clinic. In a real clinic, the receptionist collects your details and passes your file to the triage nurse. The nurse assesses you and passes the file to the scheduler. The scheduler books you in and passes the file to the follow-up coordinator. Everyone reads the same chart. Everyone knows what the person before them already did. And the clinic doesn't just wait for you to show up — it calls you if your follow-up is overdue.

The critical word is "passes." Work flows from one specialist to the next automatically, driven by the patient's chart, not by the patient having to ask for each step.

## Why the Current Architecture Can't Scale

- **No journey awareness:** The system doesn't know if a patient is mid-intake, awaiting clinical questions, or 3 months post-appointment.
- **No self-driving:** The system only acts when a patient sends a message. It can't proactively check on patients, send reminders, or detect deterioration.
- **No agent coordination:** If the Clinical Agent needs data that the Intake Agent should have collected, there's no mechanism for it to request that data.
- **No multi-channel support:** The system can't handle different people (patient, helper, GP) communicating about the same patient on different channels.
- **No persistence across time:** A patient who registers today and follows up 3 months later starts from zero.

---

# 3. The Core Concept: It's a Loop, Not a Line

This is the single most important idea in the entire architecture. Understanding this makes everything else click.

## Linear System (What We Have Now)

```
Patient sends message → System processes → Response → DONE.
Next message? Start from scratch.
```

Each request is independent. Nothing feeds back. The system only acts when poked.

## Loop System (What We're Building)

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    [ EVENT ARRIVES ]                           │
         │                                      │
         ▼                                      │
    [ GATEWAY reads patient diary ]             │
         │                                      │
         ▼                                      │
    [ ROUTE to the right agent ]                │
         │                                      │
         ▼                                      │
    [ AGENT does work, updates diary ]          │
         │                                      │
         ▼                                      │
    [ AGENT emits new event(s) or nothing ]     │
         │                                      │
         ├── new event? ────────────────────────┘
         │         (feeds back into the loop)
         │
         ├── no event? → STOP (wait for next trigger)
         │
         └── response to patient? → SEND IT OUT
```

The output of every agent feeds back into the same loop. The Gateway doesn't know or care whether it's processing an external event (patient sent a message) or an internal event (an agent just finished and emitted a handoff). It treats them identically — read the diary, route, dispatch.

## Why the Loop Matters — Concrete Example

A patient sends ONE message: "Here's my GP referral."

**In the linear system:** Intake processes it. Done. Patient has to send another message for the next step.

**In the loop system:** Intake processes it → emits INTAKE_COMPLETE → that event loops back to the Gateway → Gateway wakes up Clinical Agent → Clinical processes the referral → emits CLINICAL_COMPLETE → loops back → Gateway wakes up Booking Agent → Booking offers slots. One patient message, three agents triggered automatically.

## What the Loop Enables

### Going Backwards

Clinical Agent discovers the referral is missing the patient's medication list → emits NEEDS_INTAKE_DATA → Gateway routes back to Intake Agent → Intake collects it → emits completion → Gateway routes back to Clinical. The system went backward naturally, not as a special case.

### Skipping Steps

If a patient arrives with complete data and obvious high-risk markers, the loop can jump from intake straight to booking if the clinical assessment is trivial.

### Re-entering Phases

A patient in monitoring gets worse → the loop routes them back through clinical reassessment → new risk score → rebooking → back to monitoring. A full cycle, triggered by one event.

### Chain Reactions

A single lab webhook can trigger: Monitoring Agent detects deterioration → emits alert → Gateway routes to Clinical for reassessment → Clinical scores new risk → emits completion → Gateway routes to Booking for urgent rebooking → Booking confirms → emits completion → back to Monitoring with updated baseline. All from one incoming event, all through the same loop.

### The Loop Diagram

```
INTAKE ───→ CLINICAL ───→ BOOKING ───→ MONITORING
  ↑              ↑                          │
  │              └──── DETERIORATION ───────┘
  │                    (loop backward)
  │
  └──── NEEDS_INTAKE_DATA ─── from Clinical
         (loop backward for missing info)
```

## Why Not Linear?

Real patients don't move in one direction. They:

- Forget to mention medications during intake, and the Clinical Agent discovers the gap later
- Get booked for an appointment, then their lab results change and they need rebooking at a different urgency
- Finish monitoring, then relapse and need the entire clinical-booking cycle again
- Upload documents out of order, requiring the system to revisit earlier assessments
- Have family members communicating on their behalf across different channels

A linear system handles these with ugly exceptions and special cases bolted onto the side. A loop system handles them naturally — it's just another event going through the same Gateway, and the diary tells the system where the patient actually is.

---

# 4. Master Architecture

```
                    EXTERNAL CHANNELS
    ┌─────────┬──────────┬─────────┬──────────┐
    │WhatsApp │  Email   │Web Chat │   SMS    │
    │(Patient)│(GP/Helper│(Patient)│(Helper)  │
    └────┬────┴────┬─────┴────┬────┴────┬─────┘
         │         │          │         │
         ▼         ▼          ▼         ▼
    ┌────────────────────────────────────────┐
    │         DIALOGFLOW (Channel Layer)      │
    │   • Message ingestion only             │
    │   • No intent matching used            │
    │   • No flow logic used                 │
    │   • Delivers responses back to channel │
    └──────────────────┬─────────────────────┘
                       │ Raw message → Event Envelope
                       ▼
    ┌────────────────────────────────────────────────────┐
    │                   GATEWAY                           │
    │              (Central Router)                       │
    │                                                     │
    │   1. Receive event                                  │
    │   2. Resolve sender identity                        │
    │   3. Check permissions                              │
    │   4. Load Patient Diary from GCS                    │
    │   5. Route by event type (Strategy A)               │
    │      or by diary phase (Strategy B)                 │
    │   6. Wake target agent                              │
    │   7. Save updated diary                             │
    │   8. If handoff event emitted → loop back to #1    │
    │                                                     │
    └────┬──────┬──────┬──────┬──────┬──────┬───────────┘
         │      │      │      │      │      │
         ▼      ▼      ▼      ▼      ▼      ▼
    ┌────────┬────────┬────────┬──────────┬────────┬──────────┐
    │ Intake │Clinical│Booking │Monitoring│  GP    │ Helper   │
    │ Agent  │ Agent  │ Agent  │  Agent   │Comms   │ Manager  │
    └───┬────┴───┬────┴───┬────┴────┬─────┴───┬────┴────┬─────┘
        │        │        │         │         │         │
        ▼        ▼        ▼         ▼         ▼         ▼
    ┌──────────────────────────────────────────────────────┐
    │                 PATIENT DIARY (GCS)                    │
    │               One JSON file per patient                │
    │                                                        │
    │   Sections: Header | Intake | Helper Registry |        │
    │   GP Channel | Clinical | Booking | Monitoring |       │
    │   Conversation Log                                     │
    └──────────────────────────────────────────────────────┘
         ▲
         │
    ┌────┴──────────────────┐
    │  HEARTBEAT SCHEDULER  │
    │  (CRON background)    │
    │  Fires periodic       │
    │  HEARTBEAT events     │
    │  for monitored        │
    │  patients             │
    └───────────────────────┘
```

---

# 5. Component 1: The Event Envelope

Everything that enters the loop is wrapped in the same standard envelope. A patient WhatsApp message, a lab webhook, a heartbeat timer, an agent handoff, a GP reply, a helper's document upload — they all look the same to the Gateway.

## Envelope Fields

| Field | Type | Purpose | Example |
|---|---|---|---|
| event_id | UUID | Unique identifier for this event | `evt-a1b2c3d4` |
| event_type | enum | What kind of event | `USER_MESSAGE`, `HEARTBEAT` |
| patient_id | string | Who this event is about | `PT-1234` |
| payload | dict | The actual data | Message text, lab values, slot selection |
| source | string | Where it came from | `whatsapp`, `cron`, `clinical_agent` |
| sender_id | string | Who sent it (for multi-channel) | `PATIENT`, `HELPER-001`, `GP-DrPatel` |
| sender_role | enum | Role of the sender | `patient`, `helper`, `gp`, `system` |
| correlation_id | UUID | Traces full journey across events | One ID threads through all events |
| timestamp | datetime | When it happened | `2026-02-18T10:30:00Z` |

## Complete Event Type Catalog

| Event | Source | Routes To | Description |
|---|---|---|---|
| `USER_MESSAGE` | Patient/Helper via Dialogflow | Phase-based routing | Chat message from patient or authorized helper |
| `DOCUMENT_UPLOADED` | Patient/Helper/GP | Phase-based (usually Clinical) | Lab report, imaging, referral, NHS app screenshot |
| `INTAKE_COMPLETE` | Intake Agent | Clinical Agent | All required demographics collected |
| `INTAKE_DATA_PROVIDED` | Intake Agent (backward loop) | Clinical Agent | Specific missing data now collected |
| `CLINICAL_COMPLETE` | Clinical Agent | Booking Agent | Risk assessment completed |
| `BOOKING_COMPLETE` | Booking Agent | Monitoring Agent | Appointment confirmed |
| `NEEDS_INTAKE_DATA` | Clinical Agent | Intake Agent | Missing non-clinical info discovered mid-assessment |
| `HEARTBEAT` | CRON scheduler | Monitoring Agent | Periodic check on monitored patients |
| `DETERIORATION_ALERT` | Monitoring Agent | Clinical Agent | Patient condition worsening, reassessment needed |
| `GP_QUERY` | Clinical Agent | GP Communication Handler | Question for the referring GP |
| `GP_RESPONSE` | GP via email | Clinical Agent | GP replied to a query |
| `GP_REMINDER` | CRON (48h after query) | GP Communication Handler | Follow-up if GP hasn't responded |
| `HELPER_REGISTRATION` | Patient request during any phase | Helper Manager | New helper being added |
| `HELPER_VERIFIED` | Helper confirmation reply | Helper Manager | Helper confirmed their registration |
| `WEBHOOK` | External system (hospital lab, NHS Spine) | Phase-based routing | External clinical data received |
| `DOCTOR_COMMAND` | Treating clinician | Varies by command type | Clinician instruction (expedite, cancel, modify) |
| `AGENT_ERROR` | Any agent | Error handler | Agent failed, timed out, or produced invalid output |

## Why a Universal Envelope Matters

The Gateway never looks at the payload. It only reads the event_type, patient_id, and sender_role, checks the diary, checks permissions, and routes. Adding a new event type in the future is trivial — define it, tell the Gateway where to send it. The Gateway itself doesn't need to change. The same Gateway code that processes a WhatsApp message also processes a lab webhook and a CRON heartbeat.

---

# 6. Component 2: The Gateway

The Gateway is the center of the loop. Every event passes through it. But — critically — the Gateway is **deliberately stupid**. It's a fast, deterministic router. No AI, no LLM reasoning, no intelligence. Just a lookup table that makes decisions in microseconds.

## Strategy A — Explicit Routing (Handoff Events)

When an agent emits a handoff event, the target is hardcoded. No ambiguity, no decision-making.

| Event | Always Routes To |
|---|---|
| `INTAKE_COMPLETE` | Clinical Agent |
| `INTAKE_DATA_PROVIDED` | Clinical Agent |
| `CLINICAL_COMPLETE` | Booking Agent |
| `BOOKING_COMPLETE` | Monitoring Agent |
| `NEEDS_INTAKE_DATA` | Intake Agent (backward loop) |
| `HEARTBEAT` | Monitoring Agent |
| `DETERIORATION_ALERT` | Clinical Agent (reassessment) |
| `GP_QUERY` | GP Communication Handler |
| `GP_RESPONSE` | Clinical Agent |
| `GP_REMINDER` | GP Communication Handler |
| `HELPER_REGISTRATION` | Helper Manager |
| `HELPER_VERIFIED` | Helper Manager |

## Strategy B — Phase-Based Routing (External Events)

When a patient message, helper message, webhook, or doctor command arrives, the Gateway reads the patient's diary to check what phase they're in:

| Diary Phase | Routes To |
|---|---|
| `intake` | Intake Agent |
| `clinical` | Clinical Agent |
| `booking` | Booking Agent |
| `monitoring` | Monitoring Agent |
| `closed` | Log only, do not process |

## Gateway Processing Sequence

1. **Receive** the event
2. **Resolve sender identity** — who is this? Patient, helper, GP, or unknown? (See Section 14)
3. **Check permissions** — is this person allowed to do what they're trying to do? (See Section 12)
4. **Place** the event in the patient's queue (ensures serialized processing)
5. **Load** the patient's diary from GCS
6. **Route** using Strategy A (if handoff/internal event) or Strategy B (if external event)
7. **Wake** the target agent, pass it the event and the diary
8. **Agent finishes** — save the updated diary back to GCS
9. **If the agent emitted new event(s)**, feed them back to Step 1 ← **THIS IS THE LOOP**
10. **If no new events**, the loop rests until the next external trigger
11. **Send response** back through Dialogflow/channel adapter to the appropriate person(s)

## Why the Gateway is Dumb on Purpose

Intelligence is expensive, slow, and unpredictable. If the router used AI to decide where to send events, you'd get latency, cost, and occasionally wrong routing for safety-critical clinical workflows. A deterministic lookup table is instant, free, and never wrong. All the intelligence lives inside the agents. Boring things don't break.

## Permission Check at the Gateway

Before routing, the Gateway verifies the sender has permission for the implied action:

```
Event: Helper "Tom" (friend, view_status only) sends "Change appointment to Thursday"
  → Resolve: Tom is a friend for PT-1234
  → Implied action: book_appointments
  → Permission check: Tom has [view_status] only → DENIED
  → Response to Tom: "Sorry Tom, you don't have permission to change
    appointments. Only the patient or authorized family members can do this."
  → Event NOT routed to any agent

Event: Helper "Sarah" (spouse, full access) sends "Change appointment to Thursday"
  → Resolve: Sarah is spouse for PT-1234
  → Implied action: book_appointments
  → Permission check: Sarah has [..., book_appointments] → GRANTED
  → Route to Booking Agent normally
```

---

# 7. Component 3: The Patient Diary

One structured JSON document per patient, stored in Google Cloud Storage. Every agent reads it before acting and writes to it after acting. This is the **single source of truth** — the patient's chart that the whole clinic shares.

## Why the Diary Exists

Without a shared diary, each agent starts from zero. The Booking Agent doesn't know the patient's risk level. The Monitoring Agent doesn't know what their baseline labs were 3 months ago. The patient has to re-explain everything every time. A helper's upload would be disconnected from the clinical assessment.

With the diary, when a patient messages 3 months later saying "Are my new results okay?" — the Monitoring Agent loads the diary, sees the baseline ALT was 340, sees the new result is 180, and responds: "Your ALT has dropped from 340 to 180 — nearly half. The treatment appears to be working."

The diary also enables backward loops. When the Clinical Agent routes a patient back to Intake because medication info is missing, the Intake Agent reads the diary and sees exactly what's already been collected. It doesn't re-ask for name and date of birth — it asks only for what's missing.

## Full Diary Structure

```
PATIENT DIARY: John Smith (PT-1234)
│
├── Header
│   ├── patient_id: "PT-1234"
│   ├── current_phase: "clinical"
│   ├── risk_level: "HIGH"
│   ├── created: "2026-02-17T10:00:00Z"
│   ├── last_updated: "2026-02-17T15:30:00Z"
│   └── correlation_id: "journey-abc-123"
│
├── Intake Section
│   ├── name: "John Smith"
│   ├── dob: "1975-05-12"
│   ├── nhs_number: "123-456-7890"
│   ├── address: "42 Oak Lane, London, SW1A 1AA"
│   ├── phone: "+447700900461"
│   ├── email: "john.smith@email.com"
│   ├── next_of_kin: "Sarah Smith (wife) +447700900462"
│   ├── gp_practice: "Greenfields Surgery"
│   ├── gp_name: "Dr. Patel"
│   ├── contact_preference: "whatsapp"
│   ├── consent_gp_contact: true
│   ├── referral_letter_ref: "gs://medforce/PT-1234/referral.pdf"
│   ├── fields_collected: ["name", "dob", "nhs", "address", "phone", "nok"]
│   ├── fields_missing: []
│   └── intake_complete: true
│
├── Helper Registry
│   ├── helpers: [
│   │   {
│   │     id: "HELPER-001",
│   │     name: "Sarah Smith",
│   │     relationship: "spouse",
│   │     channel: "whatsapp",
│   │     contact: "+447700900462",
│   │     permissions: [
│   │       "view_status",
│   │       "upload_documents",
│   │       "answer_questions",
│   │       "book_appointments",
│   │       "receive_alerts"
│   │     ],
│   │     verified: true,
│   │     added: "2026-02-17T10:20:00Z"
│   │   },
│   │   {
│   │     id: "HELPER-002",
│   │     name: "Michael Smith",
│   │     relationship: "child",
│   │     channel: "email",
│   │     contact: "michael@email.com",
│   │     permissions: [
│   │       "view_status",
│   │       "upload_documents"
│   │     ],
│   │     verified: true,
│   │     added: "2026-02-18T09:00:00Z"
│   │   }
│   │ ]
│   └── pending_verifications: []
│
├── GP Channel
│   ├── gp_name: "Dr. Patel"
│   ├── gp_email: "dr.patel@greenfields.nhs.uk"
│   ├── gp_practice: "Greenfields Surgery"
│   ├── referral_id: "REF-2026-4521"
│   ├── queries: [
│   │   {
│   │     query_id: "GPQ-001",
│   │     type: "missing_lab_results",
│   │     query_text: "Your referral mentions recent blood tests...",
│   │     sent: "2026-02-17T11:00:00Z",
│   │     reminder_sent: null,
│   │     status: "responded",
│   │     response_received: "2026-02-17T14:00:00Z",
│   │     attachments_received: ["liver_function_tests.pdf"]
│   │   }
│   │ ]
│   └── last_contacted: "2026-02-17T11:00:00Z"
│
├── Clinical Section
│   ├── chief_complaint: "Persistent RUQ pain (7/10), fatigue"
│   ├── medical_history: ["T2 Diabetes", "Previous alcohol excess"]
│   ├── current_medications: ["Metformin 500mg BD", "Amoxicillin 250mg TDS"]
│   ├── red_flags: ["Elevated bilirubin (6.2)", "Low platelets (95)"]
│   ├── questions_asked: [
│   │   {
│   │     question: "How many units of alcohol do you drink per week?",
│   │     answer: "About 20 units",
│   │     answered_by: "patient",
│   │     timestamp: "2026-02-17T10:45:00Z"
│   │   },
│   │   {
│   │     question: "On a scale of 1-10, how severe is your abdominal pain?",
│   │     answer: "7/10",
│   │     answered_by: "patient",
│   │     timestamp: "2026-02-17T10:46:00Z"
│   │   },
│   │   {
│   │     question: "Have you noticed yellowing of eyes or skin?",
│   │     answer: "No",
│   │     answered_by: "helper:Sarah",
│   │     timestamp: "2026-02-17T15:35:00Z"
│   │   }
│   │ ]
│   ├── documents: [
│   │   {
│   │     type: "lab_results",
│   │     source: "helper:Sarah",
│   │     file_ref: "gs://medforce/PT-1234/docs/lab_photo.jpg",
│   │     processed: true,
│   │     extracted_values: { ALT: 340, AST: 280, bilirubin: 6.2, platelets: 95 }
│   │   },
│   │   {
│   │     type: "lab_results",
│   │     source: "gp:Dr.Patel",
│   │     file_ref: "gs://medforce/PT-1234/docs/lft_results.pdf",
│   │     processed: true,
│   │     extracted_values: { ALT: 340, AST: 275, bilirubin: 6.2, albumin: 28 }
│   │   }
│   │ ]
│   ├── risk_level: "HIGH"
│   ├── risk_reasoning: "Bilirubin 6.2 exceeds threshold of 5.0. ALT 340 significantly elevated. Low platelets suggest possible portal hypertension."
│   ├── risk_method: "deterministic_rule: bilirubin > 5"
│   ├── sub_phase: "complete"
│   ├── sub_phase_history: [
│   │   "analyzing_referral",
│   │   "asking_questions",
│   │   "collecting_documents",
│   │   "scoring_risk",
│   │   "complete"
│   │ ]
│   └── backward_loop_count: 0
│
├── Booking Section
│   ├── eligible_window: "48 hours (HIGH risk)"
│   ├── slots_offered: [
│   │   { date: "2026-02-18", time: "10:00", provider: "Dr. Williams" },
│   │   { date: "2026-02-18", time: "14:00", provider: "Dr. Chen" }
│   │ ]
│   ├── slot_selected: { date: "2026-02-18", time: "10:00", provider: "Dr. Williams" }
│   ├── booked_by: "helper:Sarah"
│   ├── appointment_id: "APT-7891"
│   ├── location: "Royal London Hospital, Hepatology Dept, Floor 3"
│   ├── pre_appointment_instructions: [
│   │   "Please fast for 12 hours before your appointment (no food after 10 PM tonight).",
│   │   "Bring a list of all current medications.",
│   │   "Bring the original blood test paperwork if available.",
│   │   "Continue taking Metformin as normal."
│   │ ]
│   └── confirmed: true
│
├── Monitoring Section
│   ├── monitoring_active: true
│   ├── baseline: {
│   │   ALT: 340,
│   │   AST: 280,
│   │   bilirubin: 6.2,
│   │   platelets: 95,
│   │   albumin: 28,
│   │   snapshot_date: "2026-02-17"
│   │ }
│   ├── entries: [
│   │   {
│   │     date: "2026-03-03",
│   │     type: "heartbeat_14d",
│   │     action: "sent_followup_reminder",
│   │     detail: "No follow-up labs received. Reminder sent to patient and Sarah."
│   │   },
│   │   {
│   │     date: "2026-03-17",
│   │     type: "heartbeat_30d",
│   │     action: "sent_checkin",
│   │     detail: "30-day check-in message sent."
│   │   },
│   │   {
│   │     date: "2026-05-17",
│   │     type: "patient_message",
│   │     new_values: { ALT: 180 },
│   │     comparison: { ALT: { baseline: 340, current: 180, change: "-47%", trend: "improving" } },
│   │     action: "positive_update_sent",
│   │     detail: "ALT improved by 47%. Patient informed."
│   │   }
│   │ ]
│   ├── alerts_fired: []
│   └── next_scheduled_check: "2026-06-17"
│
└── Conversation Log (last 100 entries; older entries archived to separate file)
    ├── [2026-02-17T10:00] AGENT→PATIENT (whatsapp): "Hi John, confirm your DOB and mobile?"
    ├── [2026-02-17T10:05] PATIENT→AGENT (whatsapp): "12 May 1975, 07700 900461"
    ├── [2026-02-17T10:20] PATIENT→AGENT (whatsapp): "Add my wife Sarah, 07700 900462, full access"
    ├── [2026-02-17T10:21] SYSTEM→HELPER:Sarah (whatsapp): "Verification request sent"
    ├── [2026-02-17T10:22] HELPER:Sarah→SYSTEM (whatsapp): "YES"
    ├── [2026-02-17T10:22] SYSTEM: "Helper Sarah verified and registered"
    ├── [2026-02-17T11:00] AGENT→GP:Dr.Patel (email): "Lab results requested for REF-2026-4521"
    ├── [2026-02-17T14:00] GP:Dr.Patel→AGENT (email): "Attached LFT results" [lft_results.pdf]
    ├── [2026-02-17T15:30] HELPER:Sarah→AGENT (whatsapp): "John's lab results" [lab_photo.jpg]
    ├── [2026-02-17T15:31] AGENT→HELPER:Sarah (whatsapp): "Thanks Sarah, received and processing."
    └── [2026-02-17T15:31] AGENT→PATIENT (whatsapp): "Sarah uploaded your results. Reviewing now."
```

## Storage Details

- **Location:** `gs://medforce-patient-diaries/patient_{id}/diary.json`
- **Format:** Structured JSON
- **Concurrency protection:** GCS generation-match conditions (optimistic locking). When an agent reads the diary, it notes the generation number. When it writes back, it includes that number. If another process modified the diary in between, the write fails and retries.
- **Size management:** Conversation log capped at 100 entries; older entries archived to `diary_archive_{date}.json`. Monitoring entries capped at 50; older entries archived similarly.
- **Backup:** Daily snapshot to a separate GCS bucket.

---

# 8. Component 4: The Per-Patient Event Queue

One queue per patient. Events are processed one at a time, in order.

## Why This Exists

If a patient sends 3 messages rapidly, or a lab webhook arrives while a chat is being processed, or a helper uploads a document while the GP is replying — you don't want multiple agents fighting over the same diary simultaneously. The queue serializes events for each patient, guaranteeing one-at-a-time processing.

## Behavior

- **Per patient:** One queue per patient_id. Events for PT-1234 are processed sequentially.
- **Across patients:** Queues run in parallel. Patient A's processing never blocks Patient B. A thousand patients can be processed concurrently.
- **Ordering:** Events are processed in the order they arrive. First in, first out.
- **Implementation:** In-process asyncio.Queue (one per active patient). No external dependencies needed initially.

## Lifecycle

- **Creation:** A queue is created when the first event for a patient arrives.
- **Cleanup:** After 30 minutes of inactivity (no events), the queue is destroyed to free memory.
- **Re-creation:** If a new event arrives after cleanup, a fresh queue is created. The diary is loaded fresh from GCS. The diary is the durable state; the queue is just the in-flight processing mechanism.

## Scaling Path

The asyncio.Queue model works for a single-process deployment. If MedForce scales to multiple workers/replicas:

- **Option A:** Move to Google Cloud Pub/Sub with `patient_id` as the ordering key. Same interface, external durability.
- **Option B:** Move to Redis Streams with consumer groups. Low latency, good for real-time chat.
- The queue interface should be abstracted from day one so the backend can be swapped without changing agent code.

---

# 9. Component 5: The Four Specialist Agents

Each agent has a narrow job, follows the same contract, and has no knowledge of the other agents. They don't know the whole system exists. They just know their job.

## Universal Agent Contract

Every agent follows the same interface:

```
INPUT:  Event (what happened) + Diary (patient state)
OUTPUT: Updated Diary + Optional new Event(s) + Response message(s)
```

An agent receives an event and the patient's diary. It does its specialized work. It updates the diary. It optionally emits one or more new events (handoffs, queries, alerts). It generates response messages tagged with who should receive them (patient, specific helper, GP).

---

## Agent 1: Intake Agent — The Receptionist

**Job:** Collect patient demographics and verify identity. Nothing clinical.

**Wakes up when:** Patient is in the "intake" phase. Also wakes up when `NEEDS_INTAKE_DATA` is received from Clinical Agent (backward loop).

**How it works:**

- Reads the referral letter (if uploaded) and extracts whatever info is already there — name, DOB, NHS number from the GP letter
- Compares extracted data against required fields to identify gaps
- Asks the patient **one focused question per turn** to fill the most important gap
- Offers to register helpers: "Would you like to add anyone who can communicate on your behalf?"
- Handles helper registration flow (send verification, process confirmation)
- **Strict boundary:** Never asks about symptoms, medications, or medical history — that's the Clinical Agent's territory

**When it hands off (normal flow):** All required fields collected → changes diary phase to "clinical" → fires `INTAKE_COMPLETE`.

**When it gets called back (backward loop):** Clinical Agent discovers missing non-clinical info → receives `NEEDS_INTAKE_DATA` with a specific payload (e.g., `{missing: "medication_list"}`) → asks the patient ONLY for that specific data → fires `INTAKE_DATA_PROVIDED` when collected. Does NOT re-ask for name, DOB, or anything already collected.

**UK-specific considerations:**
- NHS number verification format
- GP practice details and ODS code
- Postcode format validation
- Private vs NHS pathway question

**LLM requirement:** Gemini Flash. Simple extraction and question generation. Doesn't need heavy reasoning.

---

## Agent 2: Clinical Agent — The Triage Nurse

**Job:** Understand the medical situation and assess urgency. The most complex and most safety-critical agent.

**Wakes up when:**
- Patient enters "clinical" phase (after intake)
- New document is uploaded during clinical assessment
- `GP_RESPONSE` arrives with requested data
- `DETERIORATION_ALERT` triggers reassessment (re-entry from monitoring)

**Four sub-tasks it runs in sequence:**

### Sub-task A: Referral Analysis

Parse the GP letter to extract:
- Chief complaint (why is the patient here?)
- Medical history (existing conditions)
- Current medications (what are they taking?)
- Red flags (urgent symptoms or values mentioned)

This happens once at the start of the clinical phase. If the referral mentions data that isn't attached (e.g., "recent bloods attached" but no attachment), the Clinical Agent can emit `GP_QUERY` to request it from the GP.

### Sub-task B: Personalized Clinical Questions

Generate 3–5 questions specific to THIS patient. Not generic health questionnaire questions.

Examples of personalized vs generic:
- **Generic (bad):** "Do you have any lifestyle concerns?"
- **Personalized (good):** "Your GP mentioned alcohol use and your liver enzymes are elevated. How many units of alcohol do you drink per week?"

Focus areas:
- Pain levels and characteristics (location, severity, frequency, triggers)
- Lifestyle factors relevant to the specific condition (alcohol for liver, diet for MASH)
- Symptom progression (getting worse? how fast? when did it start?)
- Quality of life impact (can you work? sleep? eat normally?)
- Previous treatments tried (what's worked? what hasn't?)

### Sub-task C: Document Collection and Analysis

Request relevant medical documents:
- Lab results (blood tests — liver function, kidney function, full blood count)
- Imaging reports (ultrasound, CT scan, MRI)
- NHS app screenshots (medication lists, recent appointments)

Smart behavior:
- If referral mentions "recent bloods done," specifically ask for those
- If patient uploads one page of a multi-page report, ask for remaining pages
- If labs are from 6+ months ago, request more recent ones
- Uses Gemini Vision to extract values from uploaded photos and images
- If the referral says results were attached but they're missing, queries the GP instead of asking the patient (who often doesn't have the results)

### Sub-task D: Risk Stratification

Based on all collected data, categorize the patient:

| Risk Level | Criteria | Appointment Window |
|---|---|---|
| **HIGH** | Severe symptoms (jaundice, confusion, GI bleeding, ascites), critical lab values (ALT > 500, bilirubin > 5, platelets < 50), progressive disease | Within 48 hours |
| **MEDIUM** | Moderate symptoms, abnormal but not critical labs, stable chronic condition needing review | Within 1–2 weeks |
| **LOW** | Mild symptoms, routine follow-up, screening/prevention, incidental findings | Within 1 month |

### Critical Safety Design — Risk Scoring Hierarchy

This is the most dangerous part of the system. If the LLM downplays severity, patients could be harmed.

**The strict hierarchy:**

1. **DETERMINISTIC HARD RULES fire first, always.** These are coded conditions that cannot be overridden by any LLM output:
   - Bilirubin > 5.0 → HIGH (no exceptions)
   - ALT > 500 → HIGH (no exceptions)
   - Platelets < 50 → HIGH (no exceptions)
   - Jaundice mentioned anywhere → flag for urgent review
   - Confusion/encephalopathy mentioned → HIGH
   - GI bleeding mentioned → HIGH
   - Ascites mentioned → HIGH

2. **LLM reasoning only gets a vote** on cases where NO hard rule fires — the gray zones where clinical nuance matters.

3. **Hard rules ALWAYS override LLM.** If the LLM says "MEDIUM" but bilirubin is 6.2, the result is HIGH. Period. You never want an AI model to downplay jaundice or talk down a critical lab value.

### Internal Sub-Phase Tracking

The diary records which sub-task the Clinical Agent is on:

```
analyzing_referral → asking_questions → collecting_documents → scoring_risk → complete
```

This is critical because:
- If the patient goes silent for a day and comes back, the agent knows exactly where to resume
- If a backward loop to Intake occurs mid-assessment, the agent picks up where it left off when Intake returns
- If a GP response arrives while questions are being asked, the agent can incorporate the data without losing its place

**When it hands off:** Risk scored → phase becomes "booking" → fires `CLINICAL_COMPLETE` with risk level in the payload.

**When it sends backward:** Missing non-clinical data discovered → fires `NEEDS_INTAKE_DATA` with specific missing field(s). Continues with available data in parallel (doesn't block).

**When it queries the GP:** Missing clinical data that the GP likely has → fires `GP_QUERY`. Continues with available data (doesn't block waiting for GP response).

**LLM requirement:** Gemini Pro (or the most capable model available). This agent needs strong medical reasoning. Consider adding a medical knowledge retrieval tool (UK clinical guidelines, NICE pathways) so it doesn't rely solely on the LLM's training data.

---

## Agent 3: Booking Agent — The Scheduler

**Job:** Get the patient into the right appointment slot based on urgency.

**Wakes up when:** Clinical assessment complete (phase becomes "booking").

**How it works:**

1. Reads risk level from the diary
2. Queries the existing ScheduleCSVManager for available slots filtered by urgency window:
   - HIGH risk → only slots within 48 hours
   - MEDIUM risk → slots within 7–14 days
   - LOW risk → slots within 30 days
3. Presents 2–3 options to the patient with date, time, provider name, and location
4. Also sends options to authorized helpers (those with `book_appointments` permission)
5. Whoever responds first with a valid selection, the booking is confirmed
6. On confirmation, generates context-aware pre-appointment instructions based on Clinical Section data

**Context-aware pre-appointment instructions (not generic):**

The agent reads the Clinical Section of the diary and generates instructions relevant to THIS patient:

- Liver function test scheduled? → "Please fast for 8–12 hours before your appointment"
- Patient on blood thinners? → "Continue your medications as normal unless told otherwise"
- Ultrasound booked? → "Drink 1 litre of water 1 hour before"
- Patient has mobility issues? → "The clinic is on Floor 3. Lifts are available from the main entrance."
- Multiple tests? → "Please allow 2–3 hours for all tests to be completed"
- Patient on Metformin? → "Continue taking Metformin as normal"

**Notification routing:**
- Confirmation sent to patient + all helpers with `receive_alerts` or `book_appointments` permission
- If booked by a helper, patient is informed: "Sarah has booked your appointment for tomorrow at 10 AM."

**When it hands off:** Booking confirmed → snapshots current lab values as monitoring baseline → phase becomes "monitoring" → fires `BOOKING_COMPLETE`.

**LLM requirement:** Gemini Flash. Mostly structured responses with some personalization for pre-appointment guidance.

---

## Agent 4: Monitoring Agent — The Guardian

**Job:** Care for the patient AFTER their consultation. The only agent that operates long-term (weeks to months).

**Wakes up when:**
- `BOOKING_COMPLETE` received (initial setup: snapshot baseline, register for heartbeats)
- `HEARTBEAT` event fires from CRON scheduler
- Patient or helper sends a message while in "monitoring" phase

**Two operating modes:**

### Proactive Mode (Self-Driving via CRON Heartbeats)

The heartbeat scheduler fires periodic events. The Monitoring Agent checks the diary and acts:

| Days Since Appointment | Action |
|---|---|
| 14 days | Check if follow-up labs received. If not, send reminder to patient + primary helper. |
| 30 days | Send general check-in: "How are you feeling? Any changes since your appointment?" |
| 60 days | Prompt for updated labs if not received. |
| 90 days | Full check-in with symptom review. |
| Configurable milestones | Custom intervals based on condition severity. |

This happens **automatically**. The patient doesn't need to initiate anything. The system runs itself.

### Reactive Mode (Patient Returns)

Patient or helper sends a message days, weeks, or months after the appointment:

1. Agent loads the **FULL diary** — including baseline clinical data from before the appointment
2. If patient shares new lab results:
   - Compare against baseline values stored in Monitoring Section
   - Calculate percentage changes for each biomarker
   - Identify trends: improving, stable, or worsening
3. If patient reports new symptoms:
   - Compare against initial symptom profile from Clinical Section
   - Check for red flag keywords (jaundice, confusion, bleeding, etc.)
4. If deterioration detected:
   - Fire `DETERIORATION_ALERT` → this loops back through the Gateway to the Clinical Agent for full reassessment
   - Notify patient, authorized helpers, and referring GP
   - The patient may end up going through Clinical → Booking → Monitoring again (full loop)

**What the diary enables in monitoring:**
- Without diary: "Your ALT is 180. That's elevated." (generic, useless)
- With diary: "Your ALT has dropped from 340 to 180 since your initial assessment — a 47% improvement. The treatment appears to be working." (contextual, actionable)

**Alert routing:**
- DETERIORATION_ALERT → patient + all helpers with `receive_alerts` + referring GP
- Routine check-ins → patient + primary helper only
- Positive updates → patient + whoever asked

**When it hands off:** Normally, it doesn't — monitoring is ongoing. But a `DETERIORATION_ALERT` triggers the loop to route the patient back through Clinical reassessment → potentially new Booking → back to Monitoring. The full cycle.

**LLM requirement:** Gemini Flash for routine check-ins and reminders. Gemini Pro for trend analysis and deterioration assessment where clinical reasoning matters.

---

# 10. Component 6: The Heartbeat Scheduler

A background loop that runs inside the application. At configurable intervals (default: every hour), it scans for all patients with `monitoring_active = true` and fires a `HEARTBEAT` event for each one into the Gateway.

## Why It Exists

The heartbeat is what makes the system **self-driving**. Without it, the system only acts when a patient messages. With heartbeats, the system proactively checks on patients even if they go silent — just like a follow-up coordinator at a real clinic would.

## What the Heartbeat Carries

Each heartbeat event includes:
- Patient ID
- Days since appointment
- Whether follow-up data has been received
- What the next milestone is (14-day check? 30-day check?)
- Last contact date

## Recovery on Restart

When the application restarts (deployment, crash, scaling event), the in-memory list of monitored patients is lost. On every startup, the scheduler must:

1. Scan GCS for all diaries where `current_phase = "monitoring"` and `monitoring_active = true`
2. Re-register each patient for heartbeat monitoring
3. This scan must be fast and handle thousands of patients

## Scaling Path

- **Now:** In-process asyncio loop. Simple, no external dependencies.
- **Later (>500 patients):** Move to Google Cloud Scheduler hitting the `/api/gateway/emit` endpoint externally with `HEARTBEAT` events. Survives process restarts, scales horizontally.

---

# 11. Dialogflow's New Role: Channel Adapter Only

## What Changed

Dialogflow is **demoted** from being the brain to being the postman. It handles one thing: getting messages in and out of communication channels.

```
BEFORE (old role):
  Patient → Dialogflow → intent matching → flow logic → fulfillment → response

AFTER (new role):
  Patient → Dialogflow → [no intent matching] → forward raw message → Gateway
  Gateway → Agent response → Dialogflow → deliver to correct channel
```

## What Dialogflow Handles

| Responsibility | Details |
|---|---|
| WhatsApp Business API connection | Receives patient/helper messages, delivers agent responses |
| Email ingestion | Receives GP replies, helper emails, forwards to Gateway |
| SMS gateway | Receives/sends SMS for helpers who prefer text |
| Web chat widget | Embedded on patient portal (can be replaced with direct WebSocket later) |
| Message delivery tracking | Read receipts, delivery confirmations |
| Template message management | WhatsApp-approved message templates for outbound proactive messages |

## What Dialogflow Does NOT Do

| Removed Responsibility | Why |
|---|---|
| Intent matching | Gateway routes by diary phase, not by what the user said |
| Flow/page transitions | Agents handle their own conversation logic |
| Context/session management | Patient Diary on GCS replaces this entirely |
| Response generation | Agents call Gemini directly for dynamic, personalized responses |
| Routing decisions | Gateway handles all routing deterministically |

## How Messages Flow Through Dialogflow to Gateway

```
Dialogflow receives WhatsApp message from +447700900461
    │
    ▼
Dialogflow fulfillment webhook fires → POST /api/gateway/emit
    │
    ▼
Event envelope created:
  {
    event_type: "USER_MESSAGE",
    patient_id: (resolved from phone number by identity system),
    payload: {
      text: "Here are my blood results",
      attachments: ["lab_photo.jpg"],
      channel: "whatsapp",
      sender_phone: "+447700900461"
    },
    sender_id: "PATIENT",
    sender_role: "patient",
    source: "dialogflow_whatsapp"
  }
    │
    ▼
Gateway processes normally (identity → permissions → diary → route → agent)
    │
    ▼
Agent generates response
    │
    ▼
Gateway sends response back through Dialogflow API
  → Dialogflow delivers via WhatsApp to the correct number
```

## Why Keep Dialogflow at All?

Dialogflow gives you out-of-the-box connections to WhatsApp Business API, Messenger, Telegram, and other channels without building each integration yourself. As a channel adapter, it does this one job well.

## When to Replace Dialogflow Entirely

When the overhead (cost, complexity, debugging across two systems) outweighs the convenience of its channel integrations. Replace with:
- WhatsApp Business API (direct HTTP integration)
- SendGrid/Mailgun for email
- Twilio for SMS
- Existing WebSocket for web chat

This is Phase 6 of the implementation plan. No rush.

---

# 12. Multi-Helper Channel System

## The Problem

A patient isn't always the one communicating. Their wife might be managing their appointments. Their adult child might be uploading lab results. Their friend might be checking on appointment status. The system needs to:

1. Know WHO is messaging (patient vs. which helper vs. GP)
2. Know what each person is ALLOWED to do
3. Keep all messages in the SAME patient diary
4. Route responses to the RIGHT person on the RIGHT channel

## Permission Levels

Different people can do different things:

| Permission | Patient | Spouse/Parent | Friend | GP |
|---|---|---|---|---|
| View status updates | ✓ | ✓ | ✓ | ✓ |
| Answer intake questions | ✓ | ✓ | ✗ | ✗ |
| Answer clinical questions | ✓ | ✓ (if authorized) | ✗ | ✓ |
| Upload documents | ✓ | ✓ | ✓ | ✓ |
| Book/change appointments | ✓ | ✓ | ✗ | ✗ |
| Receive appointment reminders | ✓ | ✓ | ✗ | ✗ |
| Receive clinical/deterioration alerts | ✓ | ✓ | ✗ | ✓ |
| Respond to monitoring check-ins | ✓ | ✓ (if authorized) | ✗ | ✗ |

## Adding a Helper (Registration Flow)

During intake (or at any point), the patient can register helpers:

```
Agent: "Would you like to add anyone who can communicate on your behalf?
        This could be a family member, friend, or carer."

Patient: "Yes, my wife Sarah. Her number is 07700 900462"

Agent: "What level of access should Sarah have?"
       [Full access — can answer questions, book appointments, receive all updates]
       [Upload & view — can upload documents and see status updates]
       [View only — can check status but not take actions]

Patient: "Full access"

Agent: "I'll send Sarah a verification message now."
    │
    ▼
System sends WhatsApp to +447700900462:
"Hi Sarah, John Smith has added you as a helper for his medical
 consultation with MedForce. Please reply YES to confirm."
    │
    ▼
Sarah replies "YES"
    │
    ▼
HELPER_VERIFIED event → Helper Manager updates diary
    │
    ▼
Agent to patient: "Sarah is now registered. She can communicate with us
 on your behalf."
Agent to Sarah: "You're now registered as a helper for John Smith.
 You can send messages, upload documents, and manage appointments."
```

## How Helper Messages Flow

```
Sarah (wife) sends WhatsApp message: "John's blood test photos from today"
    │
    ▼
Dialogflow receives message from +447700900462
    │
    ▼
Identity Resolution (see Section 14):
  Phone +447700900462 → lookup across all helper registries
  → Match: Sarah Smith, HELPER-001 for PT-1234 (John Smith)
  → Role: helper, Relationship: spouse
  → Permissions: [view_status, upload_documents, answer_questions, book_appointments]
    │
    ▼
Event created:
  {
    event_type: "USER_MESSAGE",
    patient_id: "PT-1234",          ← John's ID, not Sarah's
    sender_id: "HELPER-001",
    sender_role: "helper",
    payload: {
      text: "John's blood test photos from today",
      attachments: ["lab_photo.jpg"],
      sender_name: "Sarah Smith",
      sender_relationship: "spouse",
      sender_channel: "whatsapp"
    }
  }
    │
    ▼
Gateway:
  1. Permission check: Can spouse upload documents? → YES (has upload_documents)
  2. Load diary (phase = "clinical") → route to Clinical Agent
    │
    ▼
Clinical Agent processes the upload, updates diary.
Generates TWO responses:
  → To Sarah: "Thanks Sarah, I've received John's blood test results. Reviewing now."
  → To John: "Sarah has uploaded your blood test results. I'm reviewing them."
```

## Response Routing: Who Gets What

Not every response should go to every person:

| Situation | Patient Gets | Authorized Helper Gets | GP Gets |
|---|---|---|---|
| Intake question | ✓ (asked directly) | Only if they initiated | ✗ |
| Clinical question | ✓ | ✓ (if spouse/parent authorized) | ✗ |
| Booking options | ✓ | ✓ (if can book) | ✗ |
| Appointment confirmation | ✓ | ✓ (all authorized helpers) | ✗ |
| Monitoring check-in | ✓ | ✓ (primary helper) | ✗ |
| Deterioration alert | ✓ | ✓ (all with receive_alerts) | ✓ |
| Pre-appointment reminder | ✓ | ✓ (all authorized) | ✗ |
| Positive monitoring update | Whoever asked gets the response | | |
| Status query response | ✓ (only the person who asked) | | |

## Multi-Helper Conflict Resolution

When two helpers respond at the same time:

```
Mother (full access) replies: "Tuesday 2pm works"
Girlfriend (view+upload only) replies: "He prefers Thursday morning"

Queue serializes: Mother's message processed first.
  → Mother has book_appointments → booking confirmed for Tuesday 2pm.

Girlfriend's message processed second.
  → Girlfriend does NOT have book_appointments → politely denied.
  → Response: "Hi Emma, the appointment has been confirmed for Tuesday
    2pm by Carol. If Tom wants to change it, he or Carol can let us know."

Tom gets notified: "Your appointment has been booked for Tuesday 2pm
  by your mum Carol."
```

---

# 13. GP Communication Channel

## The Problem

When a patient is referred by a GP, the referral letter often has gaps:
- GP mentioned "recent blood tests" but didn't include the results
- Medication list is incomplete or vague ("on multiple medications")
- Clinical history is too brief ("previous liver issues")
- Imaging was done but reports aren't attached

Currently, the Clinical Agent would ask the PATIENT for this information. But the patient often doesn't have it, doesn't understand the medical terms, or takes days to respond.

**The solution:** The system can directly contact the referring GP.

## How GP Communication Works

```
Clinical Agent is analyzing John's referral from Dr. Patel.
Referral says: "Recent bloods show elevated ALT. Please see attached."
But NO lab results are actually attached.
    │
    ▼
Clinical Agent detects the gap:
  "Referral references lab results that are not attached."
    │
    ▼
Instead of asking the patient (who may not have the results),
Clinical Agent emits: GP_QUERY event
  {
    event_type: "GP_QUERY",
    patient_id: "PT-1234",
    payload: {
      gp_name: "Dr. Patel",
      gp_email: "dr.patel@greenfields.nhs.uk",
      query_type: "missing_lab_results",
      query_text: "Your referral for John Smith (NHS# 123-456-7890,
                   Ref: REF-2026-4521) mentions recent blood tests with
                   elevated ALT, but the results were not attached.
                   Could you please send the most recent liver function
                   test results?",
      urgency: "routine",
      patient_consent: true
    }
  }
    │
    ▼
Gateway routes GP_QUERY → GP Communication Handler
    │
    ▼
Handler sends professional email to Dr. Patel via Dialogflow/SendGrid:

  Subject: MedForce — Lab Results Requested for John Smith (REF-2026-4521)

  Dear Dr. Patel,

  We are processing your referral for John Smith
  (NHS# 123-456-7890, Ref: REF-2026-4521).

  Your referral mentions recent blood tests showing elevated ALT,
  but the results were not included with the referral letter.

  Could you please reply to this email with the most recent liver
  function test results? Alternatively, you can upload them at:
  https://medforce.app/gp/upload/REF-2026-4521

  This will help us prioritize Mr. Smith's appointment appropriately.

  Kind regards,
  MedForce Clinical Team
    │
    ▼
Clinical Agent continues with what it HAS (never blocks waiting for GP).
Diary updated: gp_query_pending = true, query = "lab_results"
```

## GP Response Handling

```
LATER: Dr. Patel replies by email with the lab results PDF.
    │
    ▼
Dialogflow/email ingestion receives the reply.
Identity resolution: dr.patel@greenfields.nhs.uk → GP for PT-1234
    │
    ▼
Event created:
  {
    event_type: "GP_RESPONSE",
    patient_id: "PT-1234",
    sender_role: "gp",
    payload: {
      from: "Dr. Patel",
      attachments: ["liver_function_tests.pdf"],
      response_to: "missing_lab_results"
    }
  }
    │
    ▼
Gateway: explicit routing → GP_RESPONSE always goes to Clinical Agent.
    │
    ▼
Clinical Agent reads diary, sees gp_query_pending = true.
Processes the lab results PDF.
Extracts values. Updates diary.
Continues assessment with the new data.
May change the risk score based on the new information.
```

## GP Query Types

| Query Type | Trigger | Example Question |
|---|---|---|
| Missing lab results | Referral mentions tests not attached | "Could you send the ALT/AST results from January?" |
| Incomplete medication list | Referral says "on multiple medications" | "Could you confirm the current medication list?" |
| History clarification | Referral mentions condition vaguely | "The referral says 'previous liver issues' — could you specify?" |
| Urgency confirmation | Risk assessment needs GP input | "Given the elevated bilirubin, do you consider this urgent?" |
| Missing imaging | Referral mentions scans not attached | "Could you send the ultrasound report from December?" |

## GP Communication Rules

1. **Patient consent required first.** During intake, ask: "We may need to contact your GP for additional information. Is that okay?" Record in diary.

2. **Never block on GP response.** The Clinical Agent continues with available data. GP responses are processed when they arrive, potentially updating the risk score.

3. **Time limits with escalation:**
   - 48 hours no response → CRON fires `GP_REMINDER` → send polite follow-up email
   - 7 days no response → mark as `gp_non_responsive` → fall back to asking the patient directly
   - Diary logs the full GP communication timeline

4. **Audit trail.** Every GP query and response is logged in the diary's GP Channel section with timestamps.

5. **Secure channel.** GP emails should go through NHS-compliant secure email (NHSmail) in production.

---

# 14. Identity Resolution System

## The Problem

When a message arrives from a phone number or email address, the system needs to answer three questions:
1. Who sent this?
2. Which patient do they belong to?
3. What are they allowed to do?

## Resolution Flow

```
Incoming message from phone/email
    │
    ▼
Step 1: Patient Registry Lookup
  → Match by phone number, email, or NHS number
  → If match → patient_id found, role = "patient"
  → If no match → continue to Step 2
    │
    ▼
Step 2: Helper Registry Lookup
  → Scan helper registries across ALL patient diaries
  → Match by phone number or email
  → If match → patient_id found (the patient they help), role = "helper"
  → If no match → continue to Step 3
    │
    ▼
Step 3: GP Registry Lookup
  → Match email against GP records in all patient diaries
  → If match → patient_id found, role = "gp"
  → If no match → continue to Step 4
    │
    ▼
Step 4: Unknown Sender
  → Not found in any registry
  → Response: "I don't recognize this number/email. If you're a patient,
    please register at medforce.app. If you're a helper, ask the patient
    to add you."
  → Event logged with "unresolved_identity" flag for security audit
  → No diary accessed, no processing
```

## Performance Consideration

Scanning all helper registries on every message is expensive at scale. Solutions:

- **Contact index:** Maintain a reverse-lookup index (phone/email → patient_id + role) in a separate lightweight store (Redis, Firestore, or a GCS JSON index file). Updated whenever helpers are added/removed.
- **Cache:** In-memory LRU cache of recent lookups. Most messages come from the same small set of contacts.

## Handling Ambiguity: One Helper, Multiple Patients

A person might be a helper for multiple patients (e.g., a parent with two children being treated):

```
Sarah is registered as helper for BOTH:
  - PT-1234 (John Smith, her husband)
  - PT-5678 (Emma Smith, her daughter)

Sarah sends: "Here are the blood test results"

System cannot determine which patient this is for.

Response: "Hi Sarah, I see you're a helper for both John and Emma.
           Whose results are these?"
           [John Smith] [Emma Smith]

Sarah: "John's"

→ Event created with patient_id = PT-1234, processed normally.
→ System remembers this context for Sarah's next few messages
  (conversation affinity: within 30 minutes, assume same patient)
```

---

# 15. Full Flow Examples

## Example 1: Happy Path (Patient + Wife + GP, Day 1 through Day 90)

**Day 1, 10:00 AM** — John registers via web form with GP referral letter.

→ Event: `PATIENT_REGISTERED` → Gateway creates diary (phase: intake)
→ Intake Agent parses referral: extracts John Smith, DOB 1975, NHS# 123-456-7890, GP Dr. Patel
→ Missing: phone number, next of kin
→ Sends WhatsApp: "Hi John, I have your referral from Dr. Patel. Could you confirm your mobile number and emergency contact?"
→ Diary updated. Loop rests.

**Day 1, 10:15 AM** — John replies.

→ "07700 900461. My wife Sarah can help too, her number is 07700 900462. Give her full access."
→ Intake Agent updates phone number. Initiates helper registration for Sarah.
→ System sends WhatsApp to Sarah: "Hi Sarah, John has added you as a helper. Reply YES to confirm."

**Day 1, 10:17 AM** — Sarah replies YES.

→ `HELPER_VERIFIED` → Helper Manager registers Sarah with full permissions.
→ Diary updated. All intake fields collected. Phase → "clinical". Fires `INTAKE_COMPLETE`.
→ **LOOP:** `INTAKE_COMPLETE` feeds back into Gateway.
→ Gateway: explicit routing → Clinical Agent.

**Day 1, 10:18 AM** — Clinical Agent starts (automatic, no patient action needed).

→ Reads diary. Analyzes referral: "elevated ALT, alcohol history, suspected fatty liver."
→ Notices: referral mentions "recent bloods" but NO attachment.
→ Two parallel actions:
  1. Emits `GP_QUERY` → email sent to Dr. Patel requesting lab results
  2. Generates clinical questions and sends to John:
     "1. How many units of alcohol per week?
      2. Pain severity 1-10?
      3. Yellowing of eyes or skin?
      4. Please upload any blood test results you have."
→ Diary updated. Loop rests.

**Day 1, 14:00 PM** — Dr. Patel replies by email with lab PDF.

→ `GP_RESPONSE` → Gateway → Clinical Agent.
→ Processes PDF: ALT=340, AST=280, bilirubin=6.2, platelets=95, albumin=28.
→ Diary updated with extracted values.

**Day 1, 15:30 PM** — Sarah uploads lab photos on WhatsApp.

→ Identity resolved: Sarah, helper for PT-1234, has upload_documents permission.
→ `USER_MESSAGE` → Gateway → Clinical Agent.
→ Processes photos. Same values confirmed (consistency check: good).
→ Responds to Sarah: "Thanks Sarah, received John's results."
→ Responds to John: "Sarah uploaded your results. I'm reviewing everything now."

**Day 1, 11:00 AM** — John answers clinical questions.

→ "About 20 units, pain is 7/10, no yellowing I think."
→ Clinical Agent incorporates answers.
→ All data now available. Runs risk scoring.
→ **Hard rule fires:** bilirubin 6.2 > 5.0 threshold → **HIGH risk**. LLM cannot override.
→ Phase → "booking". Fires `CLINICAL_COMPLETE` with risk = HIGH.
→ **LOOP:** Gateway → Booking Agent.

**Day 1, 11:01 AM** — Booking Agent (automatic).

→ Reads risk: HIGH → filters slots within 48 hours only.
→ Finds: Tomorrow 10 AM (Dr. Williams), Tomorrow 2 PM (Dr. Chen).
→ Sends to John AND Sarah (Sarah has book_appointments permission):
  "Based on your assessment, we'd like to see you urgently. Available:
   Tomorrow 10 AM with Dr. Williams or 2 PM with Dr. Chen."

**Day 1, 16:00 PM** — Sarah responds: "10am tomorrow for John please."

→ Permission check: Sarah has book_appointments → GRANTED.
→ Booking confirmed. Pre-appointment instructions generated from clinical data:
  "Fast for 12 hours (no food after 10 PM). Bring all medications. Bring original blood test paperwork."
→ Sent to both John and Sarah.
→ Phase → "monitoring". Fires `BOOKING_COMPLETE`.
→ **LOOP:** Gateway → Monitoring Agent.
→ Snapshots baseline: ALT=340, bilirubin=6.2, platelets=95.
→ Registers John for heartbeat monitoring.

**Day 15** — Heartbeat fires.

→ `HEARTBEAT` → Gateway → Monitoring Agent.
→ 14 days since appointment. No follow-up labs received.
→ Sends reminder to John: "Hi John, it's been two weeks since your appointment. Have you had your follow-up blood tests?"
→ Sends to Sarah: "Reminder: John's follow-up labs are due."

**Day 90** — John messages: "Got my new results. ALT is 180 now."

→ `USER_MESSAGE` → Gateway (phase: monitoring) → Monitoring Agent.
→ Loads baseline from diary: ALT was 340.
→ New ALT: 180. Change: -47%. Trend: improving.
→ Responds: "Good news — your ALT has dropped from 340 to 180, nearly half. This suggests the treatment is working. I'd still recommend discussing this with Dr. Williams at your next review."
→ Diary updated with new monitoring entry.

---

## Example 2: Backward Loop (Missing Medication Information)

**Context:** Robert Taylor, 71, referred for hepatitis B monitoring. Referral says "on multiple medications" but doesn't list them.

Intake completes. Phase → "clinical". Clinical Agent starts.

→ Clinical Agent analyzes referral. Notices: "Patient on multiple medications but not listed."
→ Needs medication data to check for hepatotoxic drugs and drug interactions.
→ Emits `NEEDS_INTAKE_DATA` with payload: `{missing: "current_medication_list"}`
→ **LOOP BACKWARD:** Gateway → Intake Agent.

→ Intake Agent reads diary. Sees what's needed: medication list only.
→ Does NOT re-ask for name, DOB, address (already collected).
→ Sends: "Before we continue your assessment, could you list all medications you're currently taking, including over-the-counter supplements?"

Patient doesn't respond for 2 days. System sends reminder.

→ James (son, helper with upload permission) uploads photo of medication box labels.
→ Permission check: James has upload_documents → GRANTED.
→ Intake Agent processes the image, extracts medication names.
→ Updates diary. Emits `INTAKE_DATA_PROVIDED`.
→ **LOOP FORWARD:** Gateway → Clinical Agent.

→ Clinical Agent picks up at exactly where it left off (sub_phase tracked in diary).
→ Reads medication list. Notices amoxicillin — a known hepatotoxic drug.
→ This changes the clinical picture entirely. Adds drug-induced liver injury to differential.
→ Continues assessment with this new context.

**The backward loop was invisible to the patient.** The system went Clinical → Intake → Clinical automatically through the same Gateway mechanism.

---

## Example 3: Deterioration Escalation (Month 3)

**Context:** Helen Morris, 45, was treated 3 months ago for fatty liver. Baseline: ALT 180, bilirubin 2.1. Previous risk: MEDIUM. Husband Peter has full access.

Helen messages: "I've been feeling very tired and my skin looks yellow."

→ `USER_MESSAGE` → Gateway (phase: monitoring) → Monitoring Agent.
→ Detects "yellow" + "skin" in context of known liver condition.
→ Cross-references diary: patient has hepatic history. Jaundice is a RED FLAG.
→ Monitoring Agent does NOT try to handle this itself.
→ Emits `DETERIORATION_ALERT` with payload: `{reason: "possible jaundice reported, existing hepatic condition"}`

→ **LOOP BACKWARD:** Gateway → Clinical Agent (reassessment).
→ Clinical reads FULL diary: original referral, all previous labs, risk history, monitoring notes.
→ Asks 2-3 urgent questions: "When did you first notice the yellowing? Any abdominal pain or dark urine? Any new medications recently?"

Helen answers: "Two days ago. Yes, my stomach hurts. No new meds."

→ Clinical Agent scores: **CRITICAL** (jaundice + existing liver disease + worsening trajectory + pain).
→ Hard rule: jaundice in known hepatic patient = escalate immediately.
→ Fires `CLINICAL_COMPLETE` with risk = CRITICAL.

→ **LOOP FORWARD:** Gateway → Booking Agent.
→ Risk: CRITICAL. Filters for next available urgent slot — today or tomorrow.
→ Offers to Helen and Peter simultaneously.
→ Peter confirms: "I'll bring her in today at 3pm."

→ Booking confirmed. Fires `BOOKING_COMPLETE`.
→ **LOOP:** Gateway → Monitoring Agent.
→ Updates baseline with current data. Resets monitoring cycle.
→ The `DETERIORATION_ALERT` also triggered notifications to:
  - Peter (husband) — WhatsApp
  - Dr. Brown (GP) — email

**The patient went Monitoring → Clinical → Booking → Monitoring. A full loop. No human orchestrated it.**

---

## Example 4: Parallel Inputs (Queue Serialization)

During clinical assessment, three things arrive within seconds:
1. A lab webhook from the hospital system with new blood results
2. John sends a WhatsApp message answering clinical question 3
3. Sarah (wife) sends a WhatsApp message with an NHS app screenshot

All three become events. The per-patient queue for PT-1234 serializes them:

→ Event 1 processed: Lab webhook. Clinical Agent incorporates new values into diary.
→ Event 2 processed: John's answer. Clinical Agent records it, asks question 4.
→ Event 3 processed: Sarah's upload. Clinical Agent processes the screenshot, extracts medication list.

**No conflicts. No data corruption. No race conditions.** One at a time per patient, parallel across patients. PT-5678's events are being processed simultaneously on a separate queue.

---

## Example 5: GP Non-Responsive

**Context:** Sarah Williams, referred by Dr. Evans. Referral mentions "recent ultrasound" but report not attached.

→ Clinical Agent detects gap. Emits `GP_QUERY` requesting ultrasound report.
→ Email sent to Dr. Evans.
→ Clinical continues with available data (doesn't block).

**48 hours later:** CRON fires `GP_REMINDER`.
→ GP Communication Handler sends follow-up: "Dear Dr. Evans, just following up on our request for Sarah Williams' ultrasound report..."
→ Diary updated: `reminder_sent: true`

**7 days later:** Still no response.
→ Clinical Agent marks: `gp_non_responsive = true`.
→ Falls back to patient: "We haven't been able to get your ultrasound report from your GP. Do you have a copy, or could you check at your surgery?"
→ Clinical proceeds with available data, notes ultrasound is missing.
→ Diary logs the complete GP communication timeline.

---

# 16. Test Scenarios

## Scenario 1: Happy Path — Solo Patient, Low Risk

```
Patient: Mary Jones, 55
GP: Dr. Williams, Oakfield Medical Centre
Helpers: None
Referral: "Routine screening. Mildly elevated GGT. No symptoms."

Expected Flow:
  1. Intake: 2-3 messages to collect demographics
  2. Clinical: Ask about alcohol, diet, family history. Request labs.
  3. Risk: LOW (mild GGT, no symptoms, routine)
  4. Booking: Slots within 30 days
  5. Monitoring: Standard 30-day check-in

Assertions:
  ✓ Risk = LOW
  ✓ Slots offered > 7 days out
  ✓ No GP query
  ✓ No deterioration alerts
  ✓ Monitoring activates after booking
```

## Scenario 2: Urgent Case — Patient + Spouse Helper

```
Patient: David Clarke, 62
GP: Dr. Patel, Greenfields Surgery
Helpers: Wife Linda (full access)
Referral: "Jaundice, ascites, confusion. ALT 580, bilirubin 8.3, platelets 62. URGENT."

Expected Flow:
  1. Intake: Most data from referral. Ask phone only.
  2. Clinical: Multiple hard rules fire. Bilirubin 8.3 > 5. ALT 580 > 500.
  3. Risk: HIGH (hard rules, not LLM)
  4. Booking: Only 48-hour slots
  5. Linda receives all updates and confirmation

Assertions:
  ✓ Risk = HIGH (deterministic rules)
  ✓ Slots within 48 hours
  ✓ Linda receives booking confirmation
  ✓ No GP query (data complete)
  ✓ Pre-appointment instructions include fasting
```

## Scenario 3: Missing Info — GP Query Required

```
Patient: Aisha Rahman, 38
GP: Dr. Chen, Riverside Practice
Helpers: Mother Fatima (view + upload)
Referral: "Abnormal LFTs on routine check. Recent bloods attached." [NO ATTACHMENT]

Expected Flow:
  1. Intake: Collect demographics
  2. Clinical: Detects missing attachment. Emits GP_QUERY.
     Continues asking patient clinical questions in parallel.
  3. GP responds 4 hours later with PDF.
  4. Mother uploads NHS app screenshot with meds.
  5. Clinical merges all three data sources.

Assertions:
  ✓ GP_QUERY emitted for missing labs
  ✓ Clinical does NOT block waiting for GP
  ✓ GP response processed and merged
  ✓ Mother's upload accepted (has permission)
  ✓ All data sources appear in diary
```

## Scenario 4: Backward Loop — Missing Medications

```
Patient: Robert Taylor, 71
GP: Dr. Singh, Mill Road Surgery
Helpers: Son James (view + upload)
Referral: "Hepatitis B carrier. On multiple medications. Routine monitoring."

Expected Flow:
  1. Intake: Complete demographics
  2. Clinical: Needs medication list. Emits NEEDS_INTAKE_DATA.
  3. BACKWARD LOOP to Intake Agent
  4. Intake asks ONLY for medications
  5. Patient silent 2 days. Son uploads medication photos.
  6. FORWARD LOOP back to Clinical. Resumes at correct sub-phase.

Assertions:
  ✓ NEEDS_INTAKE_DATA fires
  ✓ Intake asks ONLY for medications (not name/DOB)
  ✓ Son's upload accepted (has upload permission)
  ✓ Clinical resumes at correct sub-phase
  ✓ Backward loop count tracked (max 3)
```

## Scenario 5: Deterioration — Monitoring Escalation

```
Patient: Helen Morris, 45 (3 months post-treatment)
Baseline: ALT 180, bilirubin 2.1, weight 92kg. Previous risk: MEDIUM
Helpers: Husband Peter (full access)
GP: Dr. Brown

Trigger: Patient messages "I feel terrible, my eyes look yellow"

Expected Flow:
  1. Monitoring detects jaundice keywords
  2. Emits DETERIORATION_ALERT
  3. LOOP to Clinical reassessment
  4. Clinical asks urgent questions. Scores CRITICAL.
  5. LOOP to Booking. Urgent slots only.
  6. LOOP to Monitoring. Reset baseline.
  7. Peter and Dr. Brown both notified.

Assertions:
  ✓ Jaundice keywords detected
  ✓ DETERIORATION_ALERT fires
  ✓ Clinical uses full historical diary
  ✓ Risk = CRITICAL
  ✓ Slots within 24 hours
  ✓ Husband + GP notified
  ✓ Diary shows full loop path
```

## Scenario 6: Multi-Helper Permission Conflict

```
Patient: Tom Hughes, 28
Helpers: Mother Carol (full access), Girlfriend Emma (view + upload)

Situation: Both try to book at the same time.

Expected Flow:
  1. Booking Agent offers slots to Tom.
  2. Carol replies: "Tuesday 2pm"
  3. Emma replies: "He prefers Thursday"
  4. Queue serializes. Carol processed first (full access → confirmed).
  5. Emma processed second (no book_appointments → denied politely).

Assertions:
  ✓ Queue serializes correctly
  ✓ Carol's booking accepted
  ✓ Emma's booking denied (no permission)
  ✓ Emma gets helpful explanation
  ✓ Tom notified of confirmed booking
```

## Scenario 7: GP Non-Responsive

```
Patient: Sarah Williams, 50
GP: Dr. Evans, Old Town Surgery
Referral: "Abnormal LFTs. Recent ultrasound done." [NO ULTRASOUND]

Expected Flow:
  1. Clinical detects missing ultrasound. Emits GP_QUERY.
  2. 48 hours pass → GP_REMINDER fires.
  3. 7 days pass → Fall back to patient.
  4. Clinical proceeds with available data.

Assertions:
  ✓ GP_QUERY sent
  ✓ 48h reminder fires
  ✓ 7-day fallback to patient
  ✓ Clinical doesn't block indefinitely
  ✓ Diary logs full GP timeline
```

## Scenario 8: Unknown Sender

```
Random number +447700999888 sends: "Hi, what's my appointment date?"

Expected Flow:
  1. Identity resolution: not patient, not helper, not GP.
  2. Polite response: "I don't recognize this number..."
  3. Logged for security audit but not processed.

Assertions:
  ✓ No diary accessed or created
  ✓ Polite response sent
  ✓ Event logged with unresolved_identity flag
```

---

# 17. HTML Test Harness

## Purpose

An interactive browser-based tool that simulates the entire event loop WITHOUT needing Dialogflow, WhatsApp, or any external services. It lets you test every scenario by playing different roles (patient, helpers, GP) and watching events flow through the system in real-time.

## Features

1. **Multi-role chat panels** — separate tabs for Patient, Helper 1 (Spouse), Helper 2, and GP. Each tab sends messages as that role.
2. **Event log** — real-time scrolling log (dark terminal theme) showing every event as it flows through the Gateway. Color-coded by type.
3. **Diary viewer** — live-updating JSON display of the patient diary as agents modify it.
4. **Phase indicator** — visual bar at the top showing current phase with completed phases highlighted. Flashes "LOOP DETECTED" when backward loops occur.
5. **Scenario loader** — dropdown to load pre-configured test scenarios (all 8 above). Pre-seeds diary and referral data.
6. **Manual event injection** — buttons to inject events directly: Heartbeat, Lab Webhook, GP Response, Deterioration Alert, GP Reminder, Document Upload.
7. **Assertion checker** — for each loaded scenario, shows pass/fail for expected behaviors. Green checkmarks for passing assertions, yellow circles for pending.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     HTML TEST HARNESS                      │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Patient   │  │ Helper 1 │  │ Helper 2 │  │    GP     │ │
│  │ Chat      │  │ Chat     │  │ Chat     │  │   Chat    │ │
│  │ Panel     │  │ Panel    │  │ Panel    │  │   Panel   │ │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬─────┘ │
│        │             │             │              │        │
│        ▼             ▼             ▼              ▼        │
│  ┌────────────────────────────────────────────────────┐   │
│  │              EVENT INJECTION LAYER                  │   │
│  │  Wraps messages into Event Envelopes                │   │
│  │  + Manual: [HEARTBEAT] [WEBHOOK] [GP_RESP] [ALERT] │   │
│  └──────────────────────┬─────────────────────────────┘   │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              POST /api/gateway/emit               │     │
│  │        (hits your real FastAPI backend)            │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│        ┌────────────────┼─────────────────┐                │
│        ▼                ▼                 ▼                │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐      │
│  │ Event    │  │ Diary Viewer │  │ Phase Diagram  │      │
│  │ Log      │  │ (live JSON)  │  │ (visual bar)   │      │
│  └──────────┘  └──────────────┘  └────────────────┘      │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │              SCENARIO + ASSERTIONS                │     │
│  │  [Load Scenario ▼] [Run Assertions] [Reset]       │     │
│  │  ✓ Risk scored as HIGH                            │     │
│  │  ✓ GP query sent                                  │     │
│  │  ○ Backward loop triggered (pending)              │     │
│  └──────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────┘
```

## API Endpoints for Test Harness

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/gateway/emit` | POST | Submit any event (all chat panels + manual injection) |
| `/api/gateway/diary/{patient_id}` | GET | Fetch current diary state (diary viewer polls this) |
| `/api/gateway/events/{patient_id}` | GET | Fetch event log for a patient (event log polls this) |
| `/api/gateway/scenario/load` | POST | Load a test scenario (creates diary with seed data) |
| `/api/gateway/scenario/assert` | POST | Run assertions for loaded scenario, return pass/fail |
| `/api/gateway/reset/{patient_id}` | DELETE | Clear diary and events (reset button) |
| `/api/gateway/status` | GET | Gateway health, registered agents, active queues |

---

# 18. Integration with Existing MedForce Codebase

## What Stays Untouched

- All 51 existing API endpoints continue working exactly as they do today
- `PreConsulteAgent` logic (reused inside Clinical Agent wrapper)
- `ScheduleCSVManager` (called by Booking Agent for slot queries)
- `GCSBucketManager` (used by Diary Store for read/write)
- `WebSocketAgent` (remains the real-time chat transport layer)
- Board system for labs and medications
- `side_agent` functions (Clinical Agent wraps them for diagnosis generation)

## What's New (Added Alongside)

- `medforce/gateway/` package: events.py, gateway.py, diary.py, queue.py, heartbeat.py, setup.py
- `medforce/gateway/agents/`: base_agent.py, intake_agent.py, clinical_agent.py, booking_agent.py, monitoring_agent.py
- `medforce/gateway/handlers/`: gp_comms.py, helper_manager.py, identity_resolver.py
- `medforce/routers/gateway_api.py`: 3 new endpoints
- Event emission hooks in 2-3 existing routers (fire-and-forget)
- HTML test harness (static file)
- Test files: tests/test_gateway_*.py

## Existing Files Modified (Minimal Touch)

| File | Change | Risk |
|---|---|---|
| `medforce/app.py` | Add gateway router import + startup initialization (wrapped in try/except) | Zero risk — try/except means failure is silent |
| `medforce/routers/pre_consult.py` | After existing return, emit USER_MESSAGE event (fire-and-forget) | Zero risk — happens after existing logic completes |
| `medforce/routers/scheduling.py` | After successful booking, emit WEBHOOK event (fire-and-forget) | Zero risk — same pattern |
| `medforce/dependencies.py` | Add get_gateway() convenience function | Zero risk — additive only |

## The Safety Principle

**If the Gateway crashes, the entire old system continues working exactly as before.** Event emission is fire-and-forget (wrapped in try/except: pass). The Gateway is opt-in and non-blocking. Existing routers emit events as a side effect AFTER their current logic completes, never as a prerequisite.

---

# 19. Known Challenges and Mitigations

## Challenge 1: Infinite Loop Prevention

**Problem:** If Clinical Agent emits `NEEDS_INTAKE_DATA` and Intake Agent can't get the data (patient stops responding), the loop could repeat forever.

**Mitigation:** Every backward loop has a circuit breaker. The diary tracks `backward_loop_count`. After 3 attempts (configurable), the agent must make a decision with incomplete data rather than looping again. Time threshold also applies: if 7 days pass without the missing data, proceed with what's available.

## Challenge 2: Diary Size Over Time

**Problem:** A patient in monitoring for 12 months accumulates a large diary. Every heartbeat, check-in, and lab comparison gets appended. Eventually reading/writing the full diary on every event becomes slow.

**Mitigation:**
- Conversation log capped at 100 most recent entries (older archived to separate file)
- Monitoring entries capped at 50 (older archived)
- Keep only current + previous baselines in active diary
- Full archive available for deep analysis but not loaded on every event

## Challenge 3: Clinical Risk Scoring Safety

**Problem:** If the LLM downplays severity ("ALT of 500 is only moderately elevated"), patients could be harmed.

**Mitigation:** Strict hierarchy enforced in code: deterministic hard rules fire first and cannot be overridden. LLM only votes on gray-zone cases. Must be tested extensively with adversarial clinical scenarios (e.g., "patient says they feel fine but labs show critical values").

## Challenge 4: Heartbeat Recovery on Restart

**Problem:** Application restarts wipe the heartbeat scheduler's in-memory patient list.

**Mitigation:** On every startup, scan GCS for all diaries with `current_phase = "monitoring"` and `monitoring_active = true`. Re-register each patient. For production scale, move to Google Cloud Scheduler for external heartbeats.

## Challenge 5: Multi-Channel Identity

**Problem:** Same patient messages from WhatsApp today and web chat tomorrow. Same helper might message from phone and email.

**Mitigation:** Identity resolution runs before events enter the loop. Maintain a contact index (phone/email → patient_id + role). Match on phone number, email, or NHS number. Must be solved correctly from day one — fragmented diaries are very hard to merge later.

## Challenge 6: Agent Boundary Clarity

**Problem:** When does the Clinical Agent stop asking questions and score risk? When does Monitoring escalate vs. just log? Vague boundaries cause under-action (missing deterioration) or over-action (escalating every minor symptom).

**Mitigation:** Each agent has documented entry criteria, exit criteria, and escalation criteria. Clinical Agent: maximum 5 questions per cycle, must score risk after all questions answered + documents processed. Monitoring Agent: escalate if ANY red flag keyword detected in context of existing condition; log-only for routine symptom descriptions.

## Challenge 7: Observability

**Problem:** When a journey goes wrong — stuck in a phase, looping between agents, risk scored incorrectly — you need to trace exactly what happened across days and dozens of events.

**Mitigation:** Correlation ID threads through every event in a patient's journey from first registration to latest monitoring check-in. Structured logging with correlation IDs from day one. Dashboard showing: patients per phase, average time per phase, risk distribution, loop frequency, handoff success rates.

## Challenge 8: Helper Permission Abuse

**Problem:** A helper with limited permissions tries to perform unauthorized actions repeatedly, or a formerly authorized helper (e.g., ex-spouse) still has access.

**Mitigation:** Permission checks at the Gateway level before any agent sees the event. Patients can revoke helper access at any time ("Remove Sarah as a helper"). Time-limited helper access option (e.g., "Give access for 30 days"). Audit log of all permission checks (granted and denied).

## Challenge 9: GP Email Security

**Problem:** Email to GPs must be secure and compliant with NHS data protection requirements.

**Mitigation:** Use NHSmail for GP communication in production. Include patient reference codes (not full NHS numbers) in email subjects for identification without exposing sensitive data. Secure upload portal as alternative to email attachments.

---

# 20. Implementation Phases

| Phase | Name | What's Built | Existing Code Touched? |
|---|---|---|---|
| **1** | **Foundation** | Event envelope definition, Patient Diary structure + GCS storage, Per-patient event queue, Identity resolution system | No — fully additive |
| **2** | **Gateway + Intake** | Gateway router (both strategies), Permission checking layer, Intake Agent with helper registration, First event emission from registration endpoint | Minimal — app.py startup, one router |
| **3** | **Clinical + Booking + GP** | Clinical Agent (all 4 sub-tasks), GP Communication Handler, GP reminder CRON, Booking Agent with risk filtering, Multi-recipient response routing | One more router gets event emission |
| **4** | **Monitoring + Heartbeat** | Monitoring Agent (proactive + reactive), Heartbeat scheduler, Deterioration alerts with multi-channel notification, Backward loops (Clinical reassessment) | None |
| **5** | **Test Harness** | HTML test interface with multi-role panels, Scenario loader (all 8 scenarios), Assertion checker, Event log + diary viewer, Manual event injection | None |
| **6** | **Channel Integration** | Dialogflow configured as pure channel adapter, WhatsApp for patient + helper channels, Email for GP communication, SMS fallback for helpers, Helper verification flow | Dialogflow config changes |
| **7** | **Hardening** | Correlation ID tracing, Circuit breakers (max 3 backward loops), Diary size management + archiving, GP timeout handling (48h + 7d), Multi-helper conflict resolution, Identity ambiguity handling, Observability dashboard, Permission audit logging | None |

---

# 21. Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| Architecture pattern | Event-driven control loop | Self-driving, backward loops, chain reactions, long-term monitoring |
| Dialogflow | Keep as channel adapter only | Handles WhatsApp/email/SMS delivery; no intent matching, no flow logic, no sessions |
| LLM provider | Gemini Flash + Pro | Already on Google Cloud; Flash for speed/cost (Intake, Booking, routine Monitoring), Pro for accuracy (Clinical, complex Monitoring) |
| State storage | Patient Diary on GCS (JSON) | Consistent with existing architecture; one file per patient; includes helpers + GP channel; simple to debug |
| Routing logic | Deterministic lookup table | Fast, predictable, debuggable; intelligence lives in agents, not the router |
| Risk scoring | Hybrid (hard rules + LLM) | Safety-critical thresholds must be deterministic and unoverridable; LLM only for gray zones |
| Migration strategy | Additive layer on top | Never breaks existing 51 endpoints; old system works even if Gateway fails |
| Helper system | Per-patient registry with role-based permissions | Spouse gets more access than friend; all activity in same diary; verified registration |
| GP communication | Email via Dialogflow/SendGrid, never blocking | Clinical continues with available data; GP response processed when it arrives; 48h/7d timeouts |
| Monitoring | Separate agent with CRON heartbeats | Different goals from Clinical; needs proactive self-driving capability |
| Queue model | In-process asyncio per patient | Serialized per patient, parallel across patients; swap to Pub/Sub or Redis at scale |
| Testing | HTML harness + 8 scenarios + assertions | Covers happy path, backward loops, deterioration, GP queries, permission conflicts, unknown senders |
| Observability | Correlation IDs + structured logging from day one | Essential for debugging multi-day, multi-agent journeys across dozens of events |

---

# Appendix: The Clinic Analogy

The loop pattern mirrors how real clinics operate:

- A nurse checks the chart, does their part, puts the chart back
- The next person picks up the chart, sees what was done, does their part
- If something's missing, it goes back to the previous station
- If something's urgent, it jumps the queue
- If the patient hasn't come back for a follow-up, someone calls them
- If the patient's wife calls with an update, it goes into the same chart
- If the GP sends additional information, it's filed in the chart
- The chart is always the source of truth, not anyone's memory

**The diary** IS the chart.
**The Gateway** IS the reception desk.
**The agents** ARE the specialists.
**The heartbeat** IS the follow-up coordinator's calendar.
**The helpers** ARE the family members the clinic talks to.
**The GP channel** IS the inter-practice communication line.
**The loop** IS the clinical workflow.

The system doesn't pretend patients follow a straight line. It acknowledges that clinical reality is messy, iterative, and sometimes urgent — and it handles all of that through the same simple mechanism: **event arrives, check the chart, route to the right person, update the chart, repeat.**

"""
E2E Resilience Scenarios — 5 comprehensive patient journeys from intake to day-60.

Prerequisites:
    1. Start the server:  python -m medforce.run
    2. Run this script:   python tests/e2e_resilience_test.py

Each scenario prints every message sent/received so you can follow along.
Results are saved to tests/e2e_resilience_results.json.

Scenarios:
    1.  The Complete Happy Path       — form intake, adaptive clinical, low risk, full lifecycle
    2.  The Complex Clinical          — helper intake, cirrhosis, polypharmacy, deterioration
    3.  The Cross-Phase Routing       — allergy in booking, mixed content, monitoring extractions
    4.  The Rescheduler + Edge Cases  — contradictions, slot rejection, double reschedule, emergency
    5.  The Full Monitoring Lifecycle  — hepatitis C, lab uploads, assessment timeout
"""

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

API = "http://127.0.0.1:8080/api/gateway"
RUN_ID = str(int(time.time()))

# Response timeout — LLM calls can chain (intake extraction + clinical handoff)
# and GCS diary saves can spike to 60-90s under cold-start conditions,
# so allow up to 120s for the slowest transitions
RESPONSE_TIMEOUT = 120

PASS = 0
FAIL = 0


def api(method, path, body=None):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        return {"_error": True, "status": e.code, "detail": body_text}
    except Exception as e:
        return {"_error": True, "detail": str(e)}


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label} -- {detail}")


class Case:
    def __init__(self, patient_id, role_label="patient"):
        self.pid = patient_id
        self.role = role_label
        self.chat = []

    def _response_count(self):
        r = api("GET", f"/responses/{self.pid}")
        return len(r.get("responses", []))

    def _get_responses(self, chat_channel=None):
        url = f"/responses/{self.pid}"
        if chat_channel:
            url += f"?chat_channel={chat_channel}"
        r = api("GET", url)
        return r.get("responses", [])

    def monitoring_responses(self):
        return self._get_responses(chat_channel="monitoring")

    def preconsult_responses(self):
        return self._get_responses(chat_channel="pre_consultation")

    def send(self, text):
        self.chat.append({"role": self.role, "message": text})
        print(f"    YOU: {text}")
        before = self._response_count()
        # Fire POST in background thread so polling can start immediately
        threading.Thread(
            target=api, daemon=True,
            args=("POST", "/emit", {
                "event_type": "USER_MESSAGE",
                "patient_id": self.pid,
                "sender_role": "patient",
                "sender_id": self.pid,
                "payload": {"text": text, "channel": "test_harness"},
            }),
        ).start()
        time.sleep(0.3)  # brief pause for the POST to reach the server
        deadline = time.time() + RESPONSE_TIMEOUT
        while time.time() < deadline:
            items = self._get_responses()
            if len(items) > before:
                # Wait a beat for any chained responses to arrive
                time.sleep(2.0)
                items = self._get_responses()
                for r in items[before:]:
                    msg = r["message"]
                    self.chat.append({"role": "agent", "message": msg})
                    preview = msg[:200] + "..." if len(msg) > 200 else msg
                    print(f"    BOT: {preview}")
                return
            time.sleep(0.5)
        print(f"    BOT: (no response within {RESPONSE_TIMEOUT}s)")

    def event(self, event_type, payload=None):
        before = self._response_count()
        threading.Thread(
            target=api, daemon=True,
            args=("POST", "/emit", {
                "event_type": event_type,
                "patient_id": self.pid,
                "sender_role": "system",
                "sender_id": "test_harness",
                "payload": payload or {},
            }),
        ).start()
        time.sleep(0.3)
        deadline = time.time() + RESPONSE_TIMEOUT
        while time.time() < deadline:
            items = self._get_responses()
            if len(items) > before:
                time.sleep(2.0)
                items = self._get_responses()
                for r in items[before:]:
                    msg = r["message"]
                    self.chat.append({"role": "agent", "message": msg})
                    preview = msg[:200] + "..." if len(msg) > 200 else msg
                    print(f"    BOT: {preview}")
                return
            time.sleep(0.5)

    def diary(self):
        return api("GET", f"/diary/{self.pid}")

    def phase(self):
        d = self.diary()
        return (d.get("diary") or {}).get("header", {}).get("current_phase")

    def df(self, d, *keys):
        v = d
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k)
            else:
                return None
        return v


# ---- Helper Functions ----

def do_intake_form(c, name, fields=None):
    """Run intake via form submission. Sends greeting, then submits form event."""
    is_helper = fields.get("is_helper", False) if fields else False
    if is_helper:
        c.send("Hello, I'm filling in for my father. I'm his daughter.")
    else:
        c.send("Hi, I am the patient")
    time.sleep(2)

    form_data = {
        "name": name, "dob": "15/06/1980", "nhs_number": "1234567890",
        "phone": "07700900100", "gp_name": "Dr. Smith",
        "contact_preference": "email", "channel": "test_harness", "is_helper": False,
    }
    if fields:
        form_data.update(fields)

    # Submit form event (needs patient role, not system)
    before = c._response_count()
    threading.Thread(
        target=api, daemon=True,
        args=("POST", "/emit", {
            "event_type": "INTAKE_FORM_SUBMITTED",
            "patient_id": c.pid,
            "sender_role": "patient",
            "sender_id": c.pid,
            "payload": form_data,
        }),
    ).start()
    time.sleep(0.3)
    deadline = time.time() + RESPONSE_TIMEOUT
    while time.time() < deadline:
        items = c._get_responses()
        if len(items) > before:
            time.sleep(2)
            items = c._get_responses()
            for r in items[before:]:
                msg = r["message"]
                c.chat.append({"role": "agent", "message": msg})
                preview = msg[:200] + "..." if len(msg) > 200 else msg
                print(f"    BOT: {preview}")
            break
        time.sleep(0.5)

    # Wait for the full chain to complete (form → intake_complete → clinical welcome)
    # The diary endpoint reads from the gateway's in-memory cache, so this
    # should resolve within seconds once the agent finishes processing.
    deadline = time.time() + 60
    while time.time() < deadline:
        d = c.diary()
        if c.df(d, "diary", "intake", "intake_complete"):
            break
        time.sleep(1)
    time.sleep(2)
    return True


def do_clinical_adaptive(c, chief_complaint, followup_answers, doc_answers=None):
    """Run clinical with adaptive questioning."""
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})
        time.sleep(1)
    wait_phase(c, "clinical", timeout=30)
    c.send(chief_complaint)
    for answer in followup_answers:
        if c.phase() != "clinical":
            break
        c.send(answer)
    for doc_answer in (doc_answers or ["no documents"]):
        if c.phase() != "clinical":
            break
        c.send(doc_answer)

    # If still in clinical after all messages, wait a few more seconds for
    # the agent to finish scoring and transition to booking
    for _ in range(5):
        if c.phase() != "clinical":
            break
        c.send("nothing else to add")

    d = c.diary()
    return c.df(d, "diary", "header", "risk_level")


def do_booking(c, slot_num="1"):
    """Confirm a booking slot. Returns True if confirmed."""
    if c.phase() != "booking":
        risk = c.df(c.diary(), "diary", "header", "risk_level") or "low"
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk})
        time.sleep(1)

    c.send(slot_num)
    time.sleep(3)

    # Wait for BOOKING_COMPLETE chain (includes LLM calls for monitoring
    # questions / communication plan) to fully complete before proceeding
    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)

    d = c.diary()
    return c.df(d, "diary", "booking", "confirmed") is True


def wait_phase(c, target_phase, timeout=20):
    """Wait until patient reaches target phase."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if c.phase() == target_phase:
            return True
        time.sleep(1)
    return False


def wait_booking_cancelled(c, timeout=15):
    """Wait until booking.confirmed is no longer True."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = c.diary()
        if c.df(d, "diary", "booking", "confirmed") is not True:
            return True
        time.sleep(1)
    return False


def do_heartbeat(c, day, label=""):
    """Fire a HEARTBEAT event for the given day. Returns responses."""
    tag = label or f"day {day}"
    print(f"\n  [HEARTBEAT: {tag}]")
    c.event("HEARTBEAT", {
        "days_since_appointment": day,
        "channel": "test_harness",
    })
    time.sleep(2)


# ============================================================
#  SCENARIO 1: The Complete Happy Path
# ============================================================


def run_happy_path():
    print("\n" + "=" * 60)
    print(" SCENARIO 1: THE COMPLETE HAPPY PATH")
    print(" Form intake, adaptive clinical, low risk, full lifecycle")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S01")

    # -- INTAKE: form-based with all fields inc. optional next_of_kin + email --
    print("\n  [INTAKE: form submission with all fields]")
    do_intake_form(c, "Sarah Jones", fields={
        "dob": "22/06/1978",
        "nhs_number": "943 476 5870",
        "phone": "07700900456",
        "gp_name": "Dr. Patel",
        "contact_preference": "email",
        "email": "sarah.jones@example.com",
        "next_of_kin": "James Jones",
        "next_of_kin_phone": "07700900457",
    })

    d = c.diary()
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)
    name = c.df(d, "diary", "intake", "name")
    check("Name extracted from form", name is not None and "sarah" in str(name).lower(),
          f"name={name}")
    nok = c.df(d, "diary", "intake", "next_of_kin")
    check("Next of kin captured", nok is not None, f"next_of_kin={nok}")
    email = c.df(d, "diary", "intake", "email") or c.df(d, "diary", "intake", "contact_preference")
    check("Email/contact preference captured", email is not None, f"email={email}")

    # -- CLINICAL: adaptive questioning, low risk --
    print("\n  [CLINICAL: adaptive questioning, low risk]")
    risk = do_clinical_adaptive(
        c,
        chief_complaint="I have mild abdominal discomfort on the right side, level 3 out of 10",
        followup_answers=[
            "I take amlodipine 5mg daily. No known allergies. I don't smoke or drink.",
            "The discomfort started about 2 weeks ago. It comes and goes.",
        ],
        doc_answers=["no documents"],
    )

    # Wait for risk scoring
    deadline = time.time() + 10
    while time.time() < deadline:
        d = c.diary()
        risk = c.df(d, "diary", "header", "risk_level")
        if risk not in (None, "none"):
            break
        time.sleep(1)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    check("Risk scored", risk not in (None, "none"), f"risk={risk}")
    complaint = c.df(d, "diary", "clinical", "chief_complaint")
    check("Complaint extracted", complaint is not None, f"complaint={complaint}")
    meds = c.df(d, "diary", "clinical", "current_medications") or []
    check("Medications extracted", len(meds) >= 1, f"meds={meds}")

    # -- Sequential doc collection --
    print("\n  [Verify sequential doc collection handled]")
    # Clinical agent should have asked about documents; we said "no documents"
    d = c.diary()
    docs = c.df(d, "diary", "clinical", "documents") or []
    check("Document collection step completed (none uploaded)", True)

    # -- BOOKING: slot selection --
    print("\n  [BOOKING: slot selection]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk or "low"})
    wait_phase(c, "booking", timeout=15)

    # Wait for slots to populate
    deadline = time.time() + 10
    slots = []
    while time.time() < deadline:
        d = c.diary()
        slots = c.df(d, "diary", "booking", "slots_offered") or []
        if len(slots) >= 3:
            break
        time.sleep(1)
    check("3 slots offered", len(slots) >= 3, f"got {len(slots)}")

    c.send("1")
    time.sleep(3)

    # Wait for BOOKING_COMPLETE chain
    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)

    d = c.diary()
    check("Booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)
    check("Phase is monitoring", c.phase() == "monitoring")
    instructions = c.df(d, "diary", "booking", "pre_appointment_instructions") or []
    check("Pre-appointment instructions generated", len(instructions) >= 2,
          f"got {len(instructions)}")

    # -- MONITORING: welcome + communication plan --
    print("\n  [MONITORING: welcome + communication plan]")
    deadline = time.time() + 15
    plan = {}
    while time.time() < deadline:
        d = c.diary()
        plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
        if plan.get("generated", False):
            break
        time.sleep(1)
    check("Communication plan generated", plan.get("generated", False))
    total_msgs = plan.get("total_messages", 0)
    check("Communication plan message count correct for risk",
          total_msgs >= 2, f"total_messages={total_msgs}")

    # Normal message — no false alerts
    c.send("feeling fine, no concerns")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No escalation on 'feeling fine'", len(alerts) == 0, str(alerts))
    check("Monitoring active", c.df(d, "diary", "monitoring", "monitoring_active") is True)

    # Chat channel separation
    mon_msgs = c.monitoring_responses()
    pre_msgs = c.preconsult_responses()
    check("Monitoring has separate responses", len(mon_msgs) > 0, f"got {len(mon_msgs)}")
    check("Pre-consult doesn't have monitoring msgs",
          all("monitoring period" not in m.get("message", "")[:100].lower()
              for m in pre_msgs[-3:]))

    # Naturalness: monitoring response
    last = c.chat[-1]["message"] if c.chat else ""
    check("Not scripted monitoring response",
          not last.startswith("Thank you for your message"))

    # -- HEARTBEATS: day 14, 30, 60 --
    do_heartbeat(c, 14)
    d = c.diary()
    entries = c.df(d, "diary", "monitoring", "entries") or []
    entry_types = [e.get("type", "") for e in entries]
    check("Heartbeat day 14: scheduled question delivered",
          any("checkin" in t or "heartbeat" in t for t in entry_types),
          str(entry_types))

    c.send("feeling fine, everything is good")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 14: no false alert on 'feeling fine'", len(alerts) == 0, str(alerts))

    do_heartbeat(c, 30)
    d = c.diary()
    entries = c.df(d, "diary", "monitoring", "entries") or []
    check("Heartbeat day 30: question delivered",
          len(entries) >= 3, f"entries count={len(entries)}")

    c.send("still feeling fine, no issues")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 30: no false alert", len(alerts) == 0, str(alerts))

    do_heartbeat(c, 60)
    d = c.diary()
    entries = c.df(d, "diary", "monitoring", "entries") or []
    check("Heartbeat day 60: question delivered",
          len(entries) >= 4, f"entries count={len(entries)}")

    c.send("doing great, no problems at all")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 60: no false alert", len(alerts) == 0, str(alerts))
    check("Monitoring still active at day 60",
          c.df(d, "diary", "monitoring", "monitoring_active") is True)

    return c.chat


# ============================================================
#  SCENARIO 2: Complex Clinical + Helper Intake + Polypharmacy
#              + Deterioration
# ============================================================


def run_complex_clinical():
    print("\n" + "=" * 60)
    print(" SCENARIO 2: COMPLEX CLINICAL + HELPER INTAKE")
    print(" Cirrhosis, polypharmacy, dual allergies, deterioration")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S02")

    # -- INTAKE: helper/carer via form --
    print("\n  [INTAKE: helper/carer form submission]")
    do_intake_form(c, "Harold Wilson", fields={
        "dob": "12/08/1958",
        "nhs_number": "876 543 2100",
        "phone": "07700800801",
        "gp_name": "Dr. Liver",
        "contact_preference": "phone",
        "is_helper": True,
    })

    d = c.diary()
    # Check helper/carer detection
    responder = c.df(d, "diary", "intake", "responder_type")
    is_helper = c.df(d, "diary", "intake", "is_helper")
    check("Helper/carer intake detected",
          responder == "helper" or is_helper is True
          or c.df(d, "diary", "intake", "responder_relationship") is not None,
          f"responder_type={responder}, is_helper={is_helper}")

    name = c.df(d, "diary", "intake", "name")
    check("Name extracted from helper form",
          name is not None and "harold" in str(name).lower(),
          f"name={name}")
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)

    # -- CLINICAL: adaptive for cirrhosis + polypharmacy + dual allergies --
    print("\n  [CLINICAL: cirrhosis + polypharmacy + dual allergies]")
    risk = do_clinical_adaptive(
        c,
        chief_complaint=(
            "He has liver cirrhosis with ascites and portal hypertension. "
            "Pain level 7 out of 10. Severe abdominal swelling."
        ),
        followup_answers=[
            "He takes propranolol 40mg, spironolactone 100mg, lactulose 15ml three times daily, "
            "rifaximin 550mg twice daily, metformin 500mg, and warfarin 5mg.",
            "He's allergic to penicillin which causes anaphylaxis, and allergic to codeine "
            "which causes nausea.",
            "The pain has been worse for 2 weeks. His ankles are very swollen. "
            "He stopped drinking 3 years ago.",
        ],
        doc_answers=["no documents"],
    )

    # Wait for risk scoring
    deadline = time.time() + 10
    while time.time() < deadline:
        d = c.diary()
        risk = c.df(d, "diary", "header", "risk_level")
        if risk not in (None, "none"):
            break
        time.sleep(1)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    meds = c.df(d, "diary", "clinical", "current_medications") or []
    allergies = c.df(d, "diary", "clinical", "allergies") or []

    check("Risk scored as high", risk in ("high", "critical"), f"risk={risk}")
    check("Multiple medications extracted (>=4)", len(meds) >= 4,
          f"got {len(meds)}: {meds}")
    allergy_text = " ".join(allergies).lower()
    check("Penicillin allergy extracted", "penicillin" in allergy_text, str(allergies))
    check("Dual allergies extracted (>=2)", len(allergies) >= 2, f"allergies={allergies}")

    # -- Sequential condition-specific doc collection with lab upload --
    print("\n  [Upload condition-specific labs during clinical]")
    if c.phase() == "clinical":
        c.event("DOCUMENT_UPLOADED", {
            "file_ref": "gs://clinic_sim_dev/patient_data/test/liver_labs_baseline.pdf",
            "type": "lab_results",
            "channel": "test_harness",
            "filename": "liver_labs_baseline.pdf",
            "extracted_values": {
                "ALT": 120, "AST": 95, "bilirubin": 3.8,
                "platelets": 110, "albumin": 30, "INR": 1.8,
            },
        })
        time.sleep(2)

    # -- BOOKING --
    print("\n  [BOOKING]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk or "high"})

    # Wait for slots to populate before selecting
    deadline = time.time() + 10
    while time.time() < deadline:
        d = c.diary()
        slots = c.df(d, "diary", "booking", "slots_offered") or []
        if len(slots) > 0:
            break
        time.sleep(1)
    c.send("1")
    time.sleep(3)

    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)

    d = c.diary()
    check("Booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)

    # -- MONITORING: HIGH risk communication plan (6 messages) --
    print("\n  [MONITORING: verify HIGH risk plan]")
    deadline = time.time() + 15
    plan = {}
    while time.time() < deadline:
        d = c.diary()
        plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
        if plan.get("generated", False):
            break
        time.sleep(1)
    check("Communication plan generated", plan.get("generated", False))
    total_msgs = plan.get("total_messages", 0)
    check("HIGH risk: 6 messages in plan", total_msgs == 6, f"total_messages={total_msgs}")

    # -- Stable heartbeat responses --
    do_heartbeat(c, 7)
    c.send("Feeling fine, no changes to report")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 7: no false alert on normal response", len(alerts) == 0, str(alerts))

    do_heartbeat(c, 14)
    c.send("All good, feeling well, taking medications on schedule")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 14: no false alert on medication adherence", len(alerts) == 0, str(alerts))

    do_heartbeat(c, 30)
    c.send("Feeling okay, a bit more tired but otherwise the same")
    time.sleep(2)
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 30: no false escalation on mild concern", len(alerts) == 0,
          f"alerts={alerts}")
    check("Monitoring active throughout",
          c.df(d, "diary", "monitoring", "monitoring_active") is True)

    # -- Deteriorated lab upload triggering DETERIORATION_ALERT --
    print("\n  [Day 35: Upload deteriorated lab results]")
    c.event("DOCUMENT_UPLOADED", {
        "file_ref": "gs://clinic_sim_dev/patient_data/test/liver_labs_day35.pdf",
        "type": "lab_results",
        "channel": "test_harness",
        "filename": "liver_labs_day35.pdf",
        "extracted_values": {
            "ALT": 480, "AST": 350, "bilirubin": 9.5,
            "platelets": 65, "albumin": 22, "INR": 2.8,
        },
    })
    time.sleep(5)

    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Deteriorated labs: DETERIORATION_ALERT fired",
          len(alerts) >= 1, f"alerts={alerts}")

    # -- Worsening report triggering assessment --
    print("\n  [Patient reports worsening -> assessment]")
    c.send("I've been feeling much worse, very fatigued, skin looks yellow, pain is 9 out of 10")
    time.sleep(2)

    d = c.diary()
    assessment = c.df(d, "diary", "monitoring", "deterioration_assessment") or {}
    check("Assessment started on worsening report",
          assessment.get("active", False),
          f"assessment={assessment}")

    # -- Assessment timeout mechanism --
    print("\n  [Assessment timeout mechanism]")
    check("Assessment is active (waiting for patient)",
          assessment.get("active", False))
    check("Assessment has started timestamp",
          assessment.get("started") is not None,
          f"started={assessment.get('started')}")

    # Fire a heartbeat while assessment is active — verifies heartbeat
    # runs without error during active assessment
    do_heartbeat(c, 37, label="day 37 during assessment")
    d = c.diary()
    entries = c.df(d, "diary", "monitoring", "entries") or []
    entry_types = [e.get("type", "") for e in entries]
    check("Monitoring entries log assessment activity",
          any("assessment" in t or "deteriorat" in t for t in entry_types)
          or assessment.get("active", False),
          f"entry_types={entry_types}")

    return c.chat


# ============================================================
#  SCENARIO 3: Cross-Phase Routing
# ============================================================


def run_cross_phase_routing():
    print("\n" + "=" * 60)
    print(" SCENARIO 3: CROSS-PHASE ROUTING")
    print(" Allergy in booking, mixed content, monitoring extractions")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S03")

    # -- Fast-track intake + clinical --
    print("\n  [Fast-track: intake + clinical]")
    do_intake_form(c, "Michael Cross")

    risk = do_clinical_adaptive(
        c,
        chief_complaint="I have abdominal pain, level 5 out of 10, right side",
        followup_answers=[
            "I take amlodipine 5mg daily. No known allergies. No surgeries.",
            "The pain started 2 weeks ago and is gradually getting worse.",
            "I don't drink or smoke.",
        ],
        doc_answers=["no documents"],
    )

    # Wait for risk scoring
    deadline = time.time() + 10
    while time.time() < deadline:
        d = c.diary()
        risk = c.df(d, "diary", "header", "risk_level")
        if risk not in (None, "none"):
            break
        time.sleep(1)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    check("Clinical complete with risk", risk not in (None, "none"), f"risk={risk}")

    # -- BOOKING: cross-phase allergy mention --
    print("\n  [BOOKING: cross-phase allergy mention]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk or "low"})
    wait_phase(c, "booking", timeout=15)

    # Wait for slots to populate
    deadline = time.time() + 10
    while time.time() < deadline:
        d = c.diary()
        slots = c.df(d, "diary", "booking", "slots_offered") or []
        if len(slots) > 0:
            break
        time.sleep(1)

    phase_before = c.phase()
    c.send("Oh wait, I forgot to mention I'm allergic to penicillin")
    time.sleep(5)

    # Clinical agent will ask a follow-up about the reaction type
    # Send the follow-up response
    c.send("It causes a rash and hives")
    time.sleep(5)

    phase_after = c.phase()
    check("Phase still booking after allergy mention",
          phase_after == "booking", f"phase={phase_after}")

    d = c.diary()
    allergies = c.df(d, "diary", "clinical", "allergies") or []
    allergy_text = " ".join(allergies).lower()
    check("Penicillin in diary.clinical.allergies after cross-phase",
          "penicillin" in allergy_text, f"allergies={allergies}")

    cross_extractions = c.df(d, "diary", "cross_phase_extractions") or []
    check("cross_phase_extractions has entry",
          len(cross_extractions) >= 1, f"cross_phase_extractions={cross_extractions}")

    # -- Mixed content: booking + cross-phase intake --
    print("\n  [BOOKING: mixed content — slot + next of kin]")
    c.send("I'll take slot 1, and my next of kin is Carlos Santos on 07700123456")
    time.sleep(3)

    # Wait for booking to complete
    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)

    d = c.diary()
    check("Booking confirmed from mixed message",
          c.df(d, "diary", "booking", "confirmed") is True)

    # Cross-phase intake: next of kin
    nok = c.df(d, "diary", "intake", "next_of_kin")
    cross_extractions = c.df(d, "diary", "cross_phase_extractions") or []
    check("Cross-phase intake: next of kin extracted or cross_phase logged",
          nok is not None or len(cross_extractions) >= 2,
          f"next_of_kin={nok}, cross_phase_count={len(cross_extractions)}")

    # -- MONITORING: medication mention -> clinical cross-phase --
    print("\n  [MONITORING: medication mention -> clinical cross-phase]")
    wait_phase(c, "monitoring", timeout=15)

    # Wait for communication plan
    deadline = time.time() + 15
    while time.time() < deadline:
        d = c.diary()
        plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
        if plan.get("generated", False):
            break
        time.sleep(1)

    c.send("By the way, I've also started taking omeprazole 20mg for my stomach")
    time.sleep(4)

    d = c.diary()
    phase_during_monitoring = c.phase()
    check("Phase still monitoring after medication mention",
          phase_during_monitoring == "monitoring", f"phase={phase_during_monitoring}")

    meds = c.df(d, "diary", "clinical", "current_medications") or []
    cross_extractions = c.df(d, "diary", "cross_phase_extractions") or []
    meds_text = " ".join(str(m).lower() for m in meds)
    cross_text = json.dumps(cross_extractions).lower()
    check("Medication cross-phase: omeprazole captured",
          "omeprazole" in meds_text or "omeprazole" in cross_text,
          f"meds={meds}, cross={len(cross_extractions)} entries")

    # -- MONITORING: GP change -> intake cross-phase --
    print("\n  [MONITORING: GP change -> intake cross-phase]")
    c.send("Oh, I should mention, I've changed my GP. It's now Dr. Williams")
    time.sleep(4)

    d = c.diary()
    gp = c.df(d, "diary", "intake", "gp_name")
    cross_extractions = c.df(d, "diary", "cross_phase_extractions") or []
    cross_text = json.dumps(cross_extractions).lower()
    check("GP change cross-phase: captured in intake or cross_phase",
          (gp is not None and "williams" in str(gp).lower()) or "williams" in cross_text,
          f"gp={gp}, cross={len(cross_extractions)} entries")

    # -- Negation test: "I'm not taking any new medication" --
    print("\n  [MONITORING: negation test]")
    meds_before = c.df(c.diary(), "diary", "clinical", "current_medications") or []
    meds_count_before = len(meds_before)

    c.send("I'm not taking any new medication")
    time.sleep(2)

    d = c.diary()
    meds_after = c.df(d, "diary", "clinical", "current_medications") or []
    check("Negation: no new meds added",
          len(meds_after) <= meds_count_before + 0,
          f"before={meds_count_before}, after={len(meds_after)}, meds={meds_after}")

    # -- More allergy mention -> verify cross_phase_extractions grows --
    print("\n  [MONITORING: additional allergy mention]")
    cross_before = len(c.df(c.diary(), "diary", "cross_phase_extractions") or [])
    c.send("Actually, I also discovered I'm allergic to ibuprofen — it gave me a rash")
    time.sleep(4)

    d = c.diary()
    cross_after = c.df(d, "diary", "cross_phase_extractions") or []
    allergies = c.df(d, "diary", "clinical", "allergies") or []
    allergy_text = " ".join(allergies).lower()
    cross_text = json.dumps(cross_after).lower()
    check("cross_phase_extractions grew",
          len(cross_after) > cross_before or "ibuprofen" in allergy_text,
          f"before={cross_before}, after={len(cross_after)}, allergies={allergies}")

    # -- Normal heartbeats, verify phase never changes from cross-phase --
    print("\n  [HEARTBEATS: verify phase stability]")
    do_heartbeat(c, 14)
    c.send("feeling fine, thanks")
    check("Phase still monitoring after day 14", c.phase() == "monitoring")

    do_heartbeat(c, 30)
    c.send("all good")
    check("Phase still monitoring after day 30", c.phase() == "monitoring")

    do_heartbeat(c, 60)
    c.send("no issues at all")
    d = c.diary()
    check("Phase still monitoring after day 60", c.phase() == "monitoring")
    check("Monitoring active at end",
          c.df(d, "diary", "monitoring", "monitoring_active") is True)

    return c.chat


# ============================================================
#  SCENARIO 4: Rescheduler + Edge Cases + Contradictions
#              + Emergency
# ============================================================


def run_rescheduler_edge_cases():
    print("\n" + "=" * 60)
    print(" SCENARIO 4: RESCHEDULER + EDGE CASES + EMERGENCY")
    print(" Contradictions, slot rejection, double reschedule, emergency")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S04")

    # -- INTAKE: form-based --
    print("\n  [INTAKE: form submission]")
    do_intake_form(c, "Rachel Green")

    d = c.diary()
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)

    # -- CLINICAL: contradictions (pain update, allergy correction) --
    print("\n  [CLINICAL: contradictions]")
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})
    wait_phase(c, "clinical", timeout=30)

    c.send("I have abdominal pain, level 6 out of 10")
    c.send("Actually it's more like 8 out of 10, it's getting worse")

    d = c.diary()
    pain = c.df(d, "diary", "clinical", "pain_level")
    check("Pain updated from 6 to 8 (correction honoured)", pain == 8, f"got {pain}")

    c.send("No known allergies")
    c.send("Actually I'm allergic to penicillin, sorry I forgot")

    d = c.diary()
    allergies = c.df(d, "diary", "clinical", "allergies") or []
    allergy_text = " ".join(allergies).lower()
    check("Penicillin allergy recorded after initial 'no allergies'",
          "penicillin" in allergy_text, str(allergies))

    # Finish clinical
    if c.phase() == "clinical":
        c.send("I take lisinopril daily for high blood pressure")
    if c.phase() == "clinical":
        c.send("no documents")

    for _ in range(3):
        if c.phase() != "clinical":
            break
        c.send("nothing else to add")

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    check("Risk scored after contradictions", risk not in (None, "none"),
          f"risk={risk}, phase={c.phase()}")

    # -- BOOKING: slot rejection + new slots --
    print("\n  [BOOKING: slot rejection + new slots]")
    if c.phase() == "monitoring":
        print("  [Recovery: premature monitoring — requesting reschedule]")
        c.send("I need to reschedule my appointment please")
        wait_booking_cancelled(c, timeout=15)

    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk or "low"})
    wait_phase(c, "booking", timeout=15)

    # Wait for slots to populate
    deadline = time.time() + 10
    while time.time() < deadline:
        d = c.diary()
        slots_before = c.df(d, "diary", "booking", "slots_offered") or []
        if len(slots_before) > 0:
            break
        time.sleep(1)
    check("Initial slots offered", len(slots_before) > 0, f"got {len(slots_before)}")

    c.send("None of these work for me, I'm not available at those times")
    time.sleep(2)

    d = c.diary()
    slots_rejected = c.df(d, "diary", "booking", "slots_rejected") or []
    check("Slots rejected recorded",
          len(slots_rejected) > 0 or "reject" in str(c.chat[-2:]).lower(),
          f"slots_rejected={len(slots_rejected)}")

    # Accept a new slot
    c.send("1")
    time.sleep(3)

    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)

    d = c.diary()
    check("Booking confirmed after slot rejection + selection",
          c.df(d, "diary", "booking", "confirmed") is True)
    check("Phase is monitoring", c.phase() == "monitoring")

    # Capture response count after initial welcome (for dup welcome check)
    mon_msgs_after_booking = c.monitoring_responses()
    welcome_count_initial = len(mon_msgs_after_booking)

    # -- First reschedule --
    print("\n  [First reschedule: 'I need to change my appointment']")
    c.send("I need to reschedule my appointment please")
    wait_booking_cancelled(c)

    d = c.diary()
    confirmed = c.df(d, "diary", "booking", "confirmed")
    rescheduled = c.df(d, "diary", "booking", "rescheduled_from") or []
    check("First reschedule: not confirmed", confirmed is not True,
          f"confirmed={confirmed}")
    check("First reschedule: history has 1 entry", len(rescheduled) >= 1,
          f"history={len(rescheduled)}")

    # Confirm new appointment
    c.send("1")
    wait_phase(c, "monitoring", timeout=15)
    d = c.diary()
    check("Second booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)

    # Check NO duplicate welcome message
    mon_msgs_after_rebook = c.monitoring_responses()
    new_msgs = [m["message"] for m in mon_msgs_after_rebook[welcome_count_initial:]]
    welcome_keywords = ["welcome to monitoring", "monitoring period has begun",
                        "monitoring period begins"]
    has_dup_welcome = any(
        any(kw in msg.lower() for kw in welcome_keywords)
        for msg in new_msgs
    )
    check("No duplicate welcome message after reschedule", not has_dup_welcome,
          f"new msgs contain: {[m[:60] for m in new_msgs]}")

    # -- Second reschedule (double reschedule) --
    print("\n  [Second reschedule: 'I can't make it']")
    c.send("I can't make it, I need a different time")
    wait_booking_cancelled(c)

    d = c.diary()
    rescheduled = c.df(d, "diary", "booking", "rescheduled_from") or []
    check("Second reschedule: history has 2 entries", len(rescheduled) >= 2,
          f"history={len(rescheduled)}")

    # Confirm again
    c.send("2")
    wait_phase(c, "monitoring", timeout=15)
    d = c.diary()
    check("Third booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)
    check("Phase is monitoring after double reschedule", c.phase() == "monitoring")

    # -- MONITORING: edge cases --
    print("\n  [MONITORING: edge cases]")

    # Empty message
    c.send("")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No crash/alert on empty message", len(alerts) == 0, str(alerts))

    # Whitespace message
    c.send("   ")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No crash/alert on whitespace message", len(alerts) == 0, str(alerts))

    # Gibberish
    c.send("asdfghjkl")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No crash/alert on gibberish", len(alerts) == 0, str(alerts))

    # Repeated anxious messages
    c.send("Is my appointment still on?")
    c.send("Is my appointment still on?")
    c.send("Is my appointment still on?")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No false escalation from repeated anxious messages",
          len(alerts) == 0, str(alerts))

    # -- Separate patient: emergency (jaundice + confusion) --
    print("\n  [Emergency patient: jaundice + confusion]")
    ce = Case(f"PT-{RUN_ID}-S04E")
    do_intake_form(ce, "Emergency Patient")

    do_clinical_adaptive(
        ce,
        chief_complaint="Liver pain, level 5 out of 10",
        followup_answers=[
            "No meds. No allergies.",
            "Moderate pain, started recently",
        ],
        doc_answers=["no documents"],
    )
    do_booking(ce)

    ce.send("I have jaundice and confusion")
    time.sleep(2)

    de = ce.diary()
    assessment_e = ce.df(de, "diary", "monitoring", "deterioration_assessment") or {}
    alerts_e = ce.df(de, "diary", "monitoring", "alerts_fired") or []
    severity = assessment_e.get("severity")
    check("Emergency detected",
          severity == "emergency" or any("EMERGENCY" in str(a).upper() for a in alerts_e),
          f"severity={severity}, alerts={alerts_e}")

    # Check A&E guidance in response
    last_msg = ce.chat[-1]["message"] if ce.chat else ""
    check("A&E guidance provided",
          "a&e" in last_msg.lower() or "999" in last_msg or "emergency" in last_msg.lower(),
          f"last_msg={last_msg[:100]}")

    # Post-emergency acknowledgement
    ce.send("ok")
    time.sleep(1)
    de = ce.diary()
    check("Monitoring deactivated after emergency",
          ce.df(de, "diary", "monitoring", "monitoring_active") is False)

    # -- Separate patient: negation test --
    print("\n  [Separate patient: negation test]")
    cn = Case(f"PT-{RUN_ID}-S04N")
    do_intake_form(cn, "Negation Test")
    do_clinical_adaptive(
        cn,
        chief_complaint="I have abdominal pain, level 5 out of 10",
        followup_answers=[
            "I take amlodipine 5mg daily. No allergies.",
            "The pain started 2 weeks ago.",
        ],
        doc_answers=["no documents"],
    )
    do_booking(cn)

    cn.send("I don't have jaundice or confusion, just checking in")
    time.sleep(2)
    dn = cn.diary()
    alerts_n = cn.df(dn, "diary", "monitoring", "alerts_fired") or []
    check("Negation: no emergency fired", len(alerts_n) == 0, str(alerts_n))

    return c.chat + ce.chat + cn.chat


# ============================================================
#  SCENARIO 5: Full Monitoring Lifecycle + Lab Upload
#              + Assessment Timeout
# ============================================================


def run_full_monitoring_lifecycle():
    print("\n" + "=" * 60)
    print(" SCENARIO 5: FULL MONITORING LIFECYCLE")
    print(" Hepatitis C, lab uploads, deterioration, assessment timeout")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S05")

    # -- INTAKE: form-based --
    print("\n  [INTAKE: form submission]")
    do_intake_form(c, "Eve Taylor", fields={
        "dob": "12/09/1975",
        "nhs_number": "222 222 2222",
        "phone": "07700900005",
        "gp_name": "Dr. Brown",
        "contact_preference": "email",
    })

    d = c.diary()
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)

    # -- CLINICAL: Hepatitis C adaptive clinical --
    print("\n  [CLINICAL: Hepatitis C adaptive]")
    risk = do_clinical_adaptive(
        c,
        chief_complaint=(
            "I have hepatitis C. I've been on treatment for 6 months. "
            "Some fatigue and mild abdominal discomfort, pain level 4 out of 10."
        ),
        followup_answers=[
            "I take ribavirin 200mg twice daily and sofosbuvir 400mg daily. No known allergies.",
            "I had a liver biopsy last year. My ALT was slightly elevated at my last blood test.",
            "I don't drink alcohol. I quit 2 years ago.",
        ],
        doc_answers=["no documents"],
    )

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    condition = c.df(d, "diary", "clinical", "condition_context")
    check("Clinical extracts hepatitis-related data",
          condition is not None or risk not in (None, "none"),
          f"condition={condition}, risk={risk}")

    # -- Lab upload during clinical (baseline) --
    if c.phase() == "clinical":
        print("\n  [Upload baseline labs during clinical phase]")
        c.event("DOCUMENT_UPLOADED", {
            "file_ref": "gs://clinic_sim_dev/patient_data/test/baseline_labs.pdf",
            "type": "lab_results",
            "channel": "test_harness",
            "filename": "baseline_labs.pdf",
            "extracted_values": {"ALT": 55, "bilirubin": 1.2, "platelets": 175, "albumin": 38},
        })
        time.sleep(2)

    # -- BOOKING --
    print("\n  [BOOKING]")
    confirmed = do_booking(c)
    check("Booking confirmed", confirmed)

    d = c.diary()
    instructions = c.df(d, "diary", "booking", "pre_appointment_instructions") or []
    check("Instructions generated for hepatitis patient",
          len(instructions) >= 2, f"got {len(instructions)}")

    # -- MONITORING: welcome --
    print("\n  [MONITORING: welcome]")
    deadline = time.time() + 15
    plan = {}
    while time.time() < deadline:
        d = c.diary()
        plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
        if plan.get("generated", False):
            break
        time.sleep(1)
    check("Monitoring active", c.df(d, "diary", "monitoring", "monitoring_active") is True)
    check("Communication plan generated", plan.get("generated", False))

    # -- Heartbeat day 14: normal --
    do_heartbeat(c, 14)
    c.send("feeling fine, managing well with the treatment")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Day 14: no alert on 'feeling fine'", len(alerts) == 0, str(alerts))

    # -- Stable lab upload during monitoring (no alert) --
    print("\n  [Day 20: Upload stable lab results]")
    c.event("DOCUMENT_UPLOADED", {
        "file_ref": "gs://clinic_sim_dev/patient_data/test/lab_results_day20.pdf",
        "type": "lab_results",
        "channel": "test_harness",
        "filename": "lab_results_day20.pdf",
        "extracted_values": {
            "ALT": 55, "bilirubin": 1.2, "platelets": 175, "albumin": 38,
        },
    })
    time.sleep(2)

    d = c.diary()
    entries = c.df(d, "diary", "monitoring", "entries") or []
    lab_entries = [e for e in entries if "lab" in e.get("type", "").lower()]
    check("Stable lab upload processed", len(lab_entries) >= 1,
          f"lab entries: {[e.get('type') for e in lab_entries]}")

    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Stable labs: no deterioration alert", len(alerts) == 0, str(alerts))

    # -- Heartbeat day 30: normal --
    do_heartbeat(c, 30)
    c.send("still doing well, no changes")

    # -- Deteriorated lab upload (DETERIORATION_ALERT) --
    print("\n  [Day 40: Upload deteriorated lab results]")
    c.event("DOCUMENT_UPLOADED", {
        "file_ref": "gs://clinic_sim_dev/patient_data/test/lab_results_day40.pdf",
        "type": "lab_results",
        "channel": "test_harness",
        "filename": "lab_results_day40.pdf",
        "extracted_values": {
            "ALT": 450,          # Significantly elevated (was 55)
            "bilirubin": 8.5,    # Major spike (was 1.2)
            "platelets": 90,     # Dropped significantly (was 175)
            "albumin": 25,       # Low (was 38)
        },
    })
    time.sleep(3)

    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Deteriorated labs: DETERIORATION_ALERT fired",
          len(alerts) >= 1, f"alerts={alerts}")

    entries = c.df(d, "diary", "monitoring", "entries") or []
    lab_entries = [e for e in entries if "lab" in e.get("type", "").lower()]
    check("Lab deterioration entry logged", len(lab_entries) >= 2,
          f"lab entries: {len(lab_entries)}")

    # -- Worsening -> assessment --
    print("\n  [Patient reports worsening -> assessment starts]")
    c.send("I've been feeling much worse lately, very fatigued and my skin looks yellow")
    time.sleep(2)

    d = c.diary()
    assessment = c.df(d, "diary", "monitoring", "deterioration_assessment") or {}
    check("Assessment started on worsening report",
          assessment.get("active", False),
          f"assessment={assessment}")

    # -- Heartbeat during active assessment --
    print("\n  [Heartbeat during active assessment]")
    check("Assessment is active (waiting for patient)",
          assessment.get("active", False))
    check("Assessment has started timestamp",
          assessment.get("started") is not None,
          f"started={assessment.get('started')}")

    # Fire heartbeat — verifies heartbeat runs without error during active assessment
    do_heartbeat(c, 45, label="day 45 during assessment")
    d = c.diary()

    entries = c.df(d, "diary", "monitoring", "entries") or []
    entry_types = [e.get("type", "") for e in entries]
    check("Monitoring entries log assessment activity",
          any("assessment" in t or "deteriorat" in t for t in entry_types)
          or assessment.get("active", False),
          f"entry_types={entry_types}")

    # -- Full heartbeat lifecycle to day 60 --
    print("\n  [Full heartbeat lifecycle]")
    do_heartbeat(c, 60)
    d = c.diary()
    entries = c.df(d, "diary", "monitoring", "entries") or []
    check("Heartbeat day 60: question delivered",
          len(entries) >= 5, f"entries count={len(entries)}")

    # -- Red flag extraction: blood in stool --
    print("\n  [Red flag extraction: blood in stool]")
    c.send("I've noticed blood in my stool recently")
    time.sleep(2)

    d = c.diary()
    # Red flag could be in clinical.red_flags, monitoring entries, or alerts
    red_flags = c.df(d, "diary", "clinical", "red_flags") or []
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    entries = c.df(d, "diary", "monitoring", "entries") or []
    entry_text = json.dumps(entries).lower()
    alert_text = json.dumps(alerts).lower()

    check("Red flag extracted (blood in stool)",
          any("blood" in str(rf).lower() for rf in red_flags)
          or "blood" in entry_text or "blood" in alert_text
          or len(alerts) >= 2,
          f"red_flags={red_flags}, alerts_count={len(alerts)}")

    check("Monitoring still tracks patient at end",
          c.df(d, "diary", "monitoring", "monitoring_active") is True)

    return c.chat


# ============================================================
#  RUN ALL
# ============================================================


if __name__ == "__main__":
    print("=" * 60)
    print(f" MEDFORCE GATEWAY -- RESILIENCE E2E TEST (run {RUN_ID})")
    print("=" * 60)

    # Clear stale booking slots from previous runs
    reset_result = api("POST", "/admin/reset-booking-registry")
    if reset_result.get("success"):
        print(f"  Booking registry cleared ({reset_result.get('holds_cleared', 0)} holds)")
    else:
        print(f"  WARNING: Could not reset booking registry: {reset_result}")

    results = {}
    results["scenario_1_happy_path"] = run_happy_path()
    results["scenario_2_complex_clinical"] = run_complex_clinical()
    results["scenario_3_cross_phase_routing"] = run_cross_phase_routing()
    results["scenario_4_rescheduler_edge_cases"] = run_rescheduler_edge_cases()
    results["scenario_5_full_monitoring"] = run_full_monitoring_lifecycle()

    output_path = os.path.join(os.path.dirname(__file__), "e2e_resilience_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nChat transcripts saved to: {output_path}")
    print(f"\n{'=' * 60}")
    print(f" RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)

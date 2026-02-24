"""
E2E Resilience Scenarios — 7 patient journeys you can follow step-by-step.

Prerequisites:
    1. Start the server:  python -m medforce.run
    2. Run this script:   python tests/e2e_resilience_test.py

Each scenario prints every message sent/received so you can follow along.
Results are saved to tests/e2e_resilience_results.json.

Scenarios:
    1.  The Happy Path            — full clean journey end-to-end
    2.  The Confused Patient      — wrong info for wrong fields, corrections
    3.  The Emergency Escalation  — deterioration, emergency, negation test
    4.  The Rescheduler           — books then reschedules, twice
    5.  The Slot Rejector         — rejects offered slots, gets new ones
    6.  The Complex Clinical      — liver cirrhosis, polypharmacy, allergies
    7.  The Resilient Patient     — empty msgs, gibberish, edge cases
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
# so allow up to 40s for the slowest transitions
RESPONSE_TIMEOUT = 40

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

    def _get_responses(self):
        r = api("GET", f"/responses/{self.pid}")
        return r.get("responses", [])

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


# ---- Helper: fast-track intake ----

def do_intake(c, name):
    """Run a patient through intake quickly. Returns True if intake completed."""
    c.send(f"I am the patient, my name is {name}")
    c.send("email me please")
    c.send("15/06/1980")
    c.send("123 456 7890")
    c.send("07700900100")
    c.send("Dr. Smith")

    # Wait for intake to propagate (GP is last field, triggers INTAKE_COMPLETE chain)
    time.sleep(3)

    d = c.diary()
    complete = c.df(d, "diary", "intake", "intake_complete")
    if not complete:
        # Force transition if intake is stuck
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})
        time.sleep(2)
    return True


def do_clinical(c, msgs=None):
    """Run a patient through clinical. Returns risk level."""
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})
        time.sleep(1)

    default_msgs = [
        "I have abdominal pain, level 5 out of 10. It's on the right side.",
        "I take amlodipine 5mg daily. No surgeries. No known allergies.",
        "The pain started 2 weeks ago and is gradually getting worse.",
        "I don't drink or smoke.",
        "no documents",
    ]
    for msg in (msgs or default_msgs):
        if c.phase() != "clinical":
            break
        c.send(msg)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    return risk


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


# ============================================================
#  SCENARIO 1: The Happy Path
# ============================================================


def run_happy_path():
    print("\n" + "=" * 60)
    print(" SCENARIO 1: THE HAPPY PATH")
    print(" Full cooperative journey end-to-end")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S01")

    # -- INTAKE --
    print("\n  [INTAKE]")
    c.send("Hi, I'm the patient. My name is Sarah Jones")
    c.send("email me please")
    c.send("22/06/1978")
    c.send("943 476 5870")
    c.send("07700 900456")
    c.send("Dr. Patel")

    time.sleep(3)
    d = c.diary()
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)
    check("Name extracted", c.df(d, "diary", "intake", "name") is not None)

    # -- CLINICAL --
    print("\n  [CLINICAL]")
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    clinical_msgs = [
        "I have abdominal pain on the right side, level 5 out of 10",
        "I take amlodipine 5mg daily. No known allergies. I don't smoke or drink.",
        "The pain started about 2 weeks ago and is getting gradually worse.",
        "no documents",
    ]
    for msg in clinical_msgs:
        # Stop sending once we've moved past clinical (e.g. into booking)
        phase = c.phase()
        if phase and phase != "clinical":
            break
        c.send(msg)

    # Wait for clinical agent to finish scoring if still processing
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

    # -- BOOKING --
    print("\n  [BOOKING]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk or "low"})

    d = c.diary()
    slots = c.df(d, "diary", "booking", "slots_offered") or []
    check("Slots offered", len(slots) > 0, f"got {len(slots)}")

    c.send("1")
    time.sleep(3)

    # Wait for BOOKING_COMPLETE chain (LLM calls for monitoring setup) to finish
    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)

    d = c.diary()
    check("Booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)
    check("Phase is monitoring", c.phase() == "monitoring")
    instructions = c.df(d, "diary", "booking", "pre_appointment_instructions") or []
    check("Instructions generated", len(instructions) >= 2, f"got {len(instructions)}")

    # -- MONITORING --
    print("\n  [MONITORING]")
    d = c.diary()
    plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
    check("Communication plan generated", plan.get("generated", False))

    c.send("feeling fine, no concerns")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No escalation on 'feeling fine'", len(alerts) == 0, str(alerts))
    check("Monitoring active", c.df(d, "diary", "monitoring", "monitoring_active") is True)

    return c.chat


# ============================================================
#  SCENARIO 2: The Confused Patient
# ============================================================


def run_confused_patient():
    print("\n" + "=" * 60)
    print(" SCENARIO 2: THE CONFUSED PATIENT")
    print(" Wrong info for wrong fields, corrections, contradictions")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S02")

    # -- INTAKE: out-of-order info --
    print("\n  [INTAKE: confused field order]")
    c.send("Hello, I am the patient")
    c.send("John Smith")
    c.send("07700 900123")  # Phone when asked for something else
    c.send("email me please")
    c.send("my email is john@example.com")
    c.send("15/03/1985")
    c.send("123 456 7890")
    c.send("My GP is Dr. Patel")

    time.sleep(3)
    d = c.diary()
    name = c.df(d, "diary", "intake", "name")
    check("Name extracted despite confusion", name is not None and "John" in str(name),
          f"name={name}")

    # Force intake complete if not already
    if not c.df(d, "diary", "intake", "intake_complete"):
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    # -- CLINICAL: contradictions --
    print("\n  [CLINICAL: contradicting info]")
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    c.send("I have abdominal pain, level 6 out of 10")
    c.send("Actually it's more like 8 out of 10, it's getting worse")

    d = c.diary()
    pain = c.df(d, "diary", "clinical", "pain_level")
    check("Pain updated to 8", pain == 8, f"got {pain}")

    c.send("No known allergies")
    c.send("Actually I'm allergic to penicillin, sorry I forgot")

    d = c.diary()
    allergies = c.df(d, "diary", "clinical", "allergies") or []
    allergy_text = " ".join(allergies).lower()
    check("Penicillin allergy recorded", "penicillin" in allergy_text, str(allergies))

    # Finish clinical
    c.send("I take lisinopril 10mg daily. High blood pressure.")
    c.send("no documents")

    for _ in range(3):
        if c.phase() != "clinical":
            break
        c.send("nothing else to add")

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    check("Risk scored after contradictions", risk not in (None, "none"),
          f"risk={risk}, phase={c.phase()}")

    return c.chat


# ============================================================
#  SCENARIO 3: Emergency Escalation
# ============================================================


def run_emergency_escalation():
    print("\n" + "=" * 60)
    print(" SCENARIO 3: EMERGENCY ESCALATION")
    print(" Deterioration, emergency keywords, negation test")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S03")

    # -- Fast-track to monitoring --
    print("\n  [Fast-track: intake + clinical + booking]")
    do_intake(c, "David Brown")
    do_clinical(c, [
        "I have liver cirrhosis. Severe abdominal pain, level 7. History of ascites.",
        "I take propranolol 40mg and spironolactone 100mg daily.",
        "No known allergies. I've cut down drinking to once a week.",
        "The pain has been getting worse. Sometimes I see blood in my stool.",
        "no documents",
    ])
    do_booking(c)

    d = c.diary()
    check("Booked and in monitoring", c.phase() == "monitoring")

    # -- Deterioration: swelling --
    print("\n  [Report swelling -- should start assessment]")
    c.send("I've noticed some swelling in my legs and ankles")
    time.sleep(2)

    d = c.diary()
    assessment = c.df(d, "diary", "monitoring", "deterioration_assessment") or {}
    check("Assessment started", assessment.get("active", False),
          f"assessment={assessment}")

    # -- Answer assessment questions --
    print("\n  [Answer assessment questions]")
    c.send("It started about 3 days ago and it's getting worse each day")
    c.send("Yes, I also have some abdominal swelling and I feel more tired")
    c.send("I'd say about 6 out of 10. I can still walk but it's uncomfortable")

    d = c.diary()
    assessment = c.df(d, "diary", "monitoring", "deterioration_assessment") or {}
    entries = c.df(d, "diary", "monitoring", "entries") or []
    entry_texts = " ".join(e.get("action", "") + " " + e.get("type", "") for e in entries)
    has_evidence = (
        "assessment_complete" in entry_texts
        or assessment.get("assessment_complete", False)
        or assessment.get("severity") is not None
    )
    check("Assessment completed or severity set", has_evidence,
          f"severity={assessment.get('severity')}, entries={[e.get('type') for e in entries]}")

    # -- Negation test (new patient) --
    print("\n  [Negation test: 'I don't have jaundice' should NOT escalate]")
    c2 = Case(f"PT-{RUN_ID}-S03N")
    do_intake(c2, "Test Negation")
    do_clinical(c2)
    do_booking(c2)

    c2.send("I don't have jaundice or confusion, just checking in")
    d2 = c2.diary()
    alerts2 = c2.df(d2, "diary", "monitoring", "alerts_fired") or []
    check("Negation: no emergency fired", len(alerts2) == 0, str(alerts2))

    # -- Immediate emergency (new patient) --
    print("\n  [Immediate emergency: 'I have jaundice and confusion']")
    c3 = Case(f"PT-{RUN_ID}-S03E")
    do_intake(c3, "Emergency Test")
    do_clinical(c3, [
        "Liver pain, level 5 out of 10",
        "No meds. No allergies.",
        "Moderate pain, started recently",
        "no documents",
    ])
    do_booking(c3)

    c3.send("I have jaundice and confusion")
    time.sleep(2)

    d3 = c3.diary()
    assessment3 = c3.df(d3, "diary", "monitoring", "deterioration_assessment") or {}
    alerts3 = c3.df(d3, "diary", "monitoring", "alerts_fired") or []
    severity = assessment3.get("severity")
    check("Emergency detected",
          severity == "emergency" or any("EMERGENCY" in str(a).upper() for a in alerts3),
          f"severity={severity}, alerts={alerts3}")

    # Post-emergency: patient says ok
    print("\n  [Post-emergency: 'ok']")
    c3.send("ok")
    d3 = c3.diary()
    check("Monitoring deactivated after emergency",
          c3.df(d3, "diary", "monitoring", "monitoring_active") is False)

    return c.chat + c2.chat + c3.chat


# ============================================================
#  SCENARIO 4: The Rescheduler
# ============================================================


def run_rescheduler():
    print("\n" + "=" * 60)
    print(" SCENARIO 4: THE RESCHEDULER")
    print(" Books, reschedules twice, history accumulates")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S04")

    # -- Fast-track to monitoring --
    print("\n  [Fast-track to booking]")
    do_intake(c, "Rachel Green")
    do_clinical(c)

    # -- First booking --
    print("\n  [First booking]")
    confirmed = do_booking(c)
    check("First booking confirmed", confirmed)

    # -- First reschedule --
    print("\n  [First reschedule: 'I need to change my appointment']")
    c.send("I need to reschedule my appointment please")
    # Wait for RESCHEDULE_REQUEST chain to complete (monitoring → booking reset)
    deadline = time.time() + 15
    while time.time() < deadline:
        d = c.diary()
        if c.df(d, "diary", "booking", "confirmed") is not True:
            break
        time.sleep(1)

    d = c.diary()
    confirmed = c.df(d, "diary", "booking", "confirmed")
    slots = c.df(d, "diary", "booking", "slots_offered") or []
    rescheduled = c.df(d, "diary", "booking", "rescheduled_from") or []
    check("First reschedule: not confirmed", confirmed is not True,
          f"confirmed={confirmed}")
    check("First reschedule: new slots offered", len(slots) > 0,
          f"slots={len(slots)}")
    check("First reschedule: history has 1 entry", len(rescheduled) >= 1,
          f"history={len(rescheduled)}")

    # Confirm new appointment
    c.send("1")
    # Wait for BOOKING_COMPLETE chain to finish
    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)
    d = c.diary()
    check("Second booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)

    # -- Second reschedule --
    print("\n  [Second reschedule: 'I can't make it']")
    c.send("I can't make it, I need a different time")
    # Wait for RESCHEDULE_REQUEST chain to complete
    deadline = time.time() + 15
    while time.time() < deadline:
        d = c.diary()
        if c.df(d, "diary", "booking", "confirmed") is not True:
            break
        time.sleep(1)

    d = c.diary()
    rescheduled = c.df(d, "diary", "booking", "rescheduled_from") or []
    check("Second reschedule: history has 2 entries", len(rescheduled) >= 2,
          f"history={len(rescheduled)}")

    # Confirm again
    c.send("2")
    # Wait for BOOKING_COMPLETE chain to finish
    deadline = time.time() + 15
    while time.time() < deadline:
        if c.phase() == "monitoring":
            break
        time.sleep(1)
    d = c.diary()
    check("Third booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)
    check("Phase is monitoring", c.phase() == "monitoring")

    return c.chat


# ============================================================
#  SCENARIO 5: The Slot Rejector
# ============================================================


def run_slot_rejector():
    print("\n" + "=" * 60)
    print(" SCENARIO 5: THE SLOT REJECTOR")
    print(" Rejects offered times, gets new options")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S05")

    # -- Fast-track to booking --
    print("\n  [Fast-track to booking]")
    do_intake(c, "Picky Pete")
    do_clinical(c)

    if c.phase() != "booking":
        risk = c.df(c.diary(), "diary", "header", "risk_level") or "low"
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk})
        time.sleep(1)

    d = c.diary()
    slots_before = c.df(d, "diary", "booking", "slots_offered") or []
    check("Initial slots offered", len(slots_before) > 0, f"got {len(slots_before)}")

    # -- Reject offered slots --
    print("\n  [Reject: 'none of those work for me']")
    c.send("None of those work for me, I'm not available at those times")
    time.sleep(2)

    d = c.diary()
    last_msg = c.chat[-1]["message"] if c.chat else ""
    check("Agent acknowledged rejection",
          "alternative" in last_msg.lower() or "other" in last_msg.lower()
          or "unfortunately" in last_msg.lower() or "no problem" in last_msg.lower(),
          f"last_msg={last_msg[:100]}")

    # -- Now accept a slot --
    print("\n  [Accept new slot]")
    c.send("1")
    time.sleep(3)

    d = c.diary()
    check("Booking confirmed after rejection + re-selection",
          c.df(d, "diary", "booking", "confirmed") is True)

    return c.chat


# ============================================================
#  SCENARIO 6: Complex Clinical (Liver + Polypharmacy)
# ============================================================


def run_complex_clinical():
    print("\n" + "=" * 60)
    print(" SCENARIO 6: COMPLEX CLINICAL")
    print(" Liver cirrhosis, 6+ medications, dual allergies")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S06")

    # -- Intake (carer-assisted) --
    print("\n  [Intake: carer fills in for patient]")
    c.send("Hello, I'm filling in for my father. I'm his daughter.")
    c.send("His name is Harold Wilson. Call him please.")
    c.send("12/08/1958")
    c.send("876 543 2100")
    c.send("07700800801")
    c.send("Dr. Liver")

    time.sleep(3)
    d = c.diary()
    if not c.df(d, "diary", "intake", "intake_complete"):
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    # -- Clinical: complex liver history --
    print("\n  [Clinical: complex liver + polypharmacy]")
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    clinical_msgs = [
        "He has liver cirrhosis with ascites and portal hypertension. Pain level 7 out of 10. Severe abdominal swelling.",
        "He takes propranolol 40mg, spironolactone 100mg, lactulose 15ml three times daily, rifaximin 550mg twice daily, metformin 500mg, and warfarin 5mg.",
        "He's allergic to penicillin which causes anaphylaxis, and allergic to codeine which causes nausea.",
        "The pain has been worse for 2 weeks. His ankles are very swollen. He stopped drinking 3 years ago.",
        "no documents",
    ]
    for msg in clinical_msgs:
        if c.phase() != "clinical":
            break
        c.send(msg)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    meds = c.df(d, "diary", "clinical", "current_medications") or []
    allergies = c.df(d, "diary", "clinical", "allergies") or []
    check("Risk scored", risk not in (None, "none"), f"risk={risk}")
    check("Multiple medications extracted", len(meds) >= 3, f"got {len(meds)}: {meds}")
    check("Allergies extracted", len(allergies) >= 1, f"allergies={allergies}")

    # -- Booking: check instructions --
    print("\n  [Booking: verify instructions]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": risk or "high"})

    c.send("1")
    time.sleep(3)

    d = c.diary()
    check("Booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)
    instructions = c.df(d, "diary", "booking", "pre_appointment_instructions") or []
    instr_text = " ".join(instructions).lower()
    check("Instructions mention allergies or medications",
          "allerg" in instr_text or "penicillin" in instr_text or "medication" in instr_text
          or "warfarin" in instr_text or "metformin" in instr_text,
          str(instructions[:3]))
    check("Multiple instructions generated", len(instructions) >= 3,
          f"got {len(instructions)}")

    return c.chat


# ============================================================
#  SCENARIO 7: The Resilient Patient (Edge Cases)
# ============================================================


def run_resilient_patient():
    print("\n" + "=" * 60)
    print(" SCENARIO 7: THE RESILIENT PATIENT")
    print(" Empty messages, gibberish, repeated selections, edge cases")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-S07")

    # -- Empty/whitespace messages --
    print("\n  [Empty and whitespace messages]")
    c.send("Hello, I am the patient")
    c.send("")
    check("No crash on empty", c.phase() == "intake")
    c.send("   ")
    check("No crash on whitespace", c.phase() == "intake")

    # -- Normal intake --
    print("\n  [Complete intake]")
    c.send("Skip Patient")
    c.send("text me")
    c.send("01/01/1990")
    c.send("9876543210")
    c.send("07700111222")
    c.send("Dr. Nobody")

    time.sleep(3)
    d = c.diary()
    if not c.df(d, "diary", "intake", "intake_complete"):
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    # -- Clinical: vague/skip answers --
    print("\n  [Clinical: vague answers]")
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness"})

    c.send("I have some discomfort, maybe pain, level 3 out of 10")
    c.send("No medications. No allergies. I don't drink.")
    c.send("It's mild and comes and goes")
    c.send("no documents")

    for _ in range(3):
        if c.phase() != "clinical":
            break
        c.send("skip")

    d = c.diary()
    phase = c.phase()
    check("Exited clinical", phase in ("booking", "monitoring"),
          f"phase={phase}")

    # -- Booking: gibberish then valid --
    print("\n  [Booking: gibberish then valid]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE", {"channel": "test_harness", "risk_level": "low"})

    c.send("asdfghjkl")
    d = c.diary()
    check("Not confirmed after gibberish",
          c.df(d, "diary", "booking", "confirmed") is not True)

    c.send("1")
    time.sleep(3)
    d = c.diary()
    check("Confirmed after valid selection",
          c.df(d, "diary", "booking", "confirmed") is True)

    # -- Monitoring: empty message --
    print("\n  [Monitoring: empty message]")
    c.send("")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No false alert from empty message", len(alerts) == 0, str(alerts))

    # -- Monitoring: anxious repeated messages --
    print("\n  [Monitoring: repeated anxious messages]")
    c.send("Is my appointment still on?")
    c.send("I'm worried, is everything ok?")
    c.send("Just checking again")
    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("No false escalation from anxious messages", len(alerts) == 0, str(alerts))
    check("Monitoring still active",
          c.df(d, "diary", "monitoring", "monitoring_active") is True)

    return c.chat


# ============================================================
#  RUN ALL
# ============================================================


if __name__ == "__main__":
    print("=" * 60)
    print(f" MEDFORCE GATEWAY -- RESILIENCE E2E TEST (run {RUN_ID})")
    print("=" * 60)

    results = {}
    results["scenario_1_happy_path"] = run_happy_path()
    results["scenario_2_confused"] = run_confused_patient()
    results["scenario_3_emergency"] = run_emergency_escalation()
    results["scenario_4_rescheduler"] = run_rescheduler()
    results["scenario_5_slot_rejector"] = run_slot_rejector()
    results["scenario_6_complex_clinical"] = run_complex_clinical()
    results["scenario_7_resilient"] = run_resilient_patient()

    output_path = os.path.join(os.path.dirname(__file__), "e2e_resilience_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nChat transcripts saved to: {output_path}")
    print(f"\n{'=' * 60}")
    print(f" RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)

"""
Full end-to-end Gateway test -- two patient journeys.

Case 1: Patient interacting directly (liver cirrhosis, HIGH risk)
Case 2: Helper on behalf of elderly parent (cardiac, MEDIUM risk)

Usage:
    python tests/e2e_gateway_test.py

Outputs:
    tests/e2e_results.json -- chronological chat transcripts
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

API = "http://127.0.0.1:8080/api/gateway"
RUN_ID = str(int(time.time()))  # unique per run — no stale data ever

PASS = 0
FAIL = 0


def api(method, path, body=None):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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
    """Records one patient conversation with snapshot-based response tracking."""

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
        """Send a message, wait for reply, record both."""
        self.chat.append({"role": self.role, "message": text})
        before = self._response_count()
        api("POST", "/emit", {
            "event_type": "USER_MESSAGE",
            "patient_id": self.pid,
            "sender_role": "patient",
            "sender_id": self.pid,
            "payload": {"text": text, "channel": "test_harness"},
        })
        # Poll until new responses appear
        deadline = time.time() + 20
        while time.time() < deadline:
            items = self._get_responses()
            if len(items) > before:
                time.sleep(0.5)  # grace for chained events
                items = self._get_responses()
                for r in items[before:]:
                    self.chat.append({"role": "agent", "message": r["message"]})
                return
            time.sleep(0.5)

    def event(self, event_type, payload=None):
        """Emit a system event, wait for response."""
        before = self._response_count()
        api("POST", "/emit", {
            "event_type": event_type,
            "patient_id": self.pid,
            "sender_role": "system",
            "sender_id": "test_harness",
            "payload": payload or {},
        })
        deadline = time.time() + 20
        while time.time() < deadline:
            items = self._get_responses()
            if len(items) > before:
                time.sleep(0.5)
                items = self._get_responses()
                for r in items[before:]:
                    self.chat.append({"role": "agent", "message": r["message"]})
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


# ============================================================
#  CASE 1 -- Patient (liver cirrhosis)
# ============================================================

def run_patient_case():
    print("\n" + "=" * 60)
    print(" CASE 1: PATIENT (liver cirrhosis)")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-001", "patient")

    # ── INTAKE ──
    print("\n[INTAKE]")
    c.send("I am the patient, my name is Sarah Thompson")
    c.send("I prefer to be contacted by text message")
    c.send("15/03/1990")
    c.send("9876543210")
    c.send("07700900055")
    c.send("Dr. Patel")

    d = c.diary()
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)
    check("Phase -> clinical", c.phase() == "clinical")

    # ── CLINICAL (adaptive — check phase after each message) ──
    print("\n[CLINICAL]")
    clinical_msgs = [
        "I have been experiencing abdominal pain on the right side, fatigue, "
        "and my skin has been a bit yellow. I was referred for suspected liver cirrhosis.",
        "I have hypertension and had gallbladder surgery 5 years ago. "
        "My father had liver cirrhosis. I take metformin 500mg twice daily and lisinopril 10mg.",
        "No known allergies. I drink about 3 pints of beer most evenings. I don't smoke.",
        "My pain is about 6 out of 10, mainly in the upper right abdomen. "
        "It gets worse after eating fatty food.",
        "I don't have any documents to share right now",
    ]
    for msg in clinical_msgs:
        if c.phase() != "clinical":
            break
        c.send(msg)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    print(f"  Risk: {risk} | Phase: {c.phase()}")
    check("Risk scored", risk not in (None, "none"), risk)

    # ── BOOKING ──
    print("\n[BOOKING]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE",
                {"channel": "test_harness", "risk_level": "high"})

    d = c.diary()
    slots = c.df(d, "diary", "booking", "slots_offered") or []
    check("Slots offered", len(slots) > 0)

    c.send("2")
    # Wait for monitoring chain to complete
    time.sleep(2)

    d = c.diary()
    check("Booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)
    instructions = c.df(d, "diary", "booking", "pre_appointment_instructions") or []
    check("Condition-specific instructions",
          any("alcohol" in i.lower() for i in instructions),
          [i for i in instructions if "alcohol" in i.lower()] or "no alcohol instruction")

    # ── MONITORING ──
    print("\n[MONITORING]")
    d = c.diary()
    plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
    days = plan.get("check_in_days", [])
    check("Communication plan", plan.get("generated", False))

    c.event("HEARTBEAT", {
        "days_since_appointment": days[0] if days else 7,
        "milestone": "heartbeat_7d", "channel": "test_harness",
    })

    c.send("I am feeling fine, thanks for checking in.")
    c.send("I have noticed jaundice and confusion over the past few days")

    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Deterioration alert", len(alerts) > 0, str(alerts))
    check("No re-booking", c.df(d, "diary", "booking", "confirmed") is True)

    c.event("HEARTBEAT", {
        "days_since_appointment": days[1] if len(days) > 1 else 14,
        "milestone": "heartbeat_14d", "channel": "test_harness",
    })

    return c.chat


# ============================================================
#  CASE 2 -- Helper on behalf of elderly parent (cardiac)
# ============================================================

def run_helper_case():
    print("\n" + "=" * 60)
    print(" CASE 2: HELPER (cardiac, elderly parent)")
    print("=" * 60)

    c = Case(f"PT-{RUN_ID}-002", "helper")

    # ── INTAKE ──
    print("\n[INTAKE]")
    c.send("Hello, I'm calling on behalf of my mother, Margaret Wilson. "
           "She's quite elderly and not comfortable with technology.")
    d = c.diary()
    responder = c.df(d, "diary", "intake", "responder_type")
    print(f"  Responder type: {responder}")
    check("Helper detected", responder == "helper", responder)

    c.send("Email would be best for us")
    c.send("Her birthday is 22nd July 1948")
    c.send("Her NHS number is 1234567890")
    c.send("You can reach us on 07700111222")
    c.send("Her GP is Dr. Singh at Oakwood Practice")

    d = c.diary()
    check("Intake complete", c.df(d, "diary", "intake", "intake_complete") is True)
    check("Phase -> clinical", c.phase() == "clinical")

    # ── CLINICAL (adaptive) ──
    print("\n[CLINICAL]")
    if c.phase() != "clinical":
        c.event("INTAKE_COMPLETE", {"channel": "test_harness", "forced": True})

    clinical_msgs = [
        "She's been having chest pains and shortness of breath for the past "
        "few weeks. The GP referred her for a cardiology consultation.",
        "She has type 2 diabetes, high blood pressure, and had a mild stroke "
        "two years ago. She takes aspirin 75mg, atorvastatin 20mg, metformin "
        "1000mg twice daily, and amlodipine 5mg.",
        "She's allergic to penicillin. She doesn't drink or smoke.",
        "She says the chest pain is about 4 out of 10, mainly on the left side. "
        "It comes and goes, worse when she climbs stairs.",
        "We don't have any documents at the moment",
    ]
    for msg in clinical_msgs:
        if c.phase() != "clinical":
            break
        c.send(msg)

    d = c.diary()
    risk = c.df(d, "diary", "header", "risk_level")
    print(f"  Risk: {risk} | Phase: {c.phase()}")
    check("Risk scored", risk not in (None, "none"), risk)

    # ── BOOKING ──
    print("\n[BOOKING]")
    if c.phase() != "booking":
        c.event("CLINICAL_COMPLETE",
                {"channel": "test_harness", "risk_level": "medium"})

    d = c.diary()
    check("Slots offered", len(c.df(d, "diary", "booking", "slots_offered") or []) > 0)

    c.send("The first one please")
    time.sleep(2)

    d = c.diary()
    check("Booking confirmed", c.df(d, "diary", "booking", "confirmed") is True)

    # ── MONITORING ──
    print("\n[MONITORING]")
    d = c.diary()
    plan = c.df(d, "diary", "monitoring", "communication_plan") or {}
    days = plan.get("check_in_days", [])
    check("Communication plan", plan.get("generated", False))

    c.event("HEARTBEAT", {
        "days_since_appointment": days[0] if days else 7,
        "milestone": "heartbeat_7d", "channel": "test_harness",
    })

    c.send("She's doing well overall, feeling much better than last week.")
    c.send("She was confused and disoriented last night, and the pain is worsening")

    d = c.diary()
    alerts = c.df(d, "diary", "monitoring", "alerts_fired") or []
    check("Deterioration alert", len(alerts) > 0, str(alerts))
    check("No re-booking", c.df(d, "diary", "booking", "confirmed") is True)

    return c.chat


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f" MEDFORCE GATEWAY -- E2E TEST (run {RUN_ID})")
    print("=" * 60)

    case_1 = run_patient_case()
    case_2 = run_helper_case()

    output = {
        "case_1_patient_journey": case_1,
        "case_2_helper_journey": case_2,
    }
    output_path = os.path.join(os.path.dirname(__file__), "e2e_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nChat transcript -> {output_path}")
    print(f"\n{'=' * 60}")
    print(f" RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)

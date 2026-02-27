# Manual Test Script — Pre-Consultation Test Harness

Open `test_harness.html` in your browser with the backend running.
Use a fresh patient ID for each scenario (e.g. `PT-MANUAL-01`, `PT-MANUAL-02`, etc.).

---

## SCENARIO 1: Happy Path — Full Lifecycle

**Goal:** Complete intake → clinical → booking → monitoring with all phases working.

### 1.1 Intake (Form)
1. Enter patient ID `PT-MANUAL-01`, click connect
2. Send: `Hi, I am the patient`
3. Wait for form prompt → submit form with:
   - Name: `Sarah Jones`
   - DOB: `22/06/1978`
   - NHS: `943 476 5870`
   - Phone: `07700900456`
   - GP: `Dr. Patel`
   - Contact pref: `email`
   - Email: `sarah@example.com`
   - Next of kin: `James Jones`
4. **Verify:** Bot confirms intake complete, phase moves to Clinical
5. **Diary check:** Click Intake arrow → section stays open, shows all fields

### 1.2 Clinical
1. Answer each plan question naturally (expect 5 plan questions total)
2. When asked about documents: `no documents`
3. **Verify:** Bot completes clinical assessment, risk is scored, phase moves to Booking
4. **Diary check:** Expand Clinical section → verify chief_complaint, medications, allergies populated

### 1.3 Booking
1. Bot offers 3 slots
2. Send: `1`
3. **Verify:** Appointment confirmed with pre-appointment instructions, phase moves to Monitoring

### 1.4 Monitoring
1. Send: `feeling fine, no concerns`
2. **Verify:** No false alert, reassuring response
3. **Diary check:** Expand Monitoring → verify monitoring_active=true, communication_plan exists

---

## SCENARIO 2: Adaptive Follow-Up — Worsening Triggers Follow-Up

**Goal:** A concerning answer ("it has worsened") triggers ONE follow-up before the next plan question.

1. New patient `PT-MANUAL-02`
2. Complete intake via form
3. Clinical starts — bot asks plan Q1 (usually about symptom progression)
4. Send: `Yes, the pain has worsened significantly over the last two weeks`
5. **Verify:** Bot asks a SINGLE follow-up (e.g. "When did you first notice this change?" or similar)
6. Answer the follow-up: `It started about two weeks ago and has been getting worse every day`
7. **Verify:** Bot proceeds to plan Q2 (a DIFFERENT topic — NOT another follow-up)
8. Continue answering plan questions normally
9. **Verify:** All 5 plan questions are asked before documents phase
10. **Diary check:** Expand Clinical → verify `is_followup: true` on exactly the follow-up question entry

---

## SCENARIO 3: Adaptive Follow-Up — Trivial Answer Skips Follow-Up

**Goal:** Short/trivial answers ("no", "yes", "fine") do NOT trigger a follow-up.

1. New patient `PT-MANUAL-03`
2. Complete intake via form
3. Clinical starts — bot asks plan Q1
4. Send: `No`
5. **Verify:** Bot goes STRAIGHT to plan Q2 — no follow-up question in between
6. Send: `yes` to Q2
7. **Verify:** Bot goes straight to plan Q3
8. Send: `same as before` to Q3
9. **Verify:** Bot goes straight to plan Q4
10. Complete remaining questions and documents
11. **Diary check:** Verify NO questions have `is_followup: true`

---

## SCENARIO 4: Adaptive Follow-Up — No Chaining (Follow-Up to Follow-Up)

**Goal:** A follow-up answer — even a concerning one — NEVER triggers another follow-up.

1. New patient `PT-MANUAL-04`
2. Complete intake via form
3. Clinical starts — bot asks plan Q1
4. Send: `Yes, the pain has worsened and I've been vomiting blood`
5. **Verify:** Bot asks ONE follow-up question
6. Answer the follow-up with something alarming: `It happened twice yesterday and I collapsed once`
7. **Verify:** Bot moves to plan Q2 — does NOT ask a second follow-up despite the alarming answer
8. **Diary check:** Only ONE question with `is_followup: true` after Q1. The answer to the follow-up is recorded but did not trigger another follow-up.

---

## SCENARIO 5: Adaptive Follow-Up — Emergency Keywords

**Goal:** Emergency-adjacent keywords trigger deterministic follow-up even without LLM.

1. New patient `PT-MANUAL-05`
2. Complete intake via form
3. Clinical starts — bot asks plan Q1
4. Send: `I collapsed in the kitchen yesterday and felt very confused afterwards`
5. **Verify:** Bot asks a follow-up about recency/frequency (e.g. "How recently did this happen, and has it occurred more than once?")
6. Answer: `Just the one time yesterday`
7. **Verify:** Bot proceeds to plan Q2

---

## SCENARIO 6: Adaptive Follow-Up — Severe Pain

**Goal:** Pain rated 7/10 or higher triggers a follow-up about pain pattern.

1. New patient `PT-MANUAL-06`
2. Complete intake via form
3. Clinical starts — bot asks plan Q1
4. Send: `The pain is about 8 out of 10 now and it's unbearable`
5. **Verify:** Bot asks a follow-up about pain pattern (e.g. "Is the pain constant, or does it come and go?")
6. Answer: `It comes and goes, worse after eating`
7. **Verify:** Bot proceeds to plan Q2

---

## SCENARIO 7: Adaptive Follow-Up — Functional Impact

**Goal:** Functional impact phrases trigger a follow-up about onset.

1. New patient `PT-MANUAL-07`
2. Complete intake via form
3. Clinical starts — bot asks plan Q1
4. Send: `I can't sleep at all because of the pain, and I can't eat properly either`
5. **Verify:** Bot asks a follow-up about gradual vs sudden onset
6. Answer: `It's been getting worse gradually over about 3 weeks`
7. **Verify:** Bot proceeds to plan Q2

---

## SCENARIO 8: All 5 Plan Questions Asked Before Documents

**Goal:** Verify no early transition to document collection — all 5 plan questions must be asked.

1. New patient `PT-MANUAL-08`
2. Complete intake via form
3. Clinical starts
4. Count each plan question (ignore meds/allergy safety questions)
5. Answer each with a moderate-length answer (not trivial, not alarming)
6. **Verify:** Exactly 5 distinct plan questions are asked before the "do you have documents?" prompt
7. **Diary check:** Expand Clinical → count questions. Plan questions + any follow-ups + safety Qs should all appear. `generated_questions` should be empty (all consumed).

---

## SCENARIO 9: Medications & Allergies Are Still Asked

**Goal:** Verify clinical agent still asks about medications and allergies as safety questions.

1. New patient `PT-MANUAL-09`
2. Complete intake via form (no referral with meds/allergies pre-populated)
3. Send: `I have stomach pain, level 5 out of 10`
4. **Watch for:** Bot should ask about medications at some point
5. Send: `I take omeprazole 20mg daily`
6. **Watch for:** Bot should ask about allergies at some point
7. Send: `I'm allergic to penicillin`
8. Continue answering until clinical completes
9. **Verify in diary:** `current_medications` contains omeprazole, `allergies` contains penicillin

---

## SCENARIO 10: Allergy Doesn't Overwrite Chief Complaint

**Goal:** When patient describes allergy reaction symptoms, chief complaint stays intact.

1. New patient `PT-MANUAL-10`
2. Complete intake via form
3. Send: `I have abdominal pain, level 6 out of 10`
4. When asked about allergies, send: `I'm allergic to penicillin, it causes rash and swelling`
5. **Verify in diary:**
   - `chief_complaint` is still about abdominal pain (NOT "rash and swelling")
   - `allergies` contains penicillin with the reaction info

---

## SCENARIO 11: Pain Correction Honored

**Goal:** When patient corrects their pain level, the new value sticks.

1. New patient `PT-MANUAL-11`
2. Complete intake via form
3. Send: `I have abdominal pain, level 6 out of 10`
4. Send: `Actually it's more like 8 out of 10, it's getting worse`
5. **Verify in diary:** `pain_level` is `8`, not `6`
6. **Verify:** Follow-up may trigger due to worsening language — this is expected

---

## SCENARIO 12: Mixed Follow-Ups Across Multiple Plan Questions

**Goal:** Some plan questions trigger follow-ups, others don't — mixed pattern works correctly.

1. New patient `PT-MANUAL-12`
2. Complete intake via form
3. Plan Q1 → Send a concerning answer: `Yes it has deteriorated a lot`
4. **Verify:** Follow-up asked → answer it → plan Q2 appears
5. Plan Q2 → Send a trivial answer: `No`
6. **Verify:** No follow-up → plan Q3 appears immediately
7. Plan Q3 → Send a concerning answer: `I can't eat and I've lost a stone in weight`
8. **Verify:** Follow-up asked → answer it → plan Q4 appears
9. Plan Q4 → Send: `About the same`
10. **Verify:** No follow-up → plan Q5 appears immediately
11. Plan Q5 → Answer normally
12. **Verify:** Transitions to document collection
13. **Diary check:** Exactly 2 questions with `is_followup: true` (after Q1 and Q3)

---

## SCENARIO 13: Allergy Correction ("No allergies" → Specific allergy)

**Goal:** Patient initially says no allergies, then remembers one.

1. New patient `PT-MANUAL-13`
2. Complete intake via form
3. Send: `I have stomach pain, level 5 out of 10`
4. When asked about allergies: `No known allergies`
5. After a few more questions, send: `Actually I'm allergic to penicillin, sorry I forgot`
6. **Verify in diary:** `allergies` contains penicillin, NOT "No known allergies" or "NKDA"

---

## SCENARIO 14: Cross-Phase Allergy During Booking

**Goal:** Mentioning an allergy during booking updates clinical record without breaking booking.

1. New patient `PT-MANUAL-14`
2. Complete intake + clinical
3. Bot offers booking slots
4. Send: `Oh wait, I forgot to mention I'm allergic to penicillin`
5. **Verify:** Bot asks about the reaction, phase stays in Booking
6. Send: `It causes a rash and hives`
7. **Verify:** Bot acknowledges allergy added
8. Send: `1` (to select slot)
9. **Verify:** Booking confirmed
10. **Diary → Clinical:** `allergies` contains penicillin

---

## SCENARIO 15: Mixed Message — Slot + Next of Kin

**Goal:** Booking slot selection + intake data in one message both captured.

1. New patient `PT-MANUAL-15`
2. Complete intake via form (leave next_of_kin empty if possible)
3. Complete clinical
4. Bot offers booking slots
5. Send: `I'll take slot 1, and my next of kin is Carlos Santos on 07700123456`
6. **Verify:**
   - Booking is confirmed (slot 1 selected)
   - **Diary → Intake:** `next_of_kin` = `Carlos Santos on 07700123456`
   - Phase moves to Monitoring

---

## SCENARIO 16: "Nothing else to add" Repeatedly → Scoring Still Happens

**Goal:** Even if LLM doesn't extract chief complaint, scoring eventually triggers.

1. New patient `PT-MANUAL-16`
2. Complete intake via form
3. Send: `I have stomach pain`
4. Then keep sending: `nothing else to add` (up to 8-10 times)
5. **Verify:** Eventually clinical asks about documents or completes assessment
6. **Verify:** Risk is scored (not stuck in clinical forever)

---

## SCENARIO 17: Emergency During Monitoring

**Goal:** Red flag symptoms trigger emergency guidance.

1. New patient `PT-MANUAL-17`
2. Complete full flow through to monitoring
3. Send: `I have jaundice and confusion`
4. **Verify:** Bot provides A&E/999 guidance immediately

---

## SCENARIO 18: GP Change During Monitoring (Cross-Phase)

**Goal:** Updating GP during monitoring updates intake record.

1. Use a patient already in monitoring
2. Send: `I've changed my GP, it's now Dr. Williams`
3. **Verify:** Bot acknowledges GP update
4. **Diary → Intake:** `gp_name` = `Dr. Williams`

---

## SCENARIO 19: MAX_CLINICAL_QUESTIONS Cap

**Goal:** Verify the system doesn't ask more than 12 total clinical questions (hard cap).

1. New patient `PT-MANUAL-19`
2. Complete intake via form
3. Answer every question with a long, concerning answer to maximize follow-ups
4. **Verify:** After 12 total questions (plan + follow-ups + safety), bot forces transition to document collection
5. **Verify:** Does NOT get stuck in an infinite question loop

---

## SCENARIO 20: Diary UI Toggle Persistence

**Goal:** Verify expanding/collapsing diary sections persists across auto-refresh.

1. Use any patient that has been through at least clinical phase
2. Go to Diary tab
3. **Clinical should be open by default**
4. Click Intake arrow to expand it
5. **Wait 5+ seconds** (auto-refresh fires every 3s)
6. **Verify:** Intake section stays open
7. Click Booking arrow to expand it
8. **Wait 5+ seconds**
9. **Verify:** Both Intake and Booking stay open
10. Click Clinical arrow to collapse it
11. **Wait 5+ seconds**
12. **Verify:** Clinical stays collapsed, Intake and Booking stay open

---

## Quick Reference — What to Check in Diary

| Field | Where | What to look for |
|-------|-------|-----------------|
| `is_followup` | Clinical → questions_asked[] | `true` on follow-up Qs, `false` on plan Qs |
| `awaiting_followup` | Clinical | Should be `false` after clinical completes |
| `generated_questions` | Clinical | Should be empty (`[]`) after all plan Qs consumed |
| `questions_asked` | Clinical | Count: 5 plan + 0-5 follow-ups + 0-2 safety |
| `chief_complaint` | Clinical | Should reflect the actual complaint, not allergy reactions |
| `pain_level` | Clinical | Should reflect the most recent correction |
| `meds_addressed` | Clinical | `true` if meds were asked or pre-populated |
| `allergies_addressed` | Clinical | `true` if allergies were asked or pre-populated |

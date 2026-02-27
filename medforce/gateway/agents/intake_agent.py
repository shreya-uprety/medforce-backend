"""
Intake Agent — The Receptionist (Referral-First).

Since patients are GP-referred, the referral letter PDF (pre-uploaded to GCS)
contains most demographic and clinical information.  The intake agent:

  1. Fetches the referral PDF from GCS
  2. Extracts demographics + clinical data in a single Gemini multimodal call
  3. Caches everything into the diary
  4. Sends a personalized hello
  5. Asks only responder type + contact preference
  6. Hands off to Clinical

Fallback: if no PDF is found, reverts to the legacy conversational flow.

Handles:
  - USER_MESSAGE in intake phase
  - NEEDS_INTAKE_DATA: backward loop from Clinical
  - INTAKE_FORM_SUBMITTED: form submission
  - CROSS_PHASE_DATA: cross-phase intake data
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import (
    ConversationEntry,
    PatientDiary,
    Phase,
)
from medforce.gateway.agents.llm_utils import is_response_complete, llm_generate
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.agents.intake")


# Human-friendly labels for each intake field
FIELD_LABELS: dict[str, str] = {
    "name": "full name",
    "dob": "date of birth",
    "nhs_number": "NHS number",
    "address": "home address",
    "phone": "phone number",
    "email": "email address",
    "next_of_kin": "next of kin (name and contact number)",
    "gp_practice": "GP practice name",
    "gp_name": "GP name",
    "contact_preference": "preferred method of contact",
}

# Extraction prompt template (legacy conversational fallback)
EXTRACTION_PROMPT = """\
You are a medical receptionist AI at a UK clinic. Extract patient demographic \
information from the following message. Return a JSON object with only the \
fields you can confidently extract. Valid fields are: {fields}.

If a field is ambiguous or not present, omit it.

For NHS numbers, validate the format (10 digits). \
For phone numbers, accept UK formats (07xxx, +447xxx, etc.). \
For dates of birth, normalise to DD/MM/YYYY.
For contact_preference, valid values are: "email", "sms", "phone", "websocket". \
Map natural language like "text me" → "sms", "call me" → "phone", "email me" → "email".

Message: "{message}"

Return ONLY valid JSON, no markdown, no explanation.\
"""

# Question generation prompt (legacy conversational fallback)
QUESTION_PROMPT = """\
You are a friendly medical receptionist at a UK clinic. Ask the patient for \
their {field} in a warm, professional manner. Keep it to one sentence. \
Do not ask about medical symptoms or history.

IMPORTANT: The following fields have ALREADY been collected — do NOT ask for \
any of them again: {already_collected}. Only ask for: {field}.\
"""

# ── Referral PDF extraction prompt (multimodal — sent with PDF bytes) ──
REFERRAL_ANALYSIS_PROMPT = """\
You are a medical receptionist AI at a UK clinic. A GP referral letter PDF \
is attached. Extract ALL patient demographic AND clinical information into \
a single FLAT JSON object. Do NOT nest fields under section headers.

Return exactly this structure (omit any field you cannot confidently extract):

{
  "name": "patient full name",
  "dob": "DD/MM/YYYY",
  "nhs_number": "10 digits no spaces",
  "phone": "patient phone number",
  "address": "patient home address",
  "email": "patient email",
  "gp_name": "referring GP full name e.g. Dr Sarah Patel",
  "gp_practice": "GP practice or surgery name",
  "next_of_kin": "next of kin name and contact if mentioned",
  "chief_complaint": "main reason for referral, brief",
  "condition_context": "identified or suspected condition e.g. cirrhosis, MASH, hepatitis B",
  "medical_history": ["condition1", "condition2"],
  "current_medications": ["med1 with dose", "med2 with dose"],
  "allergies": ["allergy1", "allergy2"],
  "red_flags": ["concerning symptom or finding"],
  "symptoms": ["symptom1", "symptom2"],
  "lab_values": {"parameter": "value with unit"},
  "key_findings": "brief summary of key clinical findings",
  "clinical_narrative": "A 200-300 word clinical summary written as a triage nurse would read it aloud. Include: the specific diagnosis/condition with subtype if known (e.g. genotype, stage), all investigation results with exact values and units, relevant social history with context (substance use timeline, alcohol quantities, smoking), reason for referral and what is requested (e.g. FibroScan, DAA therapy, surveillance), current symptom status, and any care gaps (e.g. incomplete vaccinations, pending investigations). Write in flowing prose, not bullet points."
}

Rules:
- Return a FLAT JSON object, NOT nested under section names.
- For allergies: use ["NKDA"] if the letter states no known allergies.
- For NHS numbers: 10 digits, no spaces.
- For DOB: normalise to DD/MM/YYYY.
- Return ONLY valid JSON. No markdown fences. No explanation.\
"""


class IntakeAgent(BaseAgent):
    """
    Collects patient demographics via referral-first extraction.

    New flow: PDF extraction → personalized hello → responder type →
    contact preference → complete.

    Fallback: if no referral PDF in GCS, uses legacy conversational intake.
    """

    agent_name = "intake"

    def __init__(self, llm_client=None, gcs_bucket_manager=None) -> None:
        self._client = llm_client
        self._gcs = gcs_bucket_manager
        self._model_name = os.getenv("INTAKE_MODEL", "gemini-2.0-flash")

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            except Exception as exc:
                logger.error("Failed to create Gemini client: %s", exc)
        return self._client

    async def process(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Route to the appropriate handler based on event type."""

        if event.event_type == EventType.NEEDS_INTAKE_DATA:
            return await self._handle_backward_loop(event, diary)

        if event.event_type == EventType.INTAKE_FORM_SUBMITTED:
            return await self._handle_form_submission(event, diary)

        if event.event_type == EventType.CROSS_PHASE_DATA:
            return await self._handle_cross_phase_data(event, diary)

        if event.event_type == EventType.USER_MESSAGE:
            return await self._handle_user_message(event, diary)

        logger.warning(
            "Intake received unexpected event type: %s", event.event_type.value
        )
        return AgentResult(updated_diary=diary)

    # ══════════════════════════════════════════════════════════════
    #  Referral PDF extraction pipeline
    # ══════════════════════════════════════════════════════════════

    def _fetch_referral_pdf(self, patient_id: str) -> bytes | None:
        """Download referral PDF bytes from GCS."""
        if self._gcs is None:
            logger.warning("No GCS bucket manager — cannot fetch referral PDF")
            return None

        blob_path = f"patient_data/{patient_id}/raw_data/referral_letter.pdf"
        try:
            pdf_bytes = self._gcs.read_file_as_bytes(blob_path)
            if pdf_bytes:
                logger.info(
                    "Fetched referral PDF for patient %s (%d bytes)",
                    patient_id, len(pdf_bytes),
                )
            else:
                logger.info("No referral PDF found for patient %s", patient_id)
            return pdf_bytes
        except Exception as exc:
            logger.warning("Failed to fetch referral PDF for %s: %s", patient_id, exc)
            return None

    async def _extract_from_referral(self, pdf_bytes: bytes) -> dict[str, Any]:
        """Send PDF to Gemini multimodal and extract structured data."""
        try:
            if self.client is None:
                return {}

            from google.genai import types as genai_types

            pdf_part = genai_types.Part.from_bytes(
                data=pdf_bytes, mime_type="application/pdf"
            )

            t_start = time.monotonic()
            response = await self.client.aio.models.generate_content(
                model=self._model_name,
                contents=[pdf_part, REFERRAL_ANALYSIS_PROMPT],
            )
            elapsed = time.monotonic() - t_start
            logger.info("  [timing] Referral PDF extraction: %.2fs", elapsed)

            raw = response.text
            if not raw or not raw.strip():
                logger.warning("Gemini returned empty response for referral extraction")
                return {}

            raw = raw.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            extracted = json.loads(raw)

            # Safety: if LLM returned nested structure (e.g. {"DEMOGRAPHIC FIELDS": {...}, ...})
            # flatten it by merging all dict-valued top-level keys
            expected_keys = {
                "name", "dob", "nhs_number", "phone", "address", "email",
                "gp_name", "gp_practice", "next_of_kin", "chief_complaint",
                "condition_context", "medical_history", "current_medications",
                "allergies", "red_flags", "symptoms", "lab_values", "key_findings",
            }
            if not any(k in expected_keys for k in extracted):
                # None of the expected keys found — likely nested
                flattened = {}
                for v in extracted.values():
                    if isinstance(v, dict):
                        flattened.update(v)
                if any(k in expected_keys for k in flattened):
                    logger.info("Flattened nested LLM response into %d fields", len(flattened))
                    extracted = flattened

            logger.info(
                "Referral extraction got %d fields: %s",
                len(extracted), list(extracted.keys()),
            )
            return extracted

        except Exception as exc:
            logger.error("Referral PDF extraction failed: %s", exc)
            return {}

    def _cache_referral_data(
        self, diary: PatientDiary, extracted: dict[str, Any], patient_id: str
    ) -> None:
        """Populate diary.intake + diary.clinical from extracted referral data."""

        # ── Demographic fields → diary.intake ──
        demographic_fields = [
            "name", "dob", "nhs_number", "phone", "address",
            "email", "gp_name", "gp_practice", "next_of_kin",
        ]
        for field in demographic_fields:
            value = extracted.get(field)
            if value and str(value).strip():
                diary.intake.mark_field_collected(field, str(value).strip())
                logger.info(
                    "Referral → intake.%s = '%s' for %s",
                    field, str(value)[:50], patient_id,
                )

        # Set referral letter reference
        diary.intake.referral_letter_ref = (
            f"patient_data/{patient_id}/raw_data/referral_letter.pdf"
        )

        # ── Clinical fields → diary.clinical ──
        if extracted.get("chief_complaint"):
            diary.clinical.chief_complaint = extracted["chief_complaint"]

        if extracted.get("condition_context"):
            diary.clinical.condition_context = extracted["condition_context"]

        if extracted.get("medical_history"):
            for item in extracted["medical_history"]:
                if item and item not in diary.clinical.medical_history:
                    diary.clinical.medical_history.append(item)

        if extracted.get("current_medications"):
            for med in extracted["current_medications"]:
                if med and med not in diary.clinical.current_medications:
                    diary.clinical.current_medications.append(med)

        if extracted.get("allergies"):
            for allergy in extracted["allergies"]:
                if allergy and allergy not in diary.clinical.allergies:
                    diary.clinical.allergies.append(allergy)

        if extracted.get("red_flags"):
            for flag in extracted["red_flags"]:
                if flag and flag not in diary.clinical.red_flags:
                    diary.clinical.red_flags.append(flag)

        # Store clinical narrative for richer question generation
        if extracted.get("clinical_narrative"):
            diary.clinical.referral_narrative = extracted["clinical_narrative"]

        # Store full extraction as referral_analysis for clinical agent
        diary.clinical.referral_analysis = extracted

        # Lab values into referral_analysis (already there via above)
        # Symptoms into red_flags if not already captured
        if extracted.get("symptoms"):
            for symptom in extracted["symptoms"]:
                if symptom and symptom not in diary.clinical.red_flags:
                    # Don't pollute red_flags with general symptoms
                    pass  # symptoms are already in referral_analysis

        logger.info(
            "Cached referral data for %s: intake fields=%s, chief_complaint=%s",
            patient_id,
            [f for f in demographic_fields if extracted.get(f)],
            extracted.get("chief_complaint", "none"),
        )

    # ══════════════════════════════════════════════════════════════
    #  Main handler — referral-first state machine
    # ══════════════════════════════════════════════════════════════

    async def _handle_user_message(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """
        Referral-first state machine:

        State 1 — No referral extracted yet:
            fetch PDF → extract → cache → send personalized hello → wait

        State 2 — Hello sent, awaiting green flag:
            detect affirmative → ask responder type

        State 3 — Green flag acknowledged, awaiting responder type:
            detect patient/helper → record → ask contact preference

        State 4 — Awaiting contact preference:
            detect preference → complete intake

        Fallback: if no PDF found, use legacy conversational intake.

        State tracking via diary fields:
          - referral_letter_ref not set           → State 1
          - referral_letter_ref set, no hello_ack → State 2 (green flag)
          - hello_acknowledged, responder is None → State 3 (responder type)
          - responder set, contact_pref is None   → State 4 (contact pref)
        """
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")
        patient_id = event.patient_id

        # ── State 1: No referral extracted yet ──
        if not diary.intake.referral_letter_ref:
            return await self._state_extract_and_hello(event, diary, text, channel, patient_id)

        hello_acknowledged = "hello_acknowledged" in diary.intake.fields_collected

        # ── State 2: Hello sent, awaiting green flag ──
        if not hello_acknowledged:
            return await self._state_await_green_flag(event, diary, text, channel)

        # ── State 3: Green flag received, awaiting responder type ──
        if diary.intake.responder_type is None:
            return await self._state_await_responder_type(event, diary, text, channel)

        # ── State 4: Awaiting contact preference ──
        if diary.intake.contact_preference is None:
            return await self._state_await_contact_preference(event, diary, text, channel)

        # ── All done — shouldn't normally reach here ──
        missing = diary.intake.get_missing_required()
        if not missing:
            return await self._complete_intake(event, diary, channel)

        # Edge case: required fields still missing after referral (fallback to conversational)
        return await self._handle_user_message_legacy(event, diary, text, channel)

    async def _state_extract_and_hello(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        text: str,
        channel: str,
        patient_id: str,
    ) -> AgentResult:
        """State 1: Fetch PDF, extract, cache, send personalized hello."""

        # Try to fetch and extract from referral PDF
        pdf_bytes = self._fetch_referral_pdf(patient_id)

        if pdf_bytes is None:
            # No referral PDF — fall back to legacy conversational intake
            logger.info("No referral PDF for %s — using legacy conversational intake", patient_id)
            return await self._handle_user_message_legacy(event, diary, text, channel)

        # Extract data from PDF
        extracted = await self._extract_from_referral(pdf_bytes)

        if not extracted:
            # Extraction failed — fall back to legacy
            logger.warning("Referral extraction returned empty for %s — fallback", patient_id)
            return await self._handle_user_message_legacy(event, diary, text, channel)

        # Cache all extracted data into diary
        self._cache_referral_data(diary, extracted, patient_id)

        # Build personalized hello
        full_name = diary.intake.name or ""
        parts = full_name.split() if full_name.strip() else []
        # Strip existing title (Mr, Mrs, Ms, Miss, Dr, Prof, etc.)
        titles = {"mr", "mrs", "ms", "miss", "dr", "prof", "mr.", "mrs.", "ms.", "miss.", "dr.", "prof."}
        while parts and parts[0].lower().rstrip(".") in {t.rstrip(".") for t in titles}:
            parts.pop(0)
        first_name = parts[0] if parts else ""
        name_greeting = f"Mr. {first_name}" if first_name else "there"

        gp_name = diary.intake.gp_name or "your GP"
        # Ensure GP name has "Dr." prefix
        if gp_name != "your GP" and not gp_name.lower().startswith("dr"):
            gp_name = f"Dr. {gp_name}"

        # Use specialty/department for the referral description, not the full condition
        referral_desc = "referral"
        chief = diary.clinical.chief_complaint or ""
        condition = diary.clinical.condition_context or ""
        # Try to derive a short specialty from condition_context or chief_complaint
        _text = f"{condition} {chief}".lower()
        specialty_map = {
            "hepatology": ["hepato", "liver", "cirrhosis", "mash", "nafld", "nash",
                           "hepatitis", "hepatocellular", "jaundice", "bilirubin"],
            "gastroenterology": ["gastro", "ibs", "crohn", "colitis", "bowel",
                                 "stomach", "reflux", "gerd"],
            "cardiology": ["cardiac", "heart", "cardio", "chest pain", "arrhythmia",
                           "hypertension"],
            "neurology": ["neuro", "migraine", "seizure", "epilepsy", "stroke"],
            "respiratory": ["lung", "asthma", "copd", "respiratory", "pulmonary",
                            "breathless"],
            "endocrinology": ["diabetes", "thyroid", "endocrin", "hormonal"],
            "rheumatology": ["arthritis", "rheumat", "lupus", "joint"],
            "oncology": ["cancer", "carcinoma", "tumour", "tumor", "oncol",
                         "malignant", "neoplasm"],
            "urology": ["kidney", "renal", "urolog", "bladder", "prostate"],
            "dermatology": ["skin", "dermat", "eczema", "psoriasis"],
        }
        for specialty, keywords in specialty_map.items():
            if any(kw in _text for kw in keywords):
                referral_desc = f"{specialty} referral"
                break

        hello = (
            f"Hi {name_greeting} \U0001f44b This is Alice (your virtual assistant) from "
            f"MedForce Clinic London.\n"
            f"We just received your {referral_desc} from {gp_name}. "
            f"To get your appointment scheduled, I'll need to ask a few quick "
            f"questions about your basic details and medical history in this chat.\n\n"
            f"Let me know when you have a minute, and we can get started!"
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=hello,
            metadata={"patient_id": event.patient_id},
        )
        return AgentResult(updated_diary=diary, responses=[response])

    async def _state_await_green_flag(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        text: str,
        channel: str,
    ) -> AgentResult:
        """State 2: Awaiting patient green flag → mark acknowledged, ask responder type."""
        text_lower = text.lower().strip()

        # Green-flag detection (affirmative response)
        green_flags = [
            "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead",
            "ready", "let's go", "lets go", "go on", "start", "begin",
            "alright", "fine", "cool", "sounds good", "of course", "absolutely",
            "i'm ready", "im ready", "go for it", "fire away", "shoot",
            "i have a minute", "i'm free", "right now", "now is good",
            "hi", "hello", "hey",
        ]
        is_green = any(gf in text_lower for gf in green_flags)

        if not is_green:
            # Not clearly affirmative — gently re-prompt
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message="No worries — just let me know when you're ready and we'll get started!",
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # Green flag received — mark acknowledged and ask responder type
        if "hello_acknowledged" not in diary.intake.fields_collected:
            diary.intake.fields_collected.append("hello_acknowledged")

        ask_responder = AgentResponse(
            recipient="patient",
            channel=channel,
            message="Are you the patient, or are you helping someone with their registration?",
            metadata={"patient_id": event.patient_id},
        )
        return AgentResult(updated_diary=diary, responses=[ask_responder])

    async def _state_await_responder_type(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        text: str,
        channel: str,
    ) -> AgentResult:
        """State 3: Detect responder type → record → ask contact preference."""
        responder = self._detect_responder_type_from_text(text)

        if responder:
            diary.intake.responder_type = responder
            logger.info("Responder type = '%s' for %s", responder, event.patient_id)

            if responder == "helper":
                # Try to extract helper details from same message
                helper_info = self._detect_responder(text)
                if helper_info:
                    diary.intake.responder_name = helper_info.get("name")
                    diary.intake.responder_relationship = helper_info.get("relationship")

            # Ask contact preference
            ask_pref = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "How would you prefer us to keep in touch? "
                    "We can contact you via email, text message, phone call, "
                    "or continue through this chat."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[ask_pref])
        else:
            # Couldn't detect — re-ask
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "Sorry, I didn't quite catch that. "
                    "Are you the patient, or are you helping someone with their registration?"
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

    async def _state_await_contact_preference(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        text: str,
        channel: str,
    ) -> AgentResult:
        """State 4: Detect contact preference → complete intake."""
        text_lower = text.lower().strip()

        pref = self._detect_contact_preference(text_lower)
        if pref:
            diary.intake.mark_field_collected("contact_preference", pref)
            logger.info("Contact preference = '%s' for %s", pref, event.patient_id)

            # Check if all required fields are satisfied
            missing = diary.intake.get_missing_required()
            if not missing:
                return await self._complete_intake(event, diary, channel)
            else:
                # Still missing some required fields — fall back to conversational
                return await self._handle_user_message_legacy(event, diary, text, channel)
        else:
            # Didn't detect preference — re-ask
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "How would you prefer us to keep in touch? "
                    "We can contact you via email, text message (SMS), phone call, "
                    "or continue through this chat."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

    # ══════════════════════════════════════════════════════════════
    #  Legacy conversational intake (fallback when no referral PDF)
    # ══════════════════════════════════════════════════════════════

    async def _handle_user_message_legacy(
        self, event: EventEnvelope, diary: PatientDiary, text: str, channel: str
    ) -> AgentResult:
        """
        Legacy adaptive conversation loop (used when no referral PDF exists):
          1. If no responder identified → ask patient/helper
          2. Extract fields from message
          3. Evaluate what's still missing → ask or complete
        """
        # ── Step 1: Responder identification (first interaction) ──
        if diary.intake.responder_type is None:
            responder = self._detect_responder(text)
            if responder:
                diary.intake.responder_type = responder["type"]
                if responder["type"] == "helper":
                    diary.intake.responder_name = responder.get("name")
                    diary.intake.responder_relationship = responder.get("relationship")

                # Opportunistically extract fields from the first message
                missing = diary.intake.fields_missing
                if missing and text:
                    first_extracted = self._fallback_extraction(text, missing)
                    for field, value in first_extracted.items():
                        if hasattr(diary.intake, field) and value:
                            diary.intake.mark_field_collected(field, str(value))

                # After detecting responder, prompt for form
                if responder["type"] == "helper":
                    greeting = (
                        "Thank you for helping with the registration. "
                        "I'll now show you a form to fill in the patient's details. "
                        "[SHOW_INTAKE_FORM]"
                    )
                else:
                    greeting = (
                        "Welcome! Thank you for confirming. "
                        "I'll now show you a form to fill in your details. "
                        "[SHOW_INTAKE_FORM]"
                    )
                response = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=greeting,
                    metadata={"patient_id": event.patient_id},
                )
                return AgentResult(updated_diary=diary, responses=[response])
            else:
                # First message — send welcome + patient/helper question as two messages
                welcome = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=(
                        "Welcome to The London Clinic. I'm your pre-consultation "
                        "assistant and I'll be helping you prepare for your upcoming "
                        "appointment. I'll collect some details, ask a few clinical "
                        "questions, and then help you book your consultation."
                    ),
                    metadata={"patient_id": event.patient_id},
                )
                ask_role = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=(
                        "Before we begin, are you the patient, or are you helping "
                        "someone with their registration?"
                    ),
                    metadata={"patient_id": event.patient_id},
                )
                return AgentResult(updated_diary=diary, responses=[welcome, ask_role])

        # ── Step 2: Extract fields + speculative question gen in parallel ──
        if text:
            import asyncio as _aio

            # Determine speculative next field BEFORE extraction
            missing_before = diary.intake.get_missing_required()
            speculative_field = (
                self._select_next_field(missing_before, diary)
                if len(missing_before) > 1
                else None
            )

            # Run both LLM calls in parallel when possible
            if speculative_field:
                extracted, speculative_question = await _aio.gather(
                    self._extract_fields(text, diary),
                    self._generate_question(speculative_field, diary),
                )
            else:
                extracted = await self._extract_fields(text, diary)
                speculative_question = None

            for field, value in extracted.items():
                if hasattr(diary.intake, field) and value:
                    diary.intake.mark_field_collected(field, str(value))
                    logger.info(
                        "Extracted '%s' = '%s' for patient %s",
                        field, value, event.patient_id,
                    )
        else:
            extracted = {}
            speculative_field = None
            speculative_question = None

        # ── Step 3: Referral cross-reference (once) ──
        if (
            diary.intake.referral_letter_ref
            and "referral_analysed" not in diary.intake.fields_collected
        ):
            await self._cross_reference_referral(diary)

        # ── Step 4: Evaluate and decide next action ──
        missing = diary.intake.get_missing_required()

        if not missing:
            return await self._complete_intake(event, diary, channel)

        # Reuse speculative question if the next field hasn't changed
        actual_field = self._select_next_field(missing, diary)
        if actual_field == speculative_field and speculative_question:
            question = speculative_question
        else:
            question = await self._generate_question(actual_field, diary)

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=question,
            metadata={"patient_id": event.patient_id, "asking_for": actual_field},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    # ══════════════════════════════════════════════════════════════
    #  Backward loop & other handlers (unchanged)
    # ══════════════════════════════════════════════════════════════

    async def _handle_backward_loop(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Clinical Agent sent NEEDS_INTAKE_DATA — ask for specific fields."""
        missing_fields = event.payload.get("missing_fields", [])
        channel = event.payload.get("channel", "websocket")

        diary.header.current_phase = Phase.INTAKE
        diary.clinical.backward_loop_count += 1

        for field in missing_fields:
            if field not in diary.intake.fields_missing:
                diary.intake.fields_missing.append(field)

        logger.info(
            "Backward loop #%d for patient %s: requesting %s",
            diary.clinical.backward_loop_count,
            event.patient_id,
            missing_fields,
        )

        if diary.clinical.backward_loop_count > 3:
            logger.warning(
                "Backward loop limit reached for patient %s — proceeding",
                event.patient_id,
            )
            return await self._complete_intake(event, diary, channel, forced=True)

        if missing_fields:
            field = missing_fields[0]
            label = FIELD_LABELS.get(field, field)
            question = (
                f"I'm sorry to ask again, but the clinical team needs your "
                f"{label} to proceed with your assessment. Could you please "
                f"provide it?"
            )
        else:
            question = (
                "The clinical team has requested some additional information. "
                "A team member will follow up with specific questions shortly."
            )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=question,
            metadata={"patient_id": event.patient_id, "backward_loop": True},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    # ── Form Submission Handler ──

    async def _handle_form_submission(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Handle INTAKE_FORM_SUBMITTED — all fields submitted at once via form."""
        payload = event.payload
        channel = payload.get("channel", "websocket")

        # Set responder type from form
        is_helper = payload.get("is_helper", False)
        if is_helper:
            diary.intake.responder_type = "helper"
        elif diary.intake.responder_type is None:
            diary.intake.responder_type = "patient"

        # Field mapping: form field name → diary field name
        field_map = {
            "name": "name",
            "dob": "dob",
            "nhs_number": "nhs_number",
            "phone": "phone",
            "gp_name": "gp_name",
            "contact_preference": "contact_preference",
            "email": "email",
            "address": "address",
            "next_of_kin": "next_of_kin",
            "gp_practice": "gp_practice",
        }

        # Validate and collect fields
        validation_errors = []
        for form_field, diary_field in field_map.items():
            value = payload.get(form_field)
            if value and str(value).strip():
                value_str = str(value).strip()

                # Basic validation
                if diary_field == "nhs_number":
                    clean = value_str.replace(" ", "")
                    if not clean.isdigit() or len(clean) != 10:
                        validation_errors.append(f"NHS number must be 10 digits (got: {value_str})")
                        continue
                    value_str = clean
                elif diary_field == "phone":
                    if not re.match(r'^(\+?44|0)7\d[\d\s\-]{8,}$', value_str.replace(" ", "")):
                        # Relaxed validation — accept if it looks phone-like
                        if not any(c.isdigit() for c in value_str):
                            validation_errors.append(f"Phone number format not recognised: {value_str}")
                            continue

                diary.intake.mark_field_collected(diary_field, value_str)
                logger.info(
                    "Form field '%s' = '%s' for patient %s",
                    diary_field, value_str[:50], event.patient_id,
                )

        # Check if required fields are satisfied
        missing = diary.intake.get_missing_required()

        if missing:
            # Some required fields missing — ask via chat
            missing_labels = [FIELD_LABELS.get(f, f) for f in missing]
            msg = (
                f"Thank you for filling in the form. However, we still need the following: "
                f"{', '.join(missing_labels)}. Could you please provide these?"
            )
            if validation_errors:
                msg += f"\n\nAlso: {'; '.join(validation_errors)}"
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=msg,
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # All required fields present — complete intake
        return await self._complete_intake(event, diary, channel)

    # ── Cross-Phase Data Handler ──

    async def _handle_cross_phase_data(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Handle cross-phase intake data detected by the Gateway."""
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")
        from_phase = event.payload.get("from_phase", "unknown")

        # Cross-phase: extract ALL recognizable intake fields, not just missing ones
        all_intake_fields = [
            "name", "dob", "nhs_number", "address", "phone", "email",
            "next_of_kin", "gp_practice", "gp_name", "contact_preference",
        ]
        extracted = self._fallback_extraction(text, all_intake_fields)
        # Also try LLM extraction if available
        try:
            if self.client is not None:
                prompt = EXTRACTION_PROMPT.format(fields=", ".join(all_intake_fields), message=text)
                raw_response = await llm_generate(self.client, self._model_name, prompt)
                if raw_response:
                    raw = raw_response.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()
                    llm_extracted = json.loads(raw)
                    llm_extracted = {k: v for k, v in llm_extracted.items() if k in all_intake_fields and v}
                    extracted = {**extracted, **llm_extracted}
        except Exception as exc:
            logger.warning("Cross-phase LLM extraction failed: %s", exc)
        if not extracted:
            return AgentResult(updated_diary=diary)

        # Apply to diary
        for field, value in extracted.items():
            if hasattr(diary.intake, field) and value:
                diary.intake.mark_field_collected(field, str(value))
                logger.info(
                    "Cross-phase intake field '%s' = '%s' for patient %s",
                    field, str(value)[:50], event.patient_id,
                )

        # Build audit trail entry
        from datetime import datetime, timezone
        diary.cross_phase_extractions.append({
            "from_phase": from_phase,
            "to_agent": "intake",
            "text_snippet": text[:100],
            "extracted_fields": list(extracted.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Build acknowledgment
        field_labels = [FIELD_LABELS.get(f, f) for f in extracted.keys()]
        ack_msg = f"I've updated your {', '.join(field_labels)} on file."

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=ack_msg,
            metadata={"patient_id": event.patient_id, "cross_phase": True},
        )

        logger.info(
            "Cross-phase intake extraction from %s for patient %s: %s",
            from_phase, event.patient_id, list(extracted.keys()),
        )

        # Emit CROSS_PHASE_REPROMPT to return control to the pending phase
        reprompt = EventEnvelope.handoff(
            event_type=EventType.CROSS_PHASE_REPROMPT,
            patient_id=event.patient_id,
            source_agent="intake",
            payload={"_pending_phase": from_phase, "channel": channel},
            correlation_id=event.correlation_id,
        )

        # DO NOT change the phase
        return AgentResult(
            updated_diary=diary, responses=[response], emitted_events=[reprompt],
        )

    # ══════════════════════════════════════════════════════════════
    #  Detection helpers
    # ══════════════════════════════════════════════════════════════

    def _detect_responder(self, text: str) -> dict[str, str] | None:
        """Detect whether the sender is the patient or a helper."""
        text_lower = text.lower().strip()

        # Helper indicators — checked FIRST because they are more specific
        helper_keywords = [
            "on behalf", "for my", "calling for",
            "my husband", "my wife", "my mother", "my father",
            "my mum", "my dad", "my daughter", "my son", "my partner",
            "helping", "helper", "carer", "family member", "relative", "spouse",
        ]
        for kw in helper_keywords:
            if kw in text_lower:
                relationship = ""
                rel_map = {
                    "husband": "spouse", "wife": "spouse", "partner": "spouse",
                    "mother": "parent", "father": "parent", "mum": "parent", "dad": "parent",
                    "daughter": "child", "son": "child",
                    "carer": "carer", "helper": "helper",
                }
                for word, rel in rel_map.items():
                    if word in text_lower:
                        relationship = rel
                        break
                return {"type": "helper", "relationship": relationship or "helper"}

        # Patient indicators
        patient_keywords = [
            "i am the patient", "i'm the patient", "it's me", "this is me",
            "myself", "i am", "my name is", "i'm", "patient",
        ]
        for kw in patient_keywords:
            if kw in text_lower:
                return {"type": "patient"}

        # If the message looks like a name (2-5 words, no sentence indicators)
        words = text.strip().split()
        sentence_indicators = {
            "i", "the", "is", "am", "hi", "hello", "hey",
            "been", "have", "was", "please", "can", "yes", "no",
        }
        has_sentence_words = any(w.lower() in sentence_indicators for w in words)
        if 2 <= len(words) <= 5 and not has_sentence_words and not any(c.isdigit() for c in text):
            return {"type": "patient"}

        return None

    def _detect_contact_preference(self, text_lower: str) -> str | None:
        """Detect contact preference from user message."""
        pref_map = {
            "email": ["email", "e-mail", "email me"],
            "sms": ["sms", "text", "text me", "message me", "whatsapp"],
            "phone": ["call", "phone", "ring", "call me", "telephone"],
            "websocket": ["chat", "this", "here", "online", "web", "this chat"],
        }
        for pref, keywords in pref_map.items():
            if any(kw in text_lower for kw in keywords):
                return pref
        return None

    def _detect_responder_type_from_text(self, text: str) -> str | None:
        """Detect responder type from free text answer."""
        text_lower = text.lower().strip()

        # Helper indicators
        helper_keywords = [
            "helping", "helper", "on behalf", "for my", "carer",
            "family", "relative", "spouse", "husband", "wife",
            "mother", "father", "mum", "dad", "son", "daughter", "partner",
        ]
        for kw in helper_keywords:
            if kw in text_lower:
                return "helper"

        # Patient indicators
        patient_keywords = [
            "i am the patient", "i'm the patient", "the patient",
            "it's me", "myself", "me", "i am", "patient",
        ]
        for kw in patient_keywords:
            if kw in text_lower:
                return "patient"

        return None

    # ── Referral Cross-Reference (legacy) ──

    async def _cross_reference_referral(self, diary: PatientDiary) -> None:
        """Analyze referral letter and pre-fill available fields."""
        try:
            if self.client is None:
                return

            referral_ref = diary.intake.referral_letter_ref

            if diary.intake.gp_name and "gp_name" not in diary.intake.fields_collected:
                diary.intake.mark_field_collected("gp_name", diary.intake.gp_name)

            diary.intake.fields_collected.append("referral_analysed")
            logger.info("Referral cross-reference completed for %s", diary.header.patient_id)

        except Exception as exc:
            logger.warning("Referral cross-reference failed: %s", exc)

    # ── Adaptive Field Selection ──

    def _select_next_field(self, missing: list[str], diary: PatientDiary) -> str:
        """Adaptively select which field to ask for next based on context."""
        priority = [
            "name", "contact_preference", "dob", "nhs_number",
            "phone", "gp_name", "address", "email",
            "next_of_kin", "gp_practice",
        ]
        for field in priority:
            if field in missing:
                return field
        return missing[0]

    # ── Completion ──

    async def _complete_intake(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        channel: str,
        forced: bool = False,
    ) -> AgentResult:
        """All required fields collected — advance to Clinical phase."""
        diary.intake.intake_complete = True
        diary.header.current_phase = Phase.CLINICAL

        logger.info(
            "Intake complete for patient %s%s",
            event.patient_id,
            " (forced — missing data)" if forced else "",
        )

        if forced:
            message = (
                "Thank you for your patience. We have enough information to "
                "proceed with your clinical assessment. A specialist will "
                "review your details shortly."
            )
        else:
            name = diary.intake.name or "there"
            pref = diary.intake.contact_preference or "this chat"
            message = (
                f"Thank you, {name}! I've collected all the information we need. "
                f"We've noted your preference to be contacted via {pref}. "
                f"You'll now be connected with our clinical team for your "
                f"pre-consultation assessment."
            )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id},
        )

        handoff = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id=event.patient_id,
            source_agent="intake",
            payload={
                "forced": forced,
                "fields_collected": list(diary.intake.fields_collected),
                "fields_missing": list(diary.intake.fields_missing),
                "responder_type": diary.intake.responder_type,
                "contact_preference": diary.intake.contact_preference,
                "channel": channel,
            },
            correlation_id=event.correlation_id,
        )

        return AgentResult(
            updated_diary=diary,
            emitted_events=[handoff],
            responses=[response],
        )

    # ══════════════════════════════════════════════════════════════
    #  LLM Integration (legacy conversational intake)
    # ══════════════════════════════════════════════════════════════

    async def _extract_fields(
        self, text: str, diary: PatientDiary
    ) -> dict[str, Any]:
        """Use LLM to extract demographic fields from free text."""
        missing = diary.intake.fields_missing
        if not missing:
            missing = diary.intake.get_missing_required()
            if not missing:
                return {}

        # Always run pattern-based extraction first
        fallback = self._fallback_extraction(text, missing)

        try:
            if self.client is None:
                return fallback

            prompt = EXTRACTION_PROMPT.format(
                fields=", ".join(missing),
                message=text,
            )

            t_llm = time.monotonic()
            raw_response = await llm_generate(self.client, self._model_name, prompt)
            logger.info("  [timing] LLM extraction call: %.2fs", time.monotonic() - t_llm)

            if raw_response is None:
                return fallback

            raw = raw_response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            llm_extracted = json.loads(raw)
            llm_extracted = {k: v for k, v in llm_extracted.items() if k in missing and v}

            merged = {**fallback, **llm_extracted}
            return merged

        except Exception as exc:
            logger.warning("LLM extraction failed: %s — using fallback", exc)
            return fallback

    def _fallback_extraction(self, text: str, missing: list[str]) -> dict[str, Any]:
        """Simple pattern-based extraction as fallback."""
        extracted: dict[str, Any] = {}
        text_stripped = text.strip()
        text_lower = text_stripped.lower()

        # Contact preference detection
        if "contact_preference" in missing:
            pref = self._detect_contact_preference(text_lower)
            if pref:
                extracted["contact_preference"] = pref

        # NHS number: exactly 10 digits
        if "nhs_number" in missing:
            nhs_match = re.search(r'\b(\d{3}\s?\d{3}\s?\d{4})\b', text_stripped)
            if nhs_match:
                extracted["nhs_number"] = nhs_match.group(1).replace(" ", "")

        # Phone: UK mobile patterns
        if "phone" in missing and "nhs_number" not in extracted:
            phone_match = re.search(r'(\+?44\d[\d\s\-]{8,12}|07\d[\d\s\-]{8,11})', text_stripped)
            if phone_match:
                extracted["phone"] = phone_match.group(1).strip()

        # Email
        if "email" in missing:
            email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text_stripped)
            if email_match:
                extracted["email"] = email_match.group(0)

        # DOB
        if "dob" in missing:
            dob_match = re.search(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b', text_stripped)
            if dob_match:
                extracted["dob"] = dob_match.group(1)

        # GP name
        if "gp_name" in missing:
            gp_match = re.search(r'((?:Dr\.?|Doctor)\s+[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)', text_stripped)
            if gp_match:
                extracted["gp_name"] = gp_match.group(1).strip()

        # GP practice
        if "gp_practice" in missing:
            practice_kw = re.search(
                r'(.{2,50}(?:practice|surgery|medical\s+centre|health\s+centre|clinic))',
                text_lower,
            )
            if practice_kw:
                extracted["gp_practice"] = text_stripped[:len(practice_kw.group(0))].strip()

        # Next of kin / emergency contact
        if "next_of_kin" in missing:
            nok_patterns = [
                r'next[\s\-]+of[\s\-]+kin\s+(?:is|:|-|—)\s*(.+?)(?:\s*$)',
                r'emergency\s+contact\s+(?:is|:|-|—)\s*(.+?)(?:\s*$)',
                r'(?:my\s+)?\bkin\b\s+is\s+(.+?)(?:\s*$)',
            ]
            for pattern in nok_patterns:
                nok_match = re.search(pattern, text_stripped, re.IGNORECASE)
                if nok_match:
                    nok_value = nok_match.group(1).strip()
                    nok_value = re.sub(r'\s*,?\s*(?:slot|book|appointment).*$', '', nok_value, flags=re.IGNORECASE).strip()
                    if nok_value:
                        extracted["next_of_kin"] = nok_value
                    break

        # Name: 2-5 words, no digits, no sentence words
        if "name" in missing and "name" not in extracted and not any(c.isdigit() for c in text_stripped):
            words = text_stripped.split()
            sentence_indicators = {
                "i", "my", "the", "is", "am", "hi", "hello", "hey",
                "i'm", "i've", "been", "have", "was", "please", "can",
                "could", "would", "yes", "no", "not", "it", "this",
                "that", "a", "an", "to", "for", "and", "or", "but",
            }
            has_sentence_words = any(w.lower() in sentence_indicators for w in words)
            looks_like_name = (
                2 <= len(words) <= 5
                and not has_sentence_words
                and not any(c in text_stripped for c in ".,!?;:'\"")
            )
            if looks_like_name:
                extracted["name"] = text_stripped

        return extracted

    async def _generate_question(self, field: str, diary: PatientDiary) -> str:
        """Generate a friendly, context-aware question for a specific field."""
        label = FIELD_LABELS.get(field, field)

        # Add context for contact preference
        if field == "contact_preference":
            return (
                "How would you prefer us to keep in touch with you? "
                "We can contact you via email, text message (SMS), phone call, "
                "or continue through this chat."
            )

        try:
            if self.client is None:
                return self._fallback_question(field)

            # Build context-aware prompt
            name = diary.intake.name
            context = ""
            if name:
                context = f"The patient's name is {name}. "
            if diary.intake.responder_type == "helper":
                context += "You are speaking to a helper, not the patient directly. "

            already = ", ".join(diary.intake.fields_collected) or "none"
            prompt = QUESTION_PROMPT.format(
                field=label, already_collected=already
            ) + f"\n\n{context}"

            t_llm = time.monotonic()
            raw = await llm_generate(self.client, self._model_name, prompt)
            logger.info("  [timing] LLM question gen call: %.2fs", time.monotonic() - t_llm)

            if raw and is_response_complete(raw.strip()):
                return raw.strip()
            if raw:
                logger.warning("LLM question response appears truncated — using fallback")

        except Exception as exc:
            logger.warning("LLM question generation failed: %s — using fallback", exc)

        return self._fallback_question(field)

    def _fallback_question(self, field: str) -> str:
        """Static question templates when LLM is unavailable."""
        templates = {
            "name": "Hello! Welcome to MedForce. Could you please tell me your full name?",
            "dob": "Thank you! Could you please provide your date of birth?",
            "nhs_number": "Could you please provide your NHS number? It's a 10-digit number found on any letter from the NHS.",
            "address": "Could you share your home address, please?",
            "phone": "What's the best phone number to reach you on?",
            "email": "Could you provide your email address?",
            "next_of_kin": "Could you tell me the name and contact number of your next of kin?",
            "gp_practice": "Which GP practice are you registered with?",
            "gp_name": "And what is your GP's name?",
            "contact_preference": (
                "How would you prefer us to keep in touch? "
                "We can use email, text message, phone, or this chat."
            ),
        }
        return templates.get(field, f"Could you please provide your {FIELD_LABELS.get(field, field)}?")

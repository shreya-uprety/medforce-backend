"""
Intake Agent — The Receptionist.

Collects patient demographics through an adaptive conversation loop.
Never asks clinical questions (symptoms, medications — that's Clinical's job).

Flow (adaptive, not scripted):
  1. Identify responder → patient or helper?
  2. If referral letter exists → cross-reference to pre-fill known fields
  3. Collect missing demographics one at a time (LLM-powered extraction)
  4. Collect preferred contact method (email, SMS, phone, websocket)
  5. Confirm all data → emit INTAKE_COMPLETE

Handles:
  - USER_MESSAGE in intake phase: extract data, ask for next missing field
  - NEEDS_INTAKE_DATA: backward loop from Clinical requesting specific fields
  - INTAKE_COMPLETE emission when all required fields are collected

Uses Gemini Flash for natural-language extraction and question generation.
"""

from __future__ import annotations

import json
import logging
import os
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

# Extraction prompt template
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

# Question generation prompt
QUESTION_PROMPT = """\
You are a friendly medical receptionist at a UK clinic. Ask the patient for \
their {field} in a warm, professional manner. Keep it to one sentence. \
Do not ask about medical symptoms or history.

IMPORTANT: The following fields have ALREADY been collected — do NOT ask for \
any of them again: {already_collected}. Only ask for: {field}.\
"""

# Referral cross-reference prompt
REFERRAL_ANALYSIS_PROMPT = """\
You are a medical receptionist AI. A referral letter has been provided. \
Extract any patient demographic information you can find.

Return a JSON object with these possible fields:
- name: patient's full name
- dob: date of birth (normalise to DD/MM/YYYY)
- nhs_number: NHS number (10 digits)
- address: home address
- phone: phone number
- gp_name: referring GP's name
- gp_practice: GP practice name
- chief_complaint: main reason for referral (brief)
- medical_history: list of conditions mentioned
- current_medications: list of medications mentioned

Only include fields you can confidently extract.
Return ONLY valid JSON, no markdown.\
"""


class IntakeAgent(BaseAgent):
    """
    Collects patient demographics through adaptive conversation.

    The agent operates in a loop: extract → evaluate → ask next.
    It adapts based on what information is already available (e.g. from
    referral letters) rather than following a fixed script.
    """

    agent_name = "intake"

    def __init__(self, llm_client=None) -> None:
        self._client = llm_client
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

        if event.event_type == EventType.USER_MESSAGE:
            return await self._handle_user_message(event, diary)

        logger.warning(
            "Intake received unexpected event type: %s", event.event_type.value
        )
        return AgentResult(updated_diary=diary)

    # ── Handlers ──

    async def _handle_user_message(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """
        Adaptive conversation loop:
          1. If no responder identified → ask patient/helper
          2. Extract fields from message
          3. Cross-reference referral if available and not yet done
          4. Evaluate what's still missing → ask or complete
        """
        text = event.payload.get("text", "")
        channel = event.payload.get("channel", "websocket")

        # ── Step 1: Responder identification (first interaction) ──
        if diary.intake.responder_type is None:
            responder = self._detect_responder(text)
            if responder:
                diary.intake.responder_type = responder["type"]
                if responder["type"] == "helper":
                    diary.intake.responder_name = responder.get("name")
                    diary.intake.responder_relationship = responder.get("relationship")
                # Continue to extract any demographic data from the same message
            else:
                # First message and unclear — ask directly
                response = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=(
                        "Welcome to MedForce. Before we begin, are you the patient, "
                        "or are you helping someone with their registration? "
                        "Please let us know so we can assist you properly."
                    ),
                    metadata={"patient_id": event.patient_id},
                )
                return AgentResult(updated_diary=diary, responses=[response])

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

    # ── Responder Identification ──

    def _detect_responder(self, text: str) -> dict[str, str] | None:
        """Detect whether the sender is the patient or a helper."""
        text_lower = text.lower().strip()

        # Helper indicators — checked FIRST because they are more specific
        # (e.g. "I'm calling on behalf of my mother" contains "i'm" but is a helper)
        helper_keywords = [
            "on behalf", "for my", "calling for",
            "my husband", "my wife", "my mother", "my father",
            "my mum", "my dad", "my daughter", "my son", "my partner",
            "helping", "helper", "carer", "family member", "relative", "spouse",
        ]
        for kw in helper_keywords:
            if kw in text_lower:
                # Try to extract relationship
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

        # If the message looks like a name (2-5 words, no sentence indicators),
        # assume they're the patient introducing themselves
        words = text.strip().split()
        sentence_indicators = {
            "i", "the", "is", "am", "hi", "hello", "hey",
            "been", "have", "was", "please", "can", "yes", "no",
        }
        has_sentence_words = any(w.lower() in sentence_indicators for w in words)
        if 2 <= len(words) <= 5 and not has_sentence_words and not any(c.isdigit() for c in text):
            return {"type": "patient"}

        return None

    # ── Referral Cross-Reference ──

    async def _cross_reference_referral(self, diary: PatientDiary) -> None:
        """Analyze referral letter and pre-fill available fields."""
        try:
            if self.client is None:
                return

            # In production, we'd fetch the referral content from storage.
            # For now, mark as analysed and extract what we can from context.
            referral_ref = diary.intake.referral_letter_ref

            # If GP name came from referral, auto-fill
            if diary.intake.gp_name and "gp_name" not in diary.intake.fields_collected:
                diary.intake.mark_field_collected("gp_name", diary.intake.gp_name)

            diary.intake.fields_collected.append("referral_analysed")
            logger.info("Referral cross-reference completed for %s", diary.header.patient_id)

        except Exception as exc:
            logger.warning("Referral cross-reference failed: %s", exc)

    # ── Adaptive Field Selection ──

    def _select_next_field(self, missing: list[str], diary: PatientDiary) -> str:
        """
        Adaptively select which field to ask for next based on context.

        Priority order:
          1. contact_preference (if not collected — we need to know how to reach them)
          2. name (need to address them)
          3. dob (clinical priority)
          4. nhs_number (lookup priority)
          5. phone (contact backup)
          6. gp_name (for clinical handoff)
          7. everything else
        """
        priority = [
            "contact_preference", "name", "dob", "nhs_number",
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

    # ── LLM Integration ──

    async def _extract_fields(
        self, text: str, diary: PatientDiary
    ) -> dict[str, Any]:
        """Use LLM to extract demographic fields from free text."""
        missing = diary.intake.fields_missing
        if not missing:
            # Also check required fields not yet collected
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
        import re

        extracted: dict[str, Any] = {}
        text_stripped = text.strip()
        text_lower = text_stripped.lower()

        # Contact preference detection
        if "contact_preference" in missing:
            pref_map = {
                "email": ["email", "e-mail", "email me"],
                "sms": ["sms", "text", "text me", "message me", "whatsapp"],
                "phone": ["call", "phone", "ring", "call me", "telephone"],
                "websocket": ["chat", "this", "here", "online", "web"],
            }
            for pref, keywords in pref_map.items():
                if any(kw in text_lower for kw in keywords):
                    extracted["contact_preference"] = pref
                    break

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

        # Name: 2-5 words, no digits, no sentence words
        if "name" in missing and not extracted and not any(c.isdigit() for c in text_stripped):
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

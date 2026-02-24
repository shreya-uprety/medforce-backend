"""
Booking Agent — The Scheduler.

Gets patients into consultation appointments based on their risk-determined
urgency window:
  - HIGH risk:   within 2 days
  - MEDIUM risk: within 14 days
  - LOW risk:    within 30 days

Handles:
  - CLINICAL_COMPLETE: Begin booking flow
  - USER_MESSAGE (booking phase): Process slot selection
  - RESCHEDULE_REQUEST: Cancel existing booking and re-offer slots
  - Fires BOOKING_COMPLETE when an appointment is confirmed

Uses BookingRegistry to hold slots when offered, preventing double-booking
across concurrent patients. Supports rescheduling of confirmed appointments.

Generates context-aware pre-appointment instructions based on the
patient's clinical data (medications, tests ordered, etc.)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import (
    BookingSection,
    PatientDiary,
    Phase,
    RiskLevel,
    SlotOption,
)
from medforce.gateway.events import EventEnvelope, EventType

logger = logging.getLogger("gateway.agents.booking")

# Risk → urgency window in days
URGENCY_WINDOWS: dict[str, int] = {
    RiskLevel.CRITICAL.value: 1,
    RiskLevel.HIGH.value: 2,
    RiskLevel.MEDIUM.value: 14,
    RiskLevel.LOW.value: 30,
    RiskLevel.NONE.value: 30,
}

# Keywords that indicate a reschedule request
RESCHEDULE_KEYWORDS = [
    "reschedule", "change my appointment", "different time",
    "different date", "can't make it", "cant make it",
    "change the date", "change the time", "move my appointment",
    "switch my appointment", "another time", "another date",
    "need to change", "want to change",
]

# Keywords that indicate the patient wants different slot options
SLOT_REJECTION_KEYWORDS = [
    "none of these", "none of those", "don't work", "dont work",
    "not available", "can't do any", "cant do any", "no good",
    "different options", "other times", "other options", "other dates",
    "not suitable", "won't work", "wont work", "doesn't work",
    "doesnt work", "none work", "can't make any", "cant make any",
    "busy on", "busy that", "not free", "unavailable",
]


class BookingAgent(BaseAgent):
    """
    Manages appointment booking based on clinical risk level.

    In test mode (no ScheduleCSVManager), generates mock slot options.
    In production, queries the real schedule manager.
    """

    agent_name = "booking"

    def __init__(
        self,
        schedule_manager=None,
        llm_client=None,
        booking_registry=None,
    ) -> None:
        self._schedule_manager = schedule_manager
        self._client = llm_client
        self._booking_registry = booking_registry
        self._model_name = os.getenv("BOOKING_MODEL", "gemini-2.0-flash")

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
        if event.event_type == EventType.CLINICAL_COMPLETE:
            return await self._handle_clinical_complete(event, diary)

        if event.event_type == EventType.RESCHEDULE_REQUEST:
            return await self._handle_reschedule(event, diary)

        if event.event_type == EventType.USER_MESSAGE:
            return await self._handle_user_message(event, diary)

        logger.warning(
            "Booking received unexpected event: %s", event.event_type.value
        )
        return AgentResult(updated_diary=diary)

    # ── Handlers ──

    async def _handle_clinical_complete(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Clinical assessment is done — present appointment slots."""
        channel = event.payload.get("channel", "websocket")
        diary.header.current_phase = Phase.BOOKING

        # Guard: don't re-book if patient already has a confirmed appointment
        if diary.booking.confirmed:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    f"Your appointment on {diary.booking.slot_selected.date} "
                    f"at {diary.booking.slot_selected.time} remains confirmed."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        risk_level = diary.header.risk_level.value
        window_days = URGENCY_WINDOWS.get(risk_level, 30)

        diary.booking.eligible_window = f"{window_days} days ({risk_level.upper()} risk)"

        # Get available slots
        slots = await self._get_available_slots(window_days)

        if not slots:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    f"Based on your {risk_level.upper()} risk assessment, we need to "
                    f"book you within {window_days} days. Unfortunately, no slots are "
                    f"currently available in that window. We'll follow up as soon as "
                    f"an opening becomes available."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # Hold slots in the registry to prevent double-booking
        # Pass all candidates — registry will skip held/booked slots and
        # stop after 3 successful holds
        if self._booking_registry:
            held = self._booking_registry.hold_slots(event.patient_id, slots)
            if not held:
                response = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=(
                        f"Based on your {risk_level.upper()} risk assessment, we need to "
                        f"book you within {window_days} days. Unfortunately, all available "
                        f"slots are currently being held by other patients. We'll follow up "
                        f"as soon as an opening becomes available."
                    ),
                    metadata={"patient_id": event.patient_id},
                )
                return AgentResult(updated_diary=diary, responses=[response])

            # Store only successfully held slots in diary
            diary.booking.slots_offered = [
                SlotOption(
                    date=h.date,
                    time=h.time,
                    provider=h.provider,
                    hold_id=h.hold_id,
                )
                for h in held
            ]
        else:
            # No registry — backward compatible: store all slots
            diary.booking.slots_offered = [
                SlotOption(date=s["date"], time=s["time"], provider=s.get("provider", ""))
                for s in slots[:3]
            ]

        # Build slot presentation
        slot_lines = []
        for i, slot in enumerate(diary.booking.slots_offered, 1):
            provider = f" with {slot.provider}" if slot.provider else ""
            slot_lines.append(f"  {i}. {slot.date} at {slot.time}{provider}")

        slots_text = "\n".join(slot_lines)

        message = (
            f"Based on your {risk_level.upper()} risk assessment, we need to book "
            f"your consultation within {window_days} days. Here are the available "
            f"appointments:\n\n{slots_text}\n\n"
            f"Please reply with the number of your preferred slot (1, 2, or 3)."
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    async def _handle_user_message(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Process slot selection from patient."""
        text = event.payload.get("text", "").strip()
        channel = event.payload.get("channel", "websocket")

        # If already booked, check for reschedule intent
        if diary.booking.confirmed:
            if self._is_reschedule_request(text):
                return await self._handle_reschedule(event, diary)

            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    f"Your appointment is already confirmed for "
                    f"{diary.booking.slot_selected.date} at "
                    f"{diary.booking.slot_selected.time}. "
                    f"If you'd like to reschedule, just say "
                    f"\"I'd like to change my appointment\"."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # If no slots offered yet, offer them
        if not diary.booking.slots_offered:
            return await self._handle_clinical_complete(event, diary)

        # Check if patient is rejecting all offered slots
        if self._is_slot_rejection(text):
            return await self._handle_slot_rejection(event, diary, channel)

        # Try to parse slot selection
        selected_slot = self._parse_slot_selection(text, diary.booking.slots_offered)

        if selected_slot is None:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "I didn't quite catch your selection. Please reply with "
                    "1, 2, or 3 to choose one of the offered appointment slots, "
                    "or let me know if none of these times work for you."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # Confirm the booking
        return await self._confirm_booking(event, diary, selected_slot, channel)

    async def _handle_reschedule(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Cancel existing booking and re-offer fresh slots."""
        channel = event.payload.get("channel", "websocket")

        # Save reschedule history
        if diary.booking.confirmed and diary.booking.slot_selected:
            diary.booking.rescheduled_from.append({
                "date": diary.booking.slot_selected.date,
                "time": diary.booking.slot_selected.time,
                "provider": diary.booking.slot_selected.provider,
                "appointment_id": diary.booking.appointment_id or "",
                "cancelled_at": datetime.now(timezone.utc).isoformat(),
            })

        # Cancel in registry
        if self._booking_registry:
            self._booking_registry.cancel_booking(event.patient_id)

        # Cancel in schedule manager
        if self._schedule_manager and diary.booking.slot_selected:
            try:
                self._schedule_manager.update_slot(
                    diary.booking.slot_selected.provider or "N0001",
                    diary.booking.slot_selected.date,
                    diary.booking.slot_selected.time,
                    {"patient": "", "status": "available"},
                )
            except Exception as exc:
                logger.warning("Schedule manager cancellation failed: %s", exc)

        # Reset booking fields
        diary.booking.slot_selected = None
        diary.booking.booked_by = None
        diary.booking.appointment_id = None
        diary.booking.confirmed = False
        diary.booking.slots_offered = []
        diary.booking.slots_rejected = []
        diary.booking.pre_appointment_instructions = []
        diary.header.current_phase = Phase.BOOKING

        # Deactivate monitoring
        diary.monitoring.monitoring_active = False

        # Re-offer fresh slots
        risk_level = diary.header.risk_level.value
        window_days = URGENCY_WINDOWS.get(risk_level, 30)
        diary.booking.eligible_window = f"{window_days} days ({risk_level.upper()} risk)"

        slots = await self._get_available_slots(window_days)

        if not slots:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "Your previous appointment has been cancelled. "
                    "Unfortunately, no new slots are currently available. "
                    "We'll follow up as soon as an opening becomes available."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # Hold new slots in registry
        if self._booking_registry:
            held = self._booking_registry.hold_slots(event.patient_id, slots)
            diary.booking.slots_offered = [
                SlotOption(
                    date=h.date,
                    time=h.time,
                    provider=h.provider,
                    hold_id=h.hold_id,
                )
                for h in held
            ]
        else:
            diary.booking.slots_offered = [
                SlotOption(date=s["date"], time=s["time"], provider=s.get("provider", ""))
                for s in slots[:3]
            ]

        # Build slot presentation
        slot_lines = []
        for i, slot in enumerate(diary.booking.slots_offered, 1):
            provider = f" with {slot.provider}" if slot.provider else ""
            slot_lines.append(f"  {i}. {slot.date} at {slot.time}{provider}")

        slots_text = "\n".join(slot_lines)

        message = (
            "No problem! Your previous appointment has been cancelled. "
            "Here are the available appointments:\n\n"
            f"{slots_text}\n\n"
            "Please reply with the number of your preferred slot (1, 2, or 3)."
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    # ── Booking Logic ──

    async def _confirm_booking(
        self,
        event: EventEnvelope,
        diary: PatientDiary,
        slot: SlotOption,
        channel: str,
    ) -> AgentResult:
        """Confirm appointment and generate pre-appointment instructions."""
        appointment_id = f"APT-{event.patient_id}-{slot.date}"

        # Confirm in registry if hold exists
        if self._booking_registry and slot.hold_id:
            confirmed = self._booking_registry.confirm_slot(
                event.patient_id, slot.hold_id, appointment_id
            )
            if confirmed is None:
                # Hold expired — re-offer fresh slots
                logger.info(
                    "Hold %s expired for patient %s — re-offering slots",
                    slot.hold_id, event.patient_id,
                )
                diary.booking.slots_offered = []
                return await self._handle_clinical_complete(event, diary)

        diary.booking.slot_selected = slot
        diary.booking.booked_by = event.sender_id
        diary.booking.confirmed = True
        diary.booking.appointment_id = appointment_id
        diary.header.current_phase = Phase.MONITORING

        # Generate pre-appointment instructions
        instructions = self._generate_instructions(diary)
        diary.booking.pre_appointment_instructions = instructions

        # Try to book in schedule manager
        if self._schedule_manager:
            try:
                self._schedule_manager.update_slot(
                    slot.provider or "N0001",
                    slot.date,
                    slot.time,
                    {"patient": event.patient_id, "status": "booked"},
                )
            except Exception as exc:
                logger.warning("Schedule manager booking failed: %s", exc)

        # Snapshot baseline for monitoring
        baseline = {}
        for doc in diary.clinical.documents:
            if doc.extracted_values:
                baseline.update(doc.extracted_values)
        diary.monitoring.baseline = baseline
        diary.monitoring.monitoring_active = True
        diary.monitoring.appointment_date = slot.date

        # Build confirmation message
        instructions_text = "\n".join(f"  - {inst}" for inst in instructions)
        provider_text = f" with {slot.provider}" if slot.provider else ""

        message = (
            f"Your appointment has been confirmed!\n\n"
            f"Date: {slot.date}\n"
            f"Time: {slot.time}\n"
            f"{f'Provider: {slot.provider}' + chr(10) if slot.provider else ''}"
            f"Appointment ID: {diary.booking.appointment_id}\n\n"
            f"Pre-appointment instructions:\n{instructions_text}\n\n"
            f"We'll keep in touch before your appointment to check on you."
        )

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=message,
            metadata={"patient_id": event.patient_id},
        )

        # Emit BOOKING_COMPLETE
        handoff = EventEnvelope.handoff(
            event_type=EventType.BOOKING_COMPLETE,
            patient_id=event.patient_id,
            source_agent="booking",
            payload={
                "appointment_date": slot.date,
                "appointment_time": slot.time,
                "appointment_id": diary.booking.appointment_id,
                "risk_level": diary.header.risk_level.value,
                "baseline": baseline,
                "channel": channel,
            },
            correlation_id=event.correlation_id,
        )

        return AgentResult(
            updated_diary=diary,
            emitted_events=[handoff],
            responses=[response],
        )

    # ── Helpers ──

    @staticmethod
    def _is_reschedule_request(text: str) -> bool:
        """Check if the text indicates a reschedule request."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in RESCHEDULE_KEYWORDS)

    @staticmethod
    def _is_slot_rejection(text: str) -> bool:
        """Check if the patient is rejecting all offered slots."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in SLOT_REJECTION_KEYWORDS)

    async def _handle_slot_rejection(
        self, event: EventEnvelope, diary: PatientDiary, channel: str
    ) -> AgentResult:
        """Patient doesn't want any of the offered slots — release holds and re-offer."""
        # Release existing holds
        if self._booking_registry:
            self._booking_registry.release_holds(event.patient_id)

        # Move currently offered slots to the rejected list
        diary.booking.slots_rejected.extend(diary.booking.slots_offered)
        diary.booking.slots_offered = []

        # Get fresh slots, excluding previously rejected ones
        risk_level = diary.header.risk_level.value
        window_days = URGENCY_WINDOWS.get(risk_level, 30)
        rejected_keys = {
            (s.date, s.time) for s in diary.booking.slots_rejected
        }
        all_slots = await self._get_available_slots(window_days)
        slots = [
            s for s in all_slots
            if (s["date"], s["time"]) not in rejected_keys
        ]

        if not slots:
            response = AgentResponse(
                recipient="patient",
                channel=channel,
                message=(
                    "I understand those times don't work for you. Unfortunately, "
                    "there are no other slots available within your booking window "
                    f"of {window_days} days. We'll follow up as soon as new "
                    "availability opens up."
                ),
                metadata={"patient_id": event.patient_id},
            )
            return AgentResult(updated_diary=diary, responses=[response])

        # Hold new slots
        if self._booking_registry:
            held = self._booking_registry.hold_slots(event.patient_id, slots)
            if not held:
                response = AgentResponse(
                    recipient="patient",
                    channel=channel,
                    message=(
                        "I understand those times don't work for you. Unfortunately, "
                        "all other available slots are currently being held. We'll "
                        "follow up as soon as an opening becomes available."
                    ),
                    metadata={"patient_id": event.patient_id},
                )
                return AgentResult(updated_diary=diary, responses=[response])

            diary.booking.slots_offered = [
                SlotOption(
                    date=h.date, time=h.time, provider=h.provider, hold_id=h.hold_id,
                )
                for h in held
            ]
        else:
            diary.booking.slots_offered = [
                SlotOption(date=s["date"], time=s["time"], provider=s.get("provider", ""))
                for s in slots[:3]
            ]

        slot_lines = []
        for i, slot in enumerate(diary.booking.slots_offered, 1):
            provider = f" with {slot.provider}" if slot.provider else ""
            slot_lines.append(f"  {i}. {slot.date} at {slot.time}{provider}")

        slots_text = "\n".join(slot_lines)

        response = AgentResponse(
            recipient="patient",
            channel=channel,
            message=(
                "No problem! Here are some alternative appointment times:\n\n"
                f"{slots_text}\n\n"
                "Please reply with the number of your preferred slot (1, 2, or 3), "
                "or let me know if these still don't work."
            ),
            metadata={"patient_id": event.patient_id},
        )

        return AgentResult(updated_diary=diary, responses=[response])

    async def _get_available_slots(self, window_days: int) -> list[dict]:
        """Get available appointment slots within the urgency window.

        Returns up to 12 candidate slots. The booking registry will filter
        these down to 3 un-held slots for the patient, so we need to
        over-fetch to account for slots held by other patients.
        """
        if self._schedule_manager:
            try:
                all_slots = self._schedule_manager.get_empty_schedule()
                # Filter by date window
                cutoff = datetime.now(timezone.utc) + timedelta(days=window_days)
                cutoff_str = cutoff.strftime("%Y-%m-%d")
                filtered = [
                    s for s in all_slots if s.get("date", "") <= cutoff_str
                ]
                return filtered[:12]
            except Exception as exc:
                logger.warning("Schedule manager query failed: %s", exc)

        # Fallback: generate mock slots — enough candidates so multiple
        # patients can each get 3 unique slots from the registry
        today = datetime.now(timezone.utc)
        times = ["09:00", "10:00", "11:00", "11:30", "14:00", "14:30", "15:00", "16:00"]
        slots = []
        for day_offset in range(1, max(window_days, 7) + 1):
            slot_date = today + timedelta(days=day_offset)
            date_str = slot_date.strftime("%Y-%m-%d")
            for t in times:
                slots.append({
                    "date": date_str,
                    "time": t,
                    "provider": "Dr. Available",
                })
        return slots

    def _parse_slot_selection(
        self, text: str, slots: list[SlotOption]
    ) -> SlotOption | None:
        """Parse patient's slot selection from their message."""
        text = text.strip().lower()

        # Try numeric selection: "1", "2", "3"
        for char in text:
            if char.isdigit():
                idx = int(char) - 1
                if 0 <= idx < len(slots):
                    return slots[idx]

        # Try date match
        for slot in slots:
            if slot.date in text or slot.time in text:
                return slot

        # Try ordinal: "first", "second", "third"
        ordinals = {"first": 0, "second": 1, "third": 2, "1st": 0, "2nd": 1, "3rd": 2}
        for word, idx in ordinals.items():
            if word in text and idx < len(slots):
                return slots[idx]

        return None

    def _generate_instructions(self, diary: PatientDiary) -> list[str]:
        """Generate condition-aware, personalised pre-appointment instructions."""
        instructions = [
            "Please bring a valid photo ID and your NHS card",
            "Arrive 15 minutes before your appointment time",
        ]

        # ── Medication-specific instructions ──
        meds_lower = [m.lower() for m in diary.clinical.current_medications]

        if any("metformin" in m for m in meds_lower):
            instructions.append("Continue taking Metformin as prescribed")

        if any("warfarin" in m for m in meds_lower):
            instructions.append(
                "Continue taking Warfarin — bring your latest INR results"
            )

        if any("insulin" in m for m in meds_lower):
            instructions.append(
                "Bring your blood glucose diary and insulin pen"
            )

        if any("statin" in m or "atorvastatin" in m or "simvastatin" in m for m in meds_lower):
            instructions.append("Continue taking your statin medication as prescribed")

        # ── Condition-specific dietary/lifestyle guidance ──
        condition = (diary.clinical.condition_context or "").lower()

        if any(kw in condition for kw in ["cirrhosis", "liver", "hepat"]):
            instructions.append(
                "Avoid alcohol completely for at least 48 hours before your appointment"
            )
            instructions.append(
                "If you are on a low-sodium diet, please continue following it"
            )

        if any(kw in condition for kw in ["mash", "nafld", "nash", "fatty"]):
            instructions.append(
                "Please note your current weight before your appointment — "
                "we may need to track changes over time"
            )
            instructions.append(
                "Avoid high-fat foods for 24 hours before your appointment"
            )

        # ── Risk-based instructions ──
        risk = diary.header.risk_level
        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            instructions.append(
                "Fasting is required for 8-12 hours before your appointment "
                "(water is fine)"
            )

        has_liver_tests = any(
            doc.type in ("lab_results", "blood_test")
            for doc in diary.clinical.documents
        )
        if has_liver_tests or risk == RiskLevel.HIGH:
            instructions.append(
                "A blood test may be required — please wear comfortable clothing "
                "with easy access to your arm"
            )

        # ── Allergy awareness ──
        if diary.clinical.allergies and diary.clinical.allergies != ["NKDA"]:
            instructions.append(
                "Please remind the clinician of your allergies: "
                + ", ".join(diary.clinical.allergies)
            )

        # ── Red flag warning ──
        if diary.clinical.red_flags:
            instructions.append(
                "If you experience any worsening symptoms before your appointment, "
                "please contact NHS 111 or attend A&E"
            )

        # ── Bring relevant documents ──
        instructions.append(
            "Bring any recent lab results, imaging reports, or medication lists "
            "that you haven't already shared with us"
        )

        return instructions

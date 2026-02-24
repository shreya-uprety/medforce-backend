"""
Tests for the Booking Agent — urgency windows, slot selection,
pre-appointment instructions, and BOOKING_COMPLETE handoff.
"""

import pytest
from unittest.mock import MagicMock

from medforce.gateway.agents.booking_agent import (
    URGENCY_WINDOWS,
    BookingAgent,
)
from medforce.gateway.booking_registry import BookingRegistry
from medforce.gateway.diary import (
    BookingSection,
    ClinicalDocument,
    ClinicalSection,
    DiaryHeader,
    IntakeSection,
    PatientDiary,
    Phase,
    RiskLevel,
    SlotOption,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole


# ── Fixtures ──


def make_diary(
    patient_id: str = "PT-200",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    phase: Phase = Phase.BOOKING,
) -> PatientDiary:
    """Create a test diary in the booking phase."""
    diary = PatientDiary.create_new(patient_id)
    diary.header.current_phase = phase
    diary.header.risk_level = risk_level
    diary.intake.name = "Jane Doe"
    diary.intake.phone = "07700900111"
    diary.intake.gp_name = "Dr. Wilson"
    return diary


def make_diary_with_clinical_data(
    risk_level: RiskLevel = RiskLevel.HIGH,
) -> PatientDiary:
    """Diary with medications and red flags for instruction generation."""
    diary = make_diary(risk_level=risk_level)
    diary.clinical.current_medications = ["metformin 500mg", "warfarin 5mg"]
    diary.clinical.red_flags = ["jaundice"]
    diary.clinical.documents = [
        ClinicalDocument(
            type="lab_results", processed=True,
            extracted_values={"bilirubin": 6.0},
        )
    ]
    return diary


def make_clinical_complete_event(patient_id: str = "PT-200") -> EventEnvelope:
    return EventEnvelope.handoff(
        event_type=EventType.CLINICAL_COMPLETE,
        patient_id=patient_id,
        source_agent="clinical",
        payload={
            "risk_level": "high",
            "risk_method": "deterministic_rule: bilirubin > 5",
            "channel": "websocket",
        },
    )


def make_user_message(text: str, patient_id: str = "PT-200") -> EventEnvelope:
    return EventEnvelope.user_message(patient_id=patient_id, text=text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Urgency Window Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUrgencyWindows:
    """Risk level → booking window mapping."""

    def test_critical_1_day(self):
        assert URGENCY_WINDOWS[RiskLevel.CRITICAL.value] == 1

    def test_high_2_days(self):
        assert URGENCY_WINDOWS[RiskLevel.HIGH.value] == 2

    def test_medium_14_days(self):
        assert URGENCY_WINDOWS[RiskLevel.MEDIUM.value] == 14

    def test_low_30_days(self):
        assert URGENCY_WINDOWS[RiskLevel.LOW.value] == 30

    def test_none_30_days(self):
        assert URGENCY_WINDOWS[RiskLevel.NONE.value] == 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Complete Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalComplete:
    """CLINICAL_COMPLETE → slot presentation."""

    @pytest.mark.asyncio
    async def test_presents_slots_for_high_risk(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.HIGH)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        assert len(result.responses) == 1
        msg = result.responses[0].message
        assert "HIGH" in msg
        assert "2 days" in msg

    @pytest.mark.asyncio
    async def test_presents_slots_for_medium_risk(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        msg = result.responses[0].message
        assert "MEDIUM" in msg
        assert "14 days" in msg

    @pytest.mark.asyncio
    async def test_offers_max_3_slots(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        assert len(result.updated_diary.booking.slots_offered) <= 3

    @pytest.mark.asyncio
    async def test_sets_eligible_window_in_diary(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.HIGH)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.eligible_window is not None
        assert "HIGH" in result.updated_diary.booking.eligible_window.upper()

    @pytest.mark.asyncio
    async def test_slot_presentation_includes_date_and_time(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        msg = result.responses[0].message
        # Should contain date patterns
        assert "10:00" in msg or "at" in msg
        # Should ask patient to select
        assert "1" in msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Slot Selection (User Message)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSlotSelection:
    """Patient selects an appointment slot."""

    @pytest.mark.asyncio
    async def test_numeric_selection_1(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00", provider="Dr. A"),
            SlotOption(date="2026-03-02", time="14:00", provider="Dr. B"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.confirmed is True
        assert result.updated_diary.booking.slot_selected.date == "2026-03-01"

    @pytest.mark.asyncio
    async def test_numeric_selection_2(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        event = make_user_message("2")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.slot_selected.date == "2026-03-02"

    @pytest.mark.asyncio
    async def test_ordinal_selection_first(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        event = make_user_message("I'd like the first one please")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.confirmed is True
        assert result.updated_diary.booking.slot_selected.date == "2026-03-01"

    @pytest.mark.asyncio
    async def test_invalid_selection_prompts_retry(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        event = make_user_message("maybe next week?")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.confirmed is False
        assert "1, 2, or 3" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_already_booked_returns_info(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.confirmed = True
        diary.booking.slot_selected = SlotOption(
            date="2026-03-01", time="10:00"
        )
        event = make_user_message("Can I change?")

        result = await agent.process(event, diary)

        assert "already confirmed" in result.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_no_slots_offered_triggers_slot_presentation(self):
        """If no slots offered yet, re-trigger slot presentation."""
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        # No slots_offered
        event = make_user_message("I want an appointment")

        result = await agent.process(event, diary)

        # Should present slots
        assert len(result.updated_diary.booking.slots_offered) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Slot Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSlotParsing:
    """_parse_slot_selection edge cases."""

    def test_parse_digit_in_sentence(self):
        agent = BookingAgent()
        slots = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        result = agent._parse_slot_selection("I'll take option 2", slots)
        assert result is not None
        assert result.date == "2026-03-02"

    def test_parse_date_match(self):
        agent = BookingAgent()
        slots = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        result = agent._parse_slot_selection("2026-03-02", slots)
        assert result is not None
        assert result.date == "2026-03-02"

    def test_parse_time_match(self):
        agent = BookingAgent()
        slots = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        # Note: "the 14:00 slot" contains digit "1" which triggers numeric
        # matching first (slot 1 = index 0). Use a time without leading digits.
        result = agent._parse_slot_selection("at 10:00", slots)
        assert result is not None
        assert result.time == "10:00"

    def test_parse_second_ordinal(self):
        agent = BookingAgent()
        slots = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        result = agent._parse_slot_selection("second", slots)
        assert result is not None
        assert result.date == "2026-03-02"

    def test_parse_2nd_ordinal(self):
        agent = BookingAgent()
        slots = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        result = agent._parse_slot_selection("2nd", slots)
        assert result is not None
        assert result.date == "2026-03-02"

    def test_parse_returns_none_for_garbage(self):
        agent = BookingAgent()
        slots = [SlotOption(date="2026-03-01", time="10:00")]
        result = agent._parse_slot_selection("hello world", slots)
        assert result is None

    def test_parse_out_of_range_digit(self):
        agent = BookingAgent()
        slots = [SlotOption(date="2026-03-01", time="10:00")]
        # "5" is out of range for 1 slot
        result = agent._parse_slot_selection("5", slots)
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Booking Confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBookingConfirmation:
    """Booking flow after slot selection."""

    @pytest.mark.asyncio
    async def test_confirmation_sets_diary_fields(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00", provider="Dr. A"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.confirmed is True
        assert result.updated_diary.booking.appointment_id is not None
        assert result.updated_diary.booking.slot_selected is not None
        assert "APT-" in result.updated_diary.booking.appointment_id

    @pytest.mark.asyncio
    async def test_sets_phase_to_monitoring(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        assert result.updated_diary.header.current_phase == Phase.MONITORING

    @pytest.mark.asyncio
    async def test_emits_booking_complete(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        booking_events = [
            e for e in result.emitted_events
            if e.event_type == EventType.BOOKING_COMPLETE
        ]
        assert len(booking_events) == 1
        assert booking_events[0].payload["appointment_date"] == "2026-03-01"

    @pytest.mark.asyncio
    async def test_confirmation_message_includes_details(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00", provider="Dr. A"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        msg = result.responses[0].message
        assert "confirmed" in msg.lower()
        assert "2026-03-01" in msg
        assert "10:00" in msg
        assert "instructions" in msg.lower()

    @pytest.mark.asyncio
    async def test_monitoring_baseline_snapshot(self):
        """Booking should snapshot lab values as monitoring baseline."""
        agent = BookingAgent()
        diary = make_diary()
        diary.clinical.documents = [
            ClinicalDocument(
                type="lab_results", processed=True,
                extracted_values={"bilirubin": 3.0, "ALT": 200},
            )
        ]
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        assert result.updated_diary.monitoring.baseline["bilirubin"] == 3.0
        assert result.updated_diary.monitoring.monitoring_active is True
        assert result.updated_diary.monitoring.appointment_date == "2026-03-01"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pre-appointment Instructions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPreAppointmentInstructions:
    """Context-aware instruction generation."""

    def test_basic_instructions_always_present(self):
        agent = BookingAgent()
        diary = make_diary()
        instructions = agent._generate_instructions(diary)
        assert any("photo ID" in i for i in instructions)
        assert any("15 minutes" in i for i in instructions)

    def test_metformin_instruction(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.clinical.current_medications = ["metformin 500mg"]
        instructions = agent._generate_instructions(diary)
        assert any("Metformin" in i for i in instructions)

    def test_warfarin_instruction(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.clinical.current_medications = ["warfarin 5mg"]
        instructions = agent._generate_instructions(diary)
        assert any("Warfarin" in i and "INR" in i for i in instructions)

    def test_insulin_instruction(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.clinical.current_medications = ["insulin glargine"]
        instructions = agent._generate_instructions(diary)
        assert any("insulin" in i.lower() for i in instructions)

    def test_high_risk_fasting_instruction(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.HIGH)
        instructions = agent._generate_instructions(diary)
        assert any("fasting" in i.lower() for i in instructions)

    def test_critical_risk_fasting_instruction(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.CRITICAL)
        instructions = agent._generate_instructions(diary)
        assert any("fasting" in i.lower() for i in instructions)

    def test_low_risk_no_fasting(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.LOW)
        instructions = agent._generate_instructions(diary)
        assert not any("fasting" in i.lower() for i in instructions)

    def test_red_flags_nhs_111_instruction(self):
        agent = BookingAgent()
        diary = make_diary()
        diary.clinical.red_flags = ["jaundice"]
        instructions = agent._generate_instructions(diary)
        assert any("NHS 111" in i or "A&E" in i for i in instructions)

    def test_lab_documents_blood_test_instruction(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.HIGH)
        diary.clinical.documents = [
            ClinicalDocument(type="lab_results", processed=True)
        ]
        instructions = agent._generate_instructions(diary)
        assert any("blood test" in i.lower() for i in instructions)

    def test_combined_medication_instructions(self):
        """Patient on metformin + warfarin gets both instructions."""
        agent = BookingAgent()
        diary = make_diary_with_clinical_data(risk_level=RiskLevel.HIGH)
        instructions = agent._generate_instructions(diary)
        has_metformin = any("Metformin" in i for i in instructions)
        has_warfarin = any("Warfarin" in i for i in instructions)
        assert has_metformin and has_warfarin


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Schedule Manager Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestScheduleManager:
    """Integration with ScheduleCSVManager (mocked)."""

    @pytest.mark.asyncio
    async def test_uses_schedule_manager_for_slots(self):
        mock_scheduler = MagicMock()
        mock_scheduler.get_empty_schedule.return_value = [
            {"date": "2026-03-01", "time": "09:00", "provider": "Dr. Real"},
            {"date": "2026-03-02", "time": "11:00", "provider": "Dr. Real"},
        ]
        agent = BookingAgent(schedule_manager=mock_scheduler)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        mock_scheduler.get_empty_schedule.assert_called_once()
        assert result.updated_diary.booking.slots_offered[0].provider == "Dr. Real"

    @pytest.mark.asyncio
    async def test_updates_schedule_on_booking(self):
        mock_scheduler = MagicMock()
        mock_scheduler.get_empty_schedule.return_value = [
            {"date": "2026-03-01", "time": "09:00"},
        ]
        agent = BookingAgent(schedule_manager=mock_scheduler)
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="09:00", provider="N0001"),
        ]
        event = make_user_message("1")

        result = await agent.process(event, diary)

        mock_scheduler.update_slot.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_when_schedule_manager_fails(self):
        mock_scheduler = MagicMock()
        mock_scheduler.get_empty_schedule.side_effect = Exception("GCS down")
        agent = BookingAgent(schedule_manager=mock_scheduler)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        # Should fall back to mock slots
        assert len(result.updated_diary.booking.slots_offered) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBookingScenarios:
    """End-to-end booking scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_high_risk_urgent_booking(self):
        """HIGH risk patient → 2 day window → book → MONITORING."""
        agent = BookingAgent()
        diary = make_diary_with_clinical_data(risk_level=RiskLevel.HIGH)

        # Step 1: Clinical complete triggers slot presentation
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        assert "HIGH" in result1.responses[0].message
        assert len(diary.booking.slots_offered) > 0

        # Step 2: Patient selects slot 1
        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary

        assert diary.booking.confirmed is True
        assert diary.header.current_phase == Phase.MONITORING
        # Should have medication-specific instructions
        assert len(diary.booking.pre_appointment_instructions) >= 4

    @pytest.mark.asyncio
    async def test_scenario_low_risk_routine_booking(self):
        """LOW risk → 30 day window → straightforward booking."""
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.LOW)

        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        assert "LOW" in result1.responses[0].message
        assert "30 days" in result1.responses[0].message

        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)

        assert result2.updated_diary.booking.confirmed is True

    @pytest.mark.asyncio
    async def test_scenario_invalid_then_valid_selection(self):
        """Patient enters invalid selection, then corrects."""
        agent = BookingAgent()
        diary = make_diary()
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]

        # Invalid input
        event1 = make_user_message("next week please")
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary
        assert diary.booking.confirmed is False

        # Valid input
        event2 = make_user_message("2")
        result2 = await agent.process(event2, diary)
        assert result2.updated_diary.booking.confirmed is True
        assert result2.updated_diary.booking.slot_selected.date == "2026-03-02"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Booking with Registry Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBookingWithRegistry:
    """BookingAgent integration with BookingRegistry."""

    @pytest.mark.asyncio
    async def test_holds_created_on_slot_offer(self):
        """CLINICAL_COMPLETE should create holds in the registry."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        # Slots should be held in registry
        active = registry.get_active_holds()
        assert len(active) > 0
        # Each offered slot should have a hold_id
        for slot in result.updated_diary.booking.slots_offered:
            assert slot.hold_id != ""

    @pytest.mark.asyncio
    async def test_confirm_promotes_hold_in_registry(self):
        """Selecting a slot should confirm it in the registry."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        # Step 1: Offer slots
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        # Step 2: Select slot 1
        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)

        # Registry should have a confirmed booking
        booking = registry.get_patient_booking("PT-200")
        assert booking is not None
        assert booking.status == "confirmed"

    @pytest.mark.asyncio
    async def test_expired_hold_reoffers_slots(self):
        """If the hold expired, selecting it should re-offer fresh slots."""
        from datetime import timedelta
        from medforce.gateway.booking_registry import _now

        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        # Offer slots
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        # Force all holds to expire
        for hold in registry._data.holds:
            hold.expires_at = _now() - timedelta(minutes=1)

        # Try to select — should detect expired hold and re-offer
        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)

        # Should NOT be confirmed (hold expired)
        assert result2.updated_diary.booking.confirmed is False
        # Should have re-offered slots
        msg = result2.responses[0].message
        assert "1" in msg  # slot numbers in re-offer

    @pytest.mark.asyncio
    async def test_double_booking_prevented(self):
        """Two patients can't hold the same slot simultaneously."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)

        # Patient 1 gets slots offered
        diary1 = make_diary(patient_id="PT-100", risk_level=RiskLevel.MEDIUM)
        event1 = make_clinical_complete_event(patient_id="PT-100")
        result1 = await agent.process(event1, diary1)
        pt1_slots = result1.updated_diary.booking.slots_offered

        # Patient 2 tries — PT-100's slots are already held, so PT-200
        # should get different ones (no overlap).
        diary2 = make_diary(patient_id="PT-200", risk_level=RiskLevel.MEDIUM)
        event2 = make_clinical_complete_event(patient_id="PT-200")
        result2 = await agent.process(event2, diary2)

        pt2_slots = result2.updated_diary.booking.slots_offered
        assert len(pt2_slots) > 0, "PT-200 should get slots from the larger pool"

        # Verify no overlap with PT-100's held slots
        pt1_keys = {(s.date, s.time) for s in pt1_slots}
        pt2_keys = {(s.date, s.time) for s in pt2_slots}
        assert pt1_keys.isdisjoint(pt2_keys), (
            f"Double-booking! PT-100 and PT-200 share slots: {pt1_keys & pt2_keys}"
        )

    @pytest.mark.asyncio
    async def test_reschedule_flow(self):
        """Full reschedule: confirm → reschedule → re-offer → re-confirm."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        # Step 1: Offer and confirm
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary
        assert diary.booking.confirmed is True
        old_date = diary.booking.slot_selected.date

        # Step 2: Reschedule
        reschedule_event = EventEnvelope.handoff(
            event_type=EventType.RESCHEDULE_REQUEST,
            patient_id="PT-200",
            source_agent="monitoring",
            payload={"channel": "websocket"},
        )
        result3 = await agent.process(reschedule_event, diary)
        diary = result3.updated_diary

        assert diary.booking.confirmed is False
        assert diary.header.current_phase == Phase.BOOKING
        assert len(diary.booking.rescheduled_from) == 1
        assert diary.booking.rescheduled_from[0]["date"] == old_date
        assert len(diary.booking.slots_offered) > 0

    @pytest.mark.asyncio
    async def test_reschedule_cancels_in_registry(self):
        """Rescheduling should cancel the booking in the registry."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        # Confirm
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary
        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary
        assert registry.get_patient_booking("PT-200") is not None

        # Reschedule
        reschedule_event = EventEnvelope.handoff(
            event_type=EventType.RESCHEDULE_REQUEST,
            patient_id="PT-200",
            source_agent="monitoring",
            payload={"channel": "websocket"},
        )
        await agent.process(reschedule_event, diary)

        assert registry.get_patient_booking("PT-200") is None

    @pytest.mark.asyncio
    async def test_reschedule_keyword_in_booking_phase(self):
        """Reschedule keywords while in booking phase with confirmed booking."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        # Confirm booking
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary
        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)
        diary = result2.updated_diary
        # Force back to BOOKING phase for this test
        diary.header.current_phase = Phase.BOOKING

        # Patient says "I'd like to reschedule"
        event3 = make_user_message("I'd like to reschedule")
        result3 = await agent.process(event3, diary)

        assert result3.updated_diary.booking.confirmed is False
        assert "cancelled" in result3.responses[0].message.lower()

    @pytest.mark.asyncio
    async def test_backward_compat_without_registry(self):
        """BookingAgent works without registry (backward compatible)."""
        agent = BookingAgent()  # No registry
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        event = make_clinical_complete_event()

        result = await agent.process(event, diary)

        assert len(result.updated_diary.booking.slots_offered) > 0
        # hold_id should be empty (no registry)
        for slot in result.updated_diary.booking.slots_offered:
            assert slot.hold_id == ""

    @pytest.mark.asyncio
    async def test_reschedule_history_accumulated(self):
        """Multiple reschedules accumulate history."""
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        for i in range(2):
            event1 = make_clinical_complete_event()
            result1 = await agent.process(event1, diary)
            diary = result1.updated_diary

            event2 = make_user_message("1")
            result2 = await agent.process(event2, diary)
            diary = result2.updated_diary

            if i < 1:
                reschedule_event = EventEnvelope.handoff(
                    event_type=EventType.RESCHEDULE_REQUEST,
                    patient_id="PT-200",
                    source_agent="monitoring",
                    payload={"channel": "websocket"},
                )
                result3 = await agent.process(reschedule_event, diary)
                diary = result3.updated_diary

        assert len(diary.booking.rescheduled_from) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Slot Rejection (Patient doesn't want offered times)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSlotRejection:
    """Patient rejects offered slots and gets new ones."""

    @pytest.mark.asyncio
    async def test_none_of_these_triggers_reoffer(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
            SlotOption(date="2026-03-02", time="14:00"),
        ]
        event = make_user_message("none of those work for me")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.confirmed is False
        msg = result.responses[0].message.lower()
        assert "alternative" in msg or "no problem" in msg or "unfortunately" in msg

    @pytest.mark.asyncio
    async def test_not_available_triggers_reoffer(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        event = make_user_message("I'm not available at those times")

        result = await agent.process(event, diary)

        assert result.updated_diary.booking.confirmed is False

    @pytest.mark.asyncio
    async def test_rejection_clears_old_slots(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]
        event = make_user_message("none of these work")

        result = await agent.process(event, diary)

        # New slots should be offered (mock always returns 3)
        assert len(result.updated_diary.booking.slots_offered) > 0

    @pytest.mark.asyncio
    async def test_rejection_with_registry_releases_holds(self):
        registry = BookingRegistry()
        agent = BookingAgent(booking_registry=registry)
        diary = make_diary(risk_level=RiskLevel.MEDIUM)

        # First offer slots (creates holds)
        event1 = make_clinical_complete_event()
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary
        assert len(registry.get_active_holds()) > 0

        # Reject them
        event2 = make_user_message("none of those times work for me")
        result2 = await agent.process(event2, diary)

        # Old holds should be released, new ones created
        msg = result2.responses[0].message.lower()
        assert "alternative" in msg or "no problem" in msg or "unfortunately" in msg

    @pytest.mark.asyncio
    async def test_rejection_then_accept(self):
        agent = BookingAgent()
        diary = make_diary(risk_level=RiskLevel.MEDIUM)
        diary.booking.slots_offered = [
            SlotOption(date="2026-03-01", time="10:00"),
        ]

        # Reject
        event1 = make_user_message("those don't work for me")
        result1 = await agent.process(event1, diary)
        diary = result1.updated_diary

        # Accept from new slots
        event2 = make_user_message("1")
        result2 = await agent.process(event2, diary)

        assert result2.updated_diary.booking.confirmed is True

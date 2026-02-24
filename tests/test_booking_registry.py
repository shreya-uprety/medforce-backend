"""
Tests for the Booking Registry — slot holds, double-booking prevention,
confirmation, cancellation, expiry, and rescheduling flows.
"""

from datetime import datetime, timedelta, timezone

import pytest

from medforce.gateway.booking_registry import (
    BookingRegistry,
    BookingRegistryData,
    SlotHold,
    _now,
)


# ── Helpers ──


def make_slots(*specs: tuple[str, str, str]) -> list[dict[str, str]]:
    """Create slot dicts from (date, time, provider) tuples."""
    return [
        {"date": s[0], "time": s[1], "provider": s[2]} for s in specs
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Basic Hold Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSlotHolds:
    """Creating and managing slot holds."""

    def test_hold_single_slot(self):
        registry = BookingRegistry()
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))
        held = registry.hold_slots("PT-100", slots)

        assert len(held) == 1
        assert held[0].patient_id == "PT-100"
        assert held[0].date == "2026-03-01"
        assert held[0].time == "09:00"
        assert held[0].status == "held"

    def test_hold_multiple_slots(self):
        registry = BookingRegistry()
        slots = make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. B"),
            ("2026-03-02", "14:00", "Dr. A"),
        )
        held = registry.hold_slots("PT-100", slots)
        assert len(held) == 3

    def test_hold_generates_unique_ids(self):
        registry = BookingRegistry()
        slots = make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. B"),
        )
        held = registry.hold_slots("PT-100", slots)
        assert held[0].hold_id != held[1].hold_id

    def test_hold_sets_expiry(self):
        registry = BookingRegistry(hold_ttl_minutes=10)
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))
        held = registry.hold_slots("PT-100", slots)

        # Expiry should be ~10 minutes from now
        delta = held[0].expires_at - held[0].held_at
        assert 9 <= delta.total_seconds() / 60 <= 11


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Double-Booking Prevention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDoubleBookingPrevention:
    """Preventing two patients from holding the same slot."""

    def test_second_patient_cannot_hold_same_slot(self):
        registry = BookingRegistry()
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))

        held1 = registry.hold_slots("PT-100", slots)
        held2 = registry.hold_slots("PT-200", slots)

        assert len(held1) == 1
        assert len(held2) == 0

    def test_same_patient_can_reoffer_same_slot(self):
        registry = BookingRegistry()
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))

        held1 = registry.hold_slots("PT-100", slots)
        held2 = registry.hold_slots("PT-100", slots)

        # Same patient can get the same slot again
        assert len(held1) == 1
        assert len(held2) == 1

    def test_different_providers_same_time_allowed(self):
        """Different providers at same date/time are different slots."""
        registry = BookingRegistry()

        held1 = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))
        held2 = registry.hold_slots("PT-200", make_slots(("2026-03-01", "09:00", "Dr. B")))

        assert len(held1) == 1
        assert len(held2) == 1

    def test_confirmed_slot_blocks_others(self):
        registry = BookingRegistry()
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))

        held = registry.hold_slots("PT-100", slots)
        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")

        # Another patient tries to hold the same slot
        held2 = registry.hold_slots("PT-200", slots)
        assert len(held2) == 0

    def test_partial_filtering(self):
        """If 2 of 3 slots are taken, only the free one is returned."""
        registry = BookingRegistry()

        # PT-100 holds slot 1 and 2
        registry.hold_slots("PT-100", make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. A"),
        ))

        # PT-200 tries to hold all three, only gets slot 3
        held2 = registry.hold_slots("PT-200", make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. A"),
            ("2026-03-01", "14:00", "Dr. A"),
        ))
        assert len(held2) == 1
        assert held2[0].time == "14:00"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfirmation:
    """Promoting holds to confirmed bookings."""

    def test_confirm_slot(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))

        result = registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")

        assert result is not None
        assert result.status == "confirmed"
        assert result.appointment_id == "APT-100"
        assert result.confirmed_at is not None

    def test_confirm_releases_other_holds(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. B"),
        ))

        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")

        # The other hold should be cancelled
        active = registry.get_active_holds()
        assert len(active) == 1
        assert active[0].hold_id == held[0].hold_id

    def test_confirm_nonexistent_hold_returns_none(self):
        registry = BookingRegistry()
        result = registry.confirm_slot("PT-100", "fake-id", "APT-100")
        assert result is None

    def test_confirm_wrong_patient_returns_none(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))

        result = registry.confirm_slot("PT-200", held[0].hold_id, "APT-200")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Expiry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExpiry:
    """Hold expiration and cleanup."""

    def test_expired_hold_cannot_be_confirmed(self):
        registry = BookingRegistry(hold_ttl_minutes=0)  # immediate expiry
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))

        # Force expiry by setting expires_at in the past
        for h in registry._data.holds:
            h.expires_at = _now() - timedelta(minutes=1)

        result = registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")
        assert result is None

    def test_expired_holds_freed_for_other_patients(self):
        registry = BookingRegistry()
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))

        held = registry.hold_slots("PT-100", slots)

        # Force expiry
        for h in registry._data.holds:
            h.expires_at = _now() - timedelta(minutes=1)

        # PT-200 should now be able to hold it
        held2 = registry.hold_slots("PT-200", slots)
        assert len(held2) == 1
        assert held2[0].patient_id == "PT-200"

    def test_cleanup_on_load(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))

        # Force expiry
        for h in registry._data.holds:
            h.expires_at = _now() - timedelta(minutes=1)

        # Trigger cleanup via _load
        registry._load()

        # Expired holds should be cancelled
        active = registry.get_active_holds()
        assert len(active) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cancellation & Rescheduling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCancellation:
    """Cancelling bookings and rescheduling."""

    def test_cancel_confirmed_booking(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))
        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")

        cancelled = registry.cancel_booking("PT-100")

        assert cancelled is not None
        assert cancelled.status == "cancelled"
        assert cancelled.cancelled_at is not None

    def test_cancel_when_no_booking_returns_none(self):
        registry = BookingRegistry()
        result = registry.cancel_booking("PT-100")
        assert result is None

    def test_reschedule_frees_slot(self):
        """After cancelling, the slot should be available for others."""
        registry = BookingRegistry()
        slots = make_slots(("2026-03-01", "09:00", "Dr. A"))

        held = registry.hold_slots("PT-100", slots)
        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")
        registry.cancel_booking("PT-100")

        # PT-200 should now be able to hold it
        held2 = registry.hold_slots("PT-200", slots)
        assert len(held2) == 1

    def test_release_holds(self):
        registry = BookingRegistry()
        registry.hold_slots("PT-100", make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. B"),
        ))

        count = registry.release_holds("PT-100")

        assert count == 2
        assert len(registry.get_active_holds()) == 0

    def test_release_holds_keeps_confirmed(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(
            ("2026-03-01", "09:00", "Dr. A"),
            ("2026-03-01", "11:00", "Dr. B"),
        ))

        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")
        count = registry.release_holds("PT-100")

        # Only the non-confirmed hold should be released
        # (confirm_slot already cancels other holds, so 0 here)
        active = registry.get_active_holds()
        assert len(active) == 1
        assert active[0].status == "confirmed"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Lookup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientLookup:
    """Looking up patient bookings."""

    def test_get_patient_booking(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))
        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")

        booking = registry.get_patient_booking("PT-100")

        assert booking is not None
        assert booking.appointment_id == "APT-100"
        assert booking.status == "confirmed"

    def test_get_patient_booking_none_when_not_confirmed(self):
        registry = BookingRegistry()
        registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))

        booking = registry.get_patient_booking("PT-100")
        assert booking is None

    def test_get_patient_booking_none_after_cancel(self):
        registry = BookingRegistry()
        held = registry.hold_slots("PT-100", make_slots(("2026-03-01", "09:00", "Dr. A")))
        registry.confirm_slot("PT-100", held[0].hold_id, "APT-100")
        registry.cancel_booking("PT-100")

        booking = registry.get_patient_booking("PT-100")
        assert booking is None

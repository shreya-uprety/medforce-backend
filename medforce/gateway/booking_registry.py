"""
Booking Registry — GCS-backed persistent slot-hold registry.

Prevents double-booking by holding slots when offered to a patient.
Uses GCS generation-based optimistic locking (same pattern as DiaryStore).

Storage path: gs://{bucket}/booking_registry/registry.json
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("gateway.booking_registry")

# Default hold TTL in minutes
DEFAULT_HOLD_TTL_MINUTES = 15


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


class SlotHold(BaseModel):
    """A single slot hold or confirmed booking in the registry."""

    hold_id: str = Field(default_factory=_new_id)
    patient_id: str
    date: str
    time: str
    provider: str = ""
    status: str = "held"  # held / confirmed / cancelled
    held_at: datetime = Field(default_factory=_now)
    expires_at: datetime = Field(default_factory=lambda: _now() + timedelta(minutes=DEFAULT_HOLD_TTL_MINUTES))
    confirmed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    appointment_id: Optional[str] = None


class BookingRegistryData(BaseModel):
    """Serialisable registry state."""

    holds: list[SlotHold] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=_now)


class BookingRegistry:
    """
    Persistent booking registry with GCS-backed JSON storage.

    When ``gcs_bucket_manager`` is None, operates in-memory (test mode).
    """

    REGISTRY_PATH = "booking_registry/registry.json"

    def __init__(
        self,
        gcs_bucket_manager=None,
        hold_ttl_minutes: int = DEFAULT_HOLD_TTL_MINUTES,
    ) -> None:
        self._gcs = gcs_bucket_manager
        self._hold_ttl = timedelta(minutes=hold_ttl_minutes)
        # In-memory state (always kept in sync with GCS)
        self._data: BookingRegistryData = BookingRegistryData()
        self._generation: int | None = None

    # ── Public API ──

    def hold_slots(
        self, patient_id: str, slots: list[dict[str, str]],
        max_holds: int = 3,
    ) -> list[SlotHold]:
        """
        Create holds for the given slots. Filters out slots that are
        already held or booked by another patient. Stops after
        ``max_holds`` successful holds to avoid blocking slots
        unnecessarily.

        Returns the list of successfully created SlotHold objects.
        """
        self._load()

        held: list[SlotHold] = []
        for slot in slots:
            if len(held) >= max_holds:
                break

            date = slot.get("date", "")
            time = slot.get("time", "")
            provider = slot.get("provider", "")

            if self._is_slot_taken(date, time, provider, exclude_patient=patient_id):
                logger.info(
                    "Slot %s %s already taken — skipping for patient %s",
                    date, time, patient_id,
                )
                continue

            hold = SlotHold(
                patient_id=patient_id,
                date=date,
                time=time,
                provider=provider,
                expires_at=_now() + self._hold_ttl,
            )
            self._data.holds.append(hold)
            held.append(hold)

        if held:
            self._save()

        return held

    def confirm_slot(
        self, patient_id: str, hold_id: str, appointment_id: str
    ) -> SlotHold | None:
        """
        Promote a held slot to confirmed. Returns the confirmed SlotHold,
        or None if the hold expired or doesn't exist.
        """
        self._load()

        for hold in self._data.holds:
            if (
                hold.hold_id == hold_id
                and hold.patient_id == patient_id
                and hold.status == "held"
            ):
                # Check expiry
                if hold.expires_at < _now():
                    hold.status = "cancelled"
                    hold.cancelled_at = _now()
                    self._save()
                    logger.info(
                        "Hold %s expired for patient %s", hold_id, patient_id
                    )
                    return None

                hold.status = "confirmed"
                hold.confirmed_at = _now()
                hold.appointment_id = appointment_id

                # Release other holds for this patient
                for other in self._data.holds:
                    if (
                        other.patient_id == patient_id
                        and other.hold_id != hold_id
                        and other.status == "held"
                    ):
                        other.status = "cancelled"
                        other.cancelled_at = _now()

                self._save()
                logger.info(
                    "Confirmed hold %s for patient %s (apt: %s)",
                    hold_id, patient_id, appointment_id,
                )
                return hold

        logger.warning("Hold %s not found for patient %s", hold_id, patient_id)
        return None

    def cancel_booking(self, patient_id: str) -> SlotHold | None:
        """
        Cancel a confirmed booking for rescheduling. Returns the
        cancelled SlotHold, or None if no confirmed booking exists.
        """
        self._load()

        for hold in self._data.holds:
            if hold.patient_id == patient_id and hold.status == "confirmed":
                hold.status = "cancelled"
                hold.cancelled_at = _now()
                self._save()
                logger.info(
                    "Cancelled booking for patient %s (hold %s)",
                    patient_id, hold.hold_id,
                )
                return hold

        logger.info("No confirmed booking to cancel for patient %s", patient_id)
        return None

    def get_patient_booking(self, patient_id: str) -> SlotHold | None:
        """Look up the current confirmed booking for a patient."""
        self._load()
        for hold in self._data.holds:
            if hold.patient_id == patient_id and hold.status == "confirmed":
                return hold
        return None

    def release_holds(self, patient_id: str) -> int:
        """Release all un-confirmed holds for a patient. Returns count released."""
        self._load()
        count = 0
        for hold in self._data.holds:
            if hold.patient_id == patient_id and hold.status == "held":
                hold.status = "cancelled"
                hold.cancelled_at = _now()
                count += 1
        if count:
            self._save()
            logger.info("Released %d holds for patient %s", count, patient_id)
        return count

    def get_active_holds(self) -> list[SlotHold]:
        """Return all currently active (held or confirmed) slot holds."""
        self._load()
        return [
            h for h in self._data.holds
            if h.status in ("held", "confirmed")
        ]

    # ── Internal persistence ──

    def _load(self) -> None:
        """Load registry from GCS (or use in-memory state)."""
        if self._gcs is None:
            # In-memory mode — just clean up expired holds
            self._cleanup_expired()
            return

        try:
            self._gcs._ensure_initialized()
            blob = self._gcs.bucket.blob(self.REGISTRY_PATH)
            if not blob.exists():
                self._data = BookingRegistryData()
                self._generation = None
                return

            content = blob.download_as_text()
            self._generation = blob.generation or 0
            raw = json.loads(content)
            self._data = BookingRegistryData.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed to load booking registry: %s", exc)
            if not self._data.holds:
                self._data = BookingRegistryData()

        self._cleanup_expired()

    def _save(self) -> None:
        """Save registry to GCS (or just update in-memory state)."""
        self._data.last_updated = _now()

        if self._gcs is None:
            return

        try:
            self._gcs._ensure_initialized()
            blob = self._gcs.bucket.blob(self.REGISTRY_PATH)
            content = self._data.model_dump_json(indent=2)

            if self._generation is not None:
                blob.upload_from_string(
                    content,
                    content_type="application/json",
                    if_generation_match=self._generation,
                )
            else:
                blob.upload_from_string(
                    content,
                    content_type="application/json",
                )

            blob.reload()
            self._generation = blob.generation or 0
        except Exception as exc:
            if "conditionNotMet" in str(exc) or "Precondition" in str(exc):
                logger.warning(
                    "Booking registry concurrency conflict — reloading and retrying"
                )
                self._load()
            else:
                logger.error("Failed to save booking registry: %s", exc)

    def _cleanup_expired(self) -> None:
        """Cancel holds that have expired their TTL."""
        now = _now()
        for hold in self._data.holds:
            if hold.status == "held" and hold.expires_at < now:
                hold.status = "cancelled"
                hold.cancelled_at = now

    def _is_slot_taken(
        self,
        date: str,
        time: str,
        provider: str,
        exclude_patient: str = "",
    ) -> bool:
        """Check if a slot is already held or confirmed by another patient."""
        for hold in self._data.holds:
            if hold.status not in ("held", "confirmed"):
                continue
            if hold.patient_id == exclude_patient:
                continue
            if hold.date == date and hold.time == time:
                # If provider is specified, match on provider too
                if provider and hold.provider and hold.provider != provider:
                    continue
                return True
        return False

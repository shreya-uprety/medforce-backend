"""
Identity Resolution — maps phone/email to (patient_id, role, permissions).

Resolution order:
  1. Patient registry (phone, email, NHS number)
  2. Helper registry (across all patient diaries)
  3. GP registry (email across all patient diaries)
  4. Unknown → reject

Maintains an in-memory contact index rebuilt from GCS on startup and
updated incrementally when helpers are added/removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("gateway.identity")


@dataclass
class IdentityRecord:
    """Result of identity resolution."""
    patient_id: str
    sender_id: str          # "PATIENT", "HELPER-001", "GP-DrPatel"
    sender_role: str        # "patient", "helper", "gp"
    name: str = ""
    relationship: str = ""  # spouse, child, friend (helpers only)
    permissions: list[str] = field(default_factory=list)
    channel: str = ""


@dataclass
class AmbiguousIdentity:
    """Returned when one contact maps to multiple patients."""
    records: list[IdentityRecord]


class IdentityResolver:
    """
    Resolves sender contact info to patient identity + permissions.

    The contact index is a reverse-lookup dict:
      contact_string → list[IdentityRecord]

    A list is used because one person (e.g. a parent) might be a helper
    for multiple patients.
    """

    def __init__(self) -> None:
        self._index: dict[str, list[IdentityRecord]] = {}

    # ── Public API ──

    def resolve(self, contact: str) -> IdentityRecord | AmbiguousIdentity | None:
        """
        Resolve a contact string (phone, email) to an identity.

        Returns:
          - IdentityRecord if exactly one match
          - AmbiguousIdentity if multiple matches (helper for multiple patients)
          - None if no match (unknown sender)
        """
        contact_key = self._normalise(contact)
        records = self._index.get(contact_key)

        if not records:
            return None
        if len(records) == 1:
            return records[0]
        return AmbiguousIdentity(records=list(records))

    def resolve_for_patient(
        self, contact: str, patient_id: str
    ) -> IdentityRecord | None:
        """Resolve contact in the context of a specific patient."""
        contact_key = self._normalise(contact)
        records = self._index.get(contact_key, [])
        for r in records:
            if r.patient_id == patient_id:
                return r
        return None

    # ── Index Management ──

    def rebuild_from_diaries(self, diaries: dict) -> int:
        """
        Rebuild the full contact index from all patient diaries.

        Args:
            diaries: dict of {patient_id: PatientDiary}

        Returns: number of contacts indexed
        """
        self._index.clear()
        count = 0

        for pid, diary in diaries.items():
            # Index the patient themselves
            for contact_field in ["phone", "email", "nhs_number"]:
                value = getattr(diary.intake, contact_field, None)
                if value:
                    self._add_to_index(
                        self._normalise(value),
                        IdentityRecord(
                            patient_id=pid,
                            sender_id="PATIENT",
                            sender_role="patient",
                            name=diary.intake.name or "",
                            permissions=["full_access"],
                            channel=diary.intake.contact_preference,
                        ),
                    )
                    count += 1

            # Index helpers
            for helper in diary.helper_registry.helpers:
                if helper.contact:
                    self._add_to_index(
                        self._normalise(helper.contact),
                        IdentityRecord(
                            patient_id=pid,
                            sender_id=helper.id,
                            sender_role="helper",
                            name=helper.name,
                            relationship=helper.relationship,
                            permissions=list(helper.permissions),
                            channel=helper.channel,
                        ),
                    )
                    count += 1

            # Index GP
            if diary.gp_channel.gp_email:
                self._add_to_index(
                    self._normalise(diary.gp_channel.gp_email),
                    IdentityRecord(
                        patient_id=pid,
                        sender_id=f"GP-{diary.gp_channel.gp_name or 'unknown'}",
                        sender_role="gp",
                        name=diary.gp_channel.gp_name or "",
                        permissions=["view_status", "upload_documents", "respond_to_queries"],
                        channel="email",
                    ),
                )
                count += 1

        logger.info("Identity index rebuilt: %d contacts across %d patients",
                     count, len(diaries))
        return count

    def update_for_patient(self, patient_id: str, diary) -> None:
        """Incrementally update the index for a single patient's diary."""
        # Remove all existing entries for this patient
        self._remove_patient(patient_id)

        # Re-add from current diary
        for contact_field in ["phone", "email", "nhs_number"]:
            value = getattr(diary.intake, contact_field, None)
            if value:
                self._add_to_index(
                    self._normalise(value),
                    IdentityRecord(
                        patient_id=patient_id,
                        sender_id="PATIENT",
                        sender_role="patient",
                        name=diary.intake.name or "",
                        permissions=["full_access"],
                        channel=diary.intake.contact_preference,
                    ),
                )

        for helper in diary.helper_registry.helpers:
            if helper.contact:
                self._add_to_index(
                    self._normalise(helper.contact),
                    IdentityRecord(
                        patient_id=patient_id,
                        sender_id=helper.id,
                        sender_role="helper",
                        name=helper.name,
                        relationship=helper.relationship,
                        permissions=list(helper.permissions),
                        channel=helper.channel,
                    ),
                )

        if diary.gp_channel.gp_email:
            self._add_to_index(
                self._normalise(diary.gp_channel.gp_email),
                IdentityRecord(
                    patient_id=patient_id,
                    sender_id=f"GP-{diary.gp_channel.gp_name or 'unknown'}",
                    sender_role="gp",
                    name=diary.gp_channel.gp_name or "",
                    permissions=["view_status", "upload_documents", "respond_to_queries"],
                    channel="email",
                ),
            )

    @property
    def index_size(self) -> int:
        """Total number of contact entries in the index."""
        return sum(len(v) for v in self._index.values())

    @property
    def unique_contacts(self) -> int:
        """Number of unique contact strings in the index."""
        return len(self._index)

    # ── Internal ──

    def _normalise(self, contact: str) -> str:
        """Normalise a contact string for consistent lookup."""
        contact = contact.strip().lower()
        # Strip spaces and dashes from phone numbers
        if contact.startswith("+") or contact[0:1].isdigit():
            contact = contact.replace(" ", "").replace("-", "")
            # Normalise UK mobile: 07xxx → +447xxx
            if contact.startswith("0") and len(contact) == 11:
                contact = "+44" + contact[1:]
        return contact

    def _add_to_index(self, key: str, record: IdentityRecord) -> None:
        if key not in self._index:
            self._index[key] = []
        # Avoid duplicates for same patient+role
        for existing in self._index[key]:
            if (existing.patient_id == record.patient_id
                    and existing.sender_id == record.sender_id):
                return
        self._index[key].append(record)

    def _remove_patient(self, patient_id: str) -> None:
        """Remove all index entries for a given patient."""
        keys_to_clean = []
        for key, records in self._index.items():
            self._index[key] = [r for r in records if r.patient_id != patient_id]
            if not self._index[key]:
                keys_to_clean.append(key)
        for key in keys_to_clean:
            del self._index[key]

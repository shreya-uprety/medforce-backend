"""
Permission Checker — validates sender access for each event.

Rules:
  - Patients always have full access to their own diary.
  - Helpers are checked against their diary-registered permissions.
  - GPs can view status, upload documents, and respond to queries.
  - System / Agent senders are always allowed (internal events).
  - Unknown senders are always denied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.permissions")


class Permission(str, Enum):
    """Named permissions assignable to helpers."""

    FULL_ACCESS = "full_access"
    VIEW_STATUS = "view_status"
    SEND_MESSAGES = "send_messages"
    UPLOAD_DOCUMENTS = "upload_documents"
    RESPOND_TO_QUERIES = "respond_to_queries"
    BOOK_APPOINTMENTS = "book_appointments"


# What permission is needed for each event type in each phase
_EVENT_PERMISSION_MAP: dict[EventType, str] = {
    EventType.USER_MESSAGE: Permission.SEND_MESSAGES,
    EventType.DOCUMENT_UPLOADED: Permission.UPLOAD_DOCUMENTS,
    EventType.DOCTOR_COMMAND: Permission.FULL_ACCESS,
}


@dataclass
class PermissionResult:
    """Outcome of a permission check."""

    allowed: bool
    reason: str = ""
    required_permission: str = ""


class PermissionChecker:
    """
    Checks whether a sender is allowed to perform the action
    implied by their event.

    Maintains an audit log of every permission check (Phase 7).
    """

    def __init__(self) -> None:
        self._audit_log: list[dict] = []

    def check(
        self,
        *,
        sender_role: SenderRole | str,
        sender_permissions: list[str],
        event: EventEnvelope,
        diary_phase: str,
    ) -> PermissionResult:
        role = sender_role.value if isinstance(sender_role, SenderRole) else sender_role

        # System and agent events are always internal — allow
        if role in ("system", "agent"):
            result = PermissionResult(allowed=True, reason="internal_event")
            self._audit(event, role, sender_permissions, diary_phase, result)
            return result

        # Patients always have full access to their own diary
        if role == "patient":
            result = PermissionResult(allowed=True, reason="patient_full_access")
            self._audit(event, role, sender_permissions, diary_phase, result)
            return result

        # GP-specific checks
        if role == "gp":
            result = self._check_gp(event, sender_permissions)
            self._audit(event, role, sender_permissions, diary_phase, result)
            return result

        # Helper checks
        if role == "helper":
            result = self._check_helper(event, sender_permissions, diary_phase)
            self._audit(event, role, sender_permissions, diary_phase, result)
            return result

        # Unknown role — deny
        result = PermissionResult(
            allowed=False,
            reason="unknown_sender_role",
        )
        self._audit(event, role, sender_permissions, diary_phase, result)
        return result

    def _audit(
        self,
        event: EventEnvelope,
        role: str,
        permissions: list[str],
        phase: str,
        result: PermissionResult,
    ) -> None:
        """Record every permission check for audit trail."""
        entry = {
            "event_id": event.event_id,
            "patient_id": event.patient_id,
            "sender_id": event.sender_id,
            "sender_role": role,
            "permissions": permissions,
            "event_type": event.event_type.value,
            "diary_phase": phase,
            "allowed": result.allowed,
            "reason": result.reason,
            "timestamp": event.timestamp.isoformat(),
        }
        self._audit_log.append(entry)
        # Keep audit log bounded
        if len(self._audit_log) > 500:
            self._audit_log = self._audit_log[-250:]

        level = logging.DEBUG if result.allowed else logging.WARNING
        logger.log(
            level,
            "Permission %s: %s (%s) → %s for patient %s [phase=%s, reason=%s]",
            "GRANTED" if result.allowed else "DENIED",
            event.sender_id,
            role,
            event.event_type.value,
            event.patient_id,
            phase,
            result.reason,
        )

    @property
    def audit_log(self) -> list[dict]:
        """Access the audit log for inspection/export."""
        return list(self._audit_log)

    def _check_gp(
        self, event: EventEnvelope, permissions: list[str]
    ) -> PermissionResult:
        """GPs can view status, upload documents, and respond to queries."""
        allowed_event_types = {
            EventType.GP_RESPONSE,
            EventType.DOCUMENT_UPLOADED,
            EventType.WEBHOOK,
        }

        if event.event_type in allowed_event_types:
            return PermissionResult(allowed=True, reason="gp_allowed_action")

        # GP sending a regular message — check permissions
        if event.event_type == EventType.USER_MESSAGE:
            if Permission.SEND_MESSAGES in permissions or Permission.FULL_ACCESS in permissions:
                return PermissionResult(allowed=True, reason="gp_has_send_permission")
            return PermissionResult(
                allowed=False,
                reason="gp_cannot_send_messages",
                required_permission=Permission.SEND_MESSAGES,
            )

        return PermissionResult(
            allowed=False,
            reason="gp_action_not_allowed",
            required_permission="gp_specific_action",
        )

    def _check_helper(
        self,
        event: EventEnvelope,
        permissions: list[str],
        diary_phase: str,
    ) -> PermissionResult:
        """Helpers are checked against their registered permissions."""
        # Full access helpers can do anything
        if Permission.FULL_ACCESS in permissions:
            return PermissionResult(allowed=True, reason="helper_full_access")

        # Look up required permission for this event type
        required = _EVENT_PERMISSION_MAP.get(event.event_type)

        if required is None:
            # Event types not in the map are internal — helpers can't emit them
            return PermissionResult(
                allowed=False,
                reason="helper_cannot_emit_internal_event",
                required_permission="internal",
            )

        if required in permissions:
            return PermissionResult(allowed=True, reason="helper_has_permission")

        return PermissionResult(
            allowed=False,
            reason="helper_missing_permission",
            required_permission=required,
        )

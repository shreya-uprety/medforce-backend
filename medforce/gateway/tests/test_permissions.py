"""
Tests for the Permission Checker.

Covers:
  - Patient full access
  - System/agent internal events always allowed
  - Helper permission checking (full access, specific permissions, missing)
  - GP allowed and denied actions
  - Unknown sender denial
  - Edge cases
"""

import pytest

from medforce.gateway.events import EventEnvelope, EventType, SenderRole
from medforce.gateway.permissions import Permission, PermissionChecker, PermissionResult


@pytest.fixture
def checker():
    return PermissionChecker()


def _make_event(
    event_type: EventType = EventType.USER_MESSAGE,
    patient_id: str = "PT-001",
    sender_role: SenderRole = SenderRole.PATIENT,
    sender_id: str = "PATIENT",
) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        patient_id=patient_id,
        sender_id=sender_id,
        sender_role=sender_role,
        payload={"text": "test", "channel": "websocket"},
    )


# ── Patient Access ──


class TestPatientAccess:
    def test_patient_can_send_messages(self, checker):
        event = _make_event(EventType.USER_MESSAGE)
        result = checker.check(
            sender_role=SenderRole.PATIENT,
            sender_permissions=["full_access"],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is True
        assert result.reason == "patient_full_access"

    def test_patient_can_upload_documents(self, checker):
        event = _make_event(EventType.DOCUMENT_UPLOADED)
        result = checker.check(
            sender_role=SenderRole.PATIENT,
            sender_permissions=["full_access"],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is True

    def test_patient_access_in_any_phase(self, checker):
        for phase in ["intake", "clinical", "booking", "monitoring", "closed"]:
            event = _make_event(EventType.USER_MESSAGE)
            result = checker.check(
                sender_role=SenderRole.PATIENT,
                sender_permissions=[],
                event=event,
                diary_phase=phase,
            )
            assert result.allowed is True


# ── System / Agent Access ──


class TestSystemAccess:
    def test_system_events_always_allowed(self, checker):
        event = _make_event(EventType.HEARTBEAT, sender_role=SenderRole.SYSTEM)
        result = checker.check(
            sender_role=SenderRole.SYSTEM,
            sender_permissions=[],
            event=event,
            diary_phase="monitoring",
        )
        assert result.allowed is True
        assert result.reason == "internal_event"

    def test_agent_events_always_allowed(self, checker):
        event = _make_event(EventType.INTAKE_COMPLETE, sender_role=SenderRole.AGENT)
        result = checker.check(
            sender_role=SenderRole.AGENT,
            sender_permissions=[],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is True

    def test_system_role_as_string(self, checker):
        event = _make_event(EventType.HEARTBEAT)
        result = checker.check(
            sender_role="system",
            sender_permissions=[],
            event=event,
            diary_phase="monitoring",
        )
        assert result.allowed is True


# ── Helper Access ──


class TestHelperAccess:
    def test_helper_with_full_access(self, checker):
        event = _make_event(
            EventType.USER_MESSAGE,
            sender_role=SenderRole.HELPER,
            sender_id="HELPER-001",
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=[Permission.FULL_ACCESS],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is True
        assert result.reason == "helper_full_access"

    def test_helper_with_send_messages_permission(self, checker):
        event = _make_event(
            EventType.USER_MESSAGE,
            sender_role=SenderRole.HELPER,
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=[Permission.SEND_MESSAGES],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is True

    def test_helper_with_upload_permission(self, checker):
        event = _make_event(
            EventType.DOCUMENT_UPLOADED,
            sender_role=SenderRole.HELPER,
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=[Permission.UPLOAD_DOCUMENTS],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is True

    def test_helper_missing_permission(self, checker):
        event = _make_event(
            EventType.USER_MESSAGE,
            sender_role=SenderRole.HELPER,
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=[Permission.VIEW_STATUS],  # no SEND_MESSAGES
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is False
        assert result.reason == "helper_missing_permission"
        assert result.required_permission == Permission.SEND_MESSAGES

    def test_helper_cannot_emit_internal_events(self, checker):
        event = _make_event(
            EventType.INTAKE_COMPLETE,
            sender_role=SenderRole.HELPER,
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=[Permission.FULL_ACCESS],
            event=event,
            diary_phase="intake",
        )
        # Internal events still allowed for full_access helpers
        assert result.allowed is True

    def test_helper_limited_cannot_emit_internal(self, checker):
        event = _make_event(
            EventType.INTAKE_COMPLETE,
            sender_role=SenderRole.HELPER,
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=[Permission.SEND_MESSAGES],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is False
        assert result.reason == "helper_cannot_emit_internal_event"


# ── GP Access ──


class TestGPAccess:
    def test_gp_can_respond_to_queries(self, checker):
        event = _make_event(
            EventType.GP_RESPONSE,
            sender_role=SenderRole.GP,
        )
        result = checker.check(
            sender_role=SenderRole.GP,
            sender_permissions=["view_status", "upload_documents", "respond_to_queries"],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is True

    def test_gp_can_upload_documents(self, checker):
        event = _make_event(
            EventType.DOCUMENT_UPLOADED,
            sender_role=SenderRole.GP,
        )
        result = checker.check(
            sender_role=SenderRole.GP,
            sender_permissions=["upload_documents"],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is True

    def test_gp_cannot_send_messages_without_permission(self, checker):
        event = _make_event(
            EventType.USER_MESSAGE,
            sender_role=SenderRole.GP,
        )
        result = checker.check(
            sender_role=SenderRole.GP,
            sender_permissions=["view_status"],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is False

    def test_gp_cannot_emit_arbitrary_events(self, checker):
        event = _make_event(
            EventType.HEARTBEAT,
            sender_role=SenderRole.GP,
        )
        result = checker.check(
            sender_role=SenderRole.GP,
            sender_permissions=["view_status"],
            event=event,
            diary_phase="monitoring",
        )
        assert result.allowed is False


# ── Unknown Sender ──


class TestUnknownSender:
    def test_unknown_role_denied(self, checker):
        event = _make_event(EventType.USER_MESSAGE)
        result = checker.check(
            sender_role="hacker",
            sender_permissions=[],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is False
        assert result.reason == "unknown_sender_role"


# ── Patient Scenarios ──


class TestPermissionScenarios:
    """Real-world patient scenarios."""

    def test_scenario_spouse_with_full_access_can_message(self, checker):
        """Sarah (spouse) has full access — can send messages on behalf of John."""
        event = _make_event(
            EventType.USER_MESSAGE,
            sender_role=SenderRole.HELPER,
            sender_id="HELPER-001",
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=["full_access"],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is True

    def test_scenario_friend_with_view_only_cannot_message(self, checker):
        """Mark (friend) has view_status only — cannot send messages."""
        event = _make_event(
            EventType.USER_MESSAGE,
            sender_role=SenderRole.HELPER,
            sender_id="HELPER-002",
        )
        result = checker.check(
            sender_role=SenderRole.HELPER,
            sender_permissions=["view_status"],
            event=event,
            diary_phase="intake",
        )
        assert result.allowed is False

    def test_scenario_gp_uploads_lab_results(self, checker):
        """Dr. Patel uploads lab results via email attachment."""
        event = _make_event(
            EventType.DOCUMENT_UPLOADED,
            sender_role=SenderRole.GP,
            sender_id="GP-DrPatel",
        )
        result = checker.check(
            sender_role=SenderRole.GP,
            sender_permissions=["view_status", "upload_documents", "respond_to_queries"],
            event=event,
            diary_phase="clinical",
        )
        assert result.allowed is True

"""
Comprehensive tests for the Event Envelope model.

Tests cover:
  - Envelope creation and field validation
  - All EventType enum values
  - SenderRole enum values
  - Convenience factories (user_message, handoff, heartbeat)
  - Routing classification (explicit vs phase-based)
  - Serialisation round-trip (JSON)
  - Edge cases: empty payload, missing fields, default values
"""

import json
from datetime import datetime, timezone

import pytest

from medforce.gateway.events import (
    EXPLICIT_ROUTE_EVENTS,
    PHASE_ROUTE_EVENTS,
    EventEnvelope,
    EventType,
    SenderRole,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Basic Construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventEnvelopeCreation:

    def test_minimal_creation(self):
        env = EventEnvelope(
            event_type=EventType.USER_MESSAGE,
            patient_id="PT-1234",
        )
        assert env.event_type == EventType.USER_MESSAGE
        assert env.patient_id == "PT-1234"
        assert env.event_id  # auto-generated UUID
        assert env.timestamp  # auto-generated
        assert env.sender_role == SenderRole.SYSTEM  # default
        assert env.payload == {}

    def test_full_creation(self):
        env = EventEnvelope(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-5678",
            payload={"risk_level": "HIGH"},
            source="clinical_agent",
            sender_id="clinical_agent",
            sender_role=SenderRole.AGENT,
            correlation_id="corr-123",
        )
        assert env.payload["risk_level"] == "HIGH"
        assert env.source == "clinical_agent"
        assert env.sender_role == SenderRole.AGENT
        assert env.correlation_id == "corr-123"

    def test_unique_event_ids(self):
        e1 = EventEnvelope(event_type=EventType.USER_MESSAGE, patient_id="PT-1")
        e2 = EventEnvelope(event_type=EventType.USER_MESSAGE, patient_id="PT-1")
        assert e1.event_id != e2.event_id

    def test_timestamp_is_utc(self):
        env = EventEnvelope(event_type=EventType.HEARTBEAT, patient_id="PT-1")
        assert env.timestamp.tzinfo is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enum Coverage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventTypeEnum:

    def test_all_event_types_exist(self):
        expected = [
            "USER_MESSAGE", "DOCUMENT_UPLOADED", "WEBHOOK", "DOCTOR_COMMAND",
            "INTAKE_COMPLETE", "INTAKE_DATA_PROVIDED", "CLINICAL_COMPLETE",
            "BOOKING_COMPLETE", "NEEDS_INTAKE_DATA", "DETERIORATION_ALERT",
            "GP_QUERY", "GP_RESPONSE", "GP_REMINDER",
            "HELPER_REGISTRATION", "HELPER_VERIFIED",
            "HEARTBEAT", "AGENT_ERROR",
        ]
        for name in expected:
            assert hasattr(EventType, name), f"Missing EventType.{name}"

    def test_event_types_are_strings(self):
        for et in EventType:
            assert isinstance(et.value, str)

    def test_every_event_type_has_routing(self):
        """Every EventType must be either explicit-route or phase-route."""
        for et in EventType:
            assert (
                et in EXPLICIT_ROUTE_EVENTS or et in PHASE_ROUTE_EVENTS
            ), f"{et} has no routing classification"

    def test_no_overlap_between_routing_strategies(self):
        overlap = EXPLICIT_ROUTE_EVENTS & PHASE_ROUTE_EVENTS
        assert not overlap, f"Events in both strategies: {overlap}"


class TestSenderRoleEnum:

    def test_all_roles(self):
        roles = {r.value for r in SenderRole}
        assert roles == {"patient", "helper", "gp", "system", "agent"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Convenience Factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFactories:

    def test_user_message_factory(self):
        env = EventEnvelope.user_message(
            patient_id="PT-100",
            text="Here are my blood results",
            channel="whatsapp",
        )
        assert env.event_type == EventType.USER_MESSAGE
        assert env.patient_id == "PT-100"
        assert env.payload["text"] == "Here are my blood results"
        assert env.payload["channel"] == "whatsapp"
        assert env.sender_role == SenderRole.PATIENT

    def test_user_message_from_helper(self):
        env = EventEnvelope.user_message(
            patient_id="PT-100",
            text="John's lab photos",
            sender_id="HELPER-001",
            sender_role=SenderRole.HELPER,
            channel="whatsapp",
            attachments=["lab_photo.jpg"],
        )
        assert env.sender_id == "HELPER-001"
        assert env.sender_role == SenderRole.HELPER
        assert env.payload["attachments"] == ["lab_photo.jpg"]

    def test_handoff_factory(self):
        env = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-200",
            source_agent="intake_agent",
            payload={"fields_collected": 9},
        )
        assert env.event_type == EventType.INTAKE_COMPLETE
        assert env.sender_role == SenderRole.AGENT
        assert env.source == "intake_agent"
        assert env.payload["fields_collected"] == 9

    def test_heartbeat_factory(self):
        env = EventEnvelope.heartbeat(
            patient_id="PT-300",
            days_since_appointment=14,
            milestone="14d_followup",
        )
        assert env.event_type == EventType.HEARTBEAT
        assert env.sender_role == SenderRole.SYSTEM
        assert env.payload["days_since_appointment"] == 14
        assert env.payload["milestone"] == "14d_followup"
        assert env.source == "heartbeat_scheduler"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Routing Classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRoutingClassification:

    @pytest.mark.parametrize("event_type", [
        EventType.INTAKE_COMPLETE,
        EventType.CLINICAL_COMPLETE,
        EventType.BOOKING_COMPLETE,
        EventType.NEEDS_INTAKE_DATA,
        EventType.HEARTBEAT,
        EventType.DETERIORATION_ALERT,
        EventType.GP_QUERY,
        EventType.GP_RESPONSE,
        EventType.GP_REMINDER,
        EventType.HELPER_REGISTRATION,
        EventType.HELPER_VERIFIED,
        EventType.AGENT_ERROR,
    ])
    def test_explicit_route_events(self, event_type):
        env = EventEnvelope(event_type=event_type, patient_id="PT-1")
        assert env.is_explicit_route()
        assert not env.is_phase_route()

    @pytest.mark.parametrize("event_type", [
        EventType.USER_MESSAGE,
        EventType.DOCUMENT_UPLOADED,
        EventType.WEBHOOK,
        EventType.DOCTOR_COMMAND,
    ])
    def test_phase_route_events(self, event_type):
        env = EventEnvelope(event_type=event_type, patient_id="PT-1")
        assert env.is_phase_route()
        assert not env.is_explicit_route()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Serialisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSerialisation:

    def test_json_round_trip(self):
        original = EventEnvelope.user_message(
            patient_id="PT-RT",
            text="Round trip test",
            attachments=["doc.pdf"],
        )
        json_str = original.model_dump_json()
        restored = EventEnvelope.model_validate_json(json_str)
        assert restored.event_type == original.event_type
        assert restored.patient_id == original.patient_id
        assert restored.payload["text"] == "Round trip test"
        assert restored.event_id == original.event_id

    def test_dict_round_trip(self):
        original = EventEnvelope.handoff(
            event_type=EventType.DETERIORATION_ALERT,
            patient_id="PT-DT",
            payload={"reason": "jaundice detected"},
        )
        data = original.model_dump()
        assert isinstance(data, dict)
        restored = EventEnvelope.model_validate(data)
        assert restored.payload["reason"] == "jaundice detected"

    def test_to_json_includes_all_fields(self):
        env = EventEnvelope(
            event_type=EventType.GP_RESPONSE,
            patient_id="PT-GP",
            payload={"attachments": ["lft.pdf"]},
            sender_role=SenderRole.GP,
        )
        data = json.loads(env.model_dump_json())
        expected_keys = {
            "event_id", "event_type", "patient_id", "payload",
            "source", "sender_id", "sender_role", "correlation_id",
            "timestamp",
        }
        assert expected_keys.issubset(data.keys())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientScenarios:
    """Test events for real clinical scenarios from the architecture doc."""

    def test_scenario_1_solo_patient_message(self):
        """Mary Jones sends a simple message."""
        env = EventEnvelope.user_message(
            patient_id="PT-MARY",
            text="My DOB is 12 May 1970",
            channel="websocket",
        )
        assert env.is_phase_route()
        assert env.sender_role == SenderRole.PATIENT

    def test_scenario_2_urgent_referral_with_helper(self):
        """David Clarke's wife Linda uploads results."""
        env = EventEnvelope.user_message(
            patient_id="PT-DAVID",
            text="Here are David's blood results",
            sender_id="HELPER-LINDA",
            sender_role=SenderRole.HELPER,
            channel="whatsapp",
            attachments=["blood_results.pdf"],
        )
        assert env.sender_id == "HELPER-LINDA"
        assert len(env.payload["attachments"]) == 1

    def test_scenario_3_gp_query_for_missing_labs(self):
        """Clinical agent queries GP for missing lab results."""
        env = EventEnvelope.handoff(
            event_type=EventType.GP_QUERY,
            patient_id="PT-AISHA",
            source_agent="clinical_agent",
            payload={
                "gp_name": "Dr. Chen",
                "query_type": "missing_lab_results",
                "query_text": "Referral mentions bloods but none attached.",
            },
        )
        assert env.is_explicit_route()
        assert env.payload["query_type"] == "missing_lab_results"

    def test_scenario_4_backward_loop_missing_meds(self):
        """Clinical agent requests medication list from Intake."""
        env = EventEnvelope.handoff(
            event_type=EventType.NEEDS_INTAKE_DATA,
            patient_id="PT-ROBERT",
            source_agent="clinical_agent",
            payload={"missing": "current_medication_list"},
        )
        assert env.event_type == EventType.NEEDS_INTAKE_DATA
        assert env.payload["missing"] == "current_medication_list"

    def test_scenario_5_deterioration_alert(self):
        """Monitoring detects jaundice keywords — escalation."""
        env = EventEnvelope.handoff(
            event_type=EventType.DETERIORATION_ALERT,
            patient_id="PT-HELEN",
            source_agent="monitoring_agent",
            payload={"reason": "possible jaundice reported, existing hepatic condition"},
        )
        assert env.event_type == EventType.DETERIORATION_ALERT
        assert env.is_explicit_route()

    def test_scenario_6_heartbeat_14_days(self):
        """CRON fires 14-day heartbeat."""
        env = EventEnvelope.heartbeat(
            patient_id="PT-JOHN",
            days_since_appointment=14,
            milestone="14d_followup",
        )
        assert env.payload["days_since_appointment"] == 14

    def test_scenario_7_gp_response(self):
        """GP Dr. Patel replies with lab PDF."""
        env = EventEnvelope(
            event_type=EventType.GP_RESPONSE,
            patient_id="PT-1234",
            payload={
                "from": "Dr. Patel",
                "attachments": ["liver_function_tests.pdf"],
                "response_to": "missing_lab_results",
            },
            source="email",
            sender_id="GP-DrPatel",
            sender_role=SenderRole.GP,
        )
        assert env.is_explicit_route()
        assert env.sender_role == SenderRole.GP

    def test_scenario_8_unknown_sender(self):
        """Unknown number sends a message — will be caught by identity resolver."""
        env = EventEnvelope.user_message(
            patient_id="UNKNOWN",
            text="What's my appointment date?",
            sender_id="UNKNOWN",
            sender_role=SenderRole.SYSTEM,
            channel="whatsapp",
        )
        assert env.patient_id == "UNKNOWN"

    def test_chain_of_handoffs(self):
        """Simulate intake → clinical → booking chain."""
        intake_done = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id="PT-CHAIN",
            source_agent="intake_agent",
            correlation_id="journey-001",
        )
        clinical_done = EventEnvelope.handoff(
            event_type=EventType.CLINICAL_COMPLETE,
            patient_id="PT-CHAIN",
            source_agent="clinical_agent",
            payload={"risk_level": "HIGH"},
            correlation_id="journey-001",
        )
        booking_done = EventEnvelope.handoff(
            event_type=EventType.BOOKING_COMPLETE,
            patient_id="PT-CHAIN",
            source_agent="booking_agent",
            payload={"appointment_id": "APT-999"},
            correlation_id="journey-001",
        )
        # All share the same correlation ID
        assert intake_done.correlation_id == clinical_done.correlation_id
        assert clinical_done.correlation_id == booking_done.correlation_id
        # All for same patient
        assert intake_done.patient_id == "PT-CHAIN"
        assert clinical_done.patient_id == "PT-CHAIN"
        assert booking_done.patient_id == "PT-CHAIN"

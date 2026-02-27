"""
Event Envelope — Universal event format for the MedForce Gateway.

Every signal that enters the control loop (patient message, lab webhook,
CRON heartbeat, agent handoff, GP reply, helper upload) is wrapped in
the same EventEnvelope. The Gateway only reads envelope metadata for
routing — it never inspects the payload.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """All event types recognised by the Gateway."""

    # External events (from patients, helpers, GPs, external systems)
    USER_MESSAGE = "USER_MESSAGE"
    DOCUMENT_UPLOADED = "DOCUMENT_UPLOADED"
    WEBHOOK = "WEBHOOK"
    DOCTOR_COMMAND = "DOCTOR_COMMAND"

    # Agent handoff events (internal, looped back through Gateway)
    INTAKE_COMPLETE = "INTAKE_COMPLETE"
    INTAKE_DATA_PROVIDED = "INTAKE_DATA_PROVIDED"
    CLINICAL_COMPLETE = "CLINICAL_COMPLETE"
    BOOKING_COMPLETE = "BOOKING_COMPLETE"
    NEEDS_INTAKE_DATA = "NEEDS_INTAKE_DATA"
    DETERIORATION_ALERT = "DETERIORATION_ALERT"
    RESCHEDULE_REQUEST = "RESCHEDULE_REQUEST"

    # GP communication events
    GP_QUERY = "GP_QUERY"
    GP_RESPONSE = "GP_RESPONSE"
    GP_REMINDER = "GP_REMINDER"

    # Helper management events
    HELPER_REGISTRATION = "HELPER_REGISTRATION"
    HELPER_VERIFIED = "HELPER_VERIFIED"

    # Cross-phase content routing
    CROSS_PHASE_DATA = "CROSS_PHASE_DATA"
    CROSS_PHASE_REPROMPT = "CROSS_PHASE_REPROMPT"

    # Form-based intake
    INTAKE_FORM_SUBMITTED = "INTAKE_FORM_SUBMITTED"

    # System events
    HEARTBEAT = "HEARTBEAT"
    AGENT_ERROR = "AGENT_ERROR"


class SenderRole(str, Enum):
    """Who sent the event."""

    PATIENT = "patient"
    HELPER = "helper"
    GP = "gp"
    SYSTEM = "system"
    AGENT = "agent"


# Events that the Gateway routes via Strategy A (explicit target)
EXPLICIT_ROUTE_EVENTS = {
    EventType.INTAKE_COMPLETE,
    EventType.INTAKE_DATA_PROVIDED,
    EventType.CLINICAL_COMPLETE,
    EventType.BOOKING_COMPLETE,
    EventType.NEEDS_INTAKE_DATA,
    EventType.HEARTBEAT,
    EventType.DETERIORATION_ALERT,
    EventType.RESCHEDULE_REQUEST,
    EventType.GP_QUERY,
    EventType.GP_RESPONSE,
    EventType.GP_REMINDER,
    EventType.HELPER_REGISTRATION,
    EventType.HELPER_VERIFIED,
    EventType.AGENT_ERROR,
    EventType.CROSS_PHASE_DATA,
    EventType.CROSS_PHASE_REPROMPT,
    EventType.INTAKE_FORM_SUBMITTED,
}

# Events that the Gateway routes via Strategy B (diary phase lookup)
PHASE_ROUTE_EVENTS = {
    EventType.USER_MESSAGE,
    EventType.DOCUMENT_UPLOADED,
    EventType.WEBHOOK,
    EventType.DOCTOR_COMMAND,
}


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventEnvelope(BaseModel):
    """Universal event wrapper — the only object that enters the Gateway loop."""

    event_id: str = Field(default_factory=_new_uuid)
    event_type: EventType
    patient_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    sender_id: str = ""
    sender_role: SenderRole = SenderRole.SYSTEM
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=_now)
    # Internal tracking — not part of the external contract
    _chain_depth: int = 0

    model_config = {"use_enum_values": False}

    # ── Convenience factories ──

    @classmethod
    def user_message(
        cls,
        patient_id: str,
        text: str,
        *,
        sender_id: str = "PATIENT",
        sender_role: SenderRole = SenderRole.PATIENT,
        channel: str = "websocket",
        attachments: list[str] | None = None,
        correlation_id: str | None = None,
    ) -> EventEnvelope:
        return cls(
            event_type=EventType.USER_MESSAGE,
            patient_id=patient_id,
            payload={
                "text": text,
                "channel": channel,
                "attachments": attachments or [],
            },
            source=channel,
            sender_id=sender_id,
            sender_role=sender_role,
            correlation_id=correlation_id,
        )

    @classmethod
    def handoff(
        cls,
        event_type: EventType,
        patient_id: str,
        *,
        source_agent: str = "",
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> EventEnvelope:
        return cls(
            event_type=event_type,
            patient_id=patient_id,
            payload=payload or {},
            source=source_agent,
            sender_id=source_agent,
            sender_role=SenderRole.AGENT,
            correlation_id=correlation_id,
        )

    @classmethod
    def heartbeat(
        cls,
        patient_id: str,
        *,
        days_since_appointment: int = 0,
        milestone: str = "",
    ) -> EventEnvelope:
        return cls(
            event_type=EventType.HEARTBEAT,
            patient_id=patient_id,
            payload={
                "days_since_appointment": days_since_appointment,
                "milestone": milestone,
            },
            source="heartbeat_scheduler",
            sender_id="system",
            sender_role=SenderRole.SYSTEM,
        )

    def is_explicit_route(self) -> bool:
        """True if this event uses Strategy A (hardcoded target)."""
        return self.event_type in EXPLICIT_ROUTE_EVENTS

    def is_phase_route(self) -> bool:
        """True if this event uses Strategy B (diary phase lookup)."""
        return self.event_type in PHASE_ROUTE_EVENTS

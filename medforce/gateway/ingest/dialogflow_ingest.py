"""
Dialogflow Ingest — converts Dialogflow CX fulfillment webhook bodies
into EventEnvelopes for the Gateway.

Dialogflow CX is configured as a "pass-through" — no intent matching,
no flow logic. Every user message triggers the fulfillment webhook
immediately, and this ingest converts it into the Gateway's event format.

Expected webhook body structure (Dialogflow CX fulfillment):
{
  "detectIntentResponseId": "...",
  "intentInfo": {...},
  "pageInfo": {...},
  "sessionInfo": {
    "session": "projects/.../sessions/SESSION_ID",
    "parameters": {
      "phone": "+447700900001",
      ...
    }
  },
  "fulfillmentInfo": {"tag": "..."},
  "text": "Patient's message text",
  "languageCode": "en"
}
"""

from __future__ import annotations

import logging
import re
from typing import Any

from medforce.gateway.channels import ChannelIngest
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.ingest.dialogflow")


class DialogflowIngest(ChannelIngest):
    """Converts Dialogflow CX fulfillment webhook body into EventEnvelope."""

    channel_name = "dialogflow_whatsapp"

    def __init__(self, identity_resolver=None) -> None:
        self._identity_resolver = identity_resolver

    async def to_envelope(self, raw_input: dict[str, Any]) -> EventEnvelope:
        """
        Parse Dialogflow webhook body into an EventEnvelope.

        Resolution flow:
          1. Extract sender phone from sessionInfo.parameters or session ID
          2. Extract message text
          3. Use IdentityResolver to map phone → (patient_id, role, permissions)
          4. Build and return EventEnvelope
        """
        # Extract session info
        session_info = raw_input.get("sessionInfo", {})
        parameters = session_info.get("parameters", {})
        session_path = session_info.get("session", "")

        # Extract phone — from parameters or session ID
        phone = parameters.get("phone", "")
        if not phone:
            # Try to extract from session path (last segment)
            session_id = session_path.rsplit("/", 1)[-1] if session_path else ""
            if session_id and re.match(r"^\+?\d{10,15}$", session_id):
                phone = session_id

        # Extract message text
        text = raw_input.get("text", "")
        if not text:
            # Fallback: check transcript
            text = raw_input.get("transcript", "")

        # Extract media/attachments
        media = raw_input.get("media", [])
        attachments = [m.get("url", "") for m in media if m.get("url")]

        # Determine event type
        event_type = EventType.USER_MESSAGE
        if attachments:
            event_type = EventType.DOCUMENT_UPLOADED

        # Resolve identity
        patient_id = ""
        sender_id = phone
        sender_role = SenderRole.PATIENT
        permissions = []

        if self._identity_resolver and phone:
            from medforce.gateway.handlers.identity_resolver import (
                AmbiguousIdentity,
                IdentityRecord,
            )

            identity = self._identity_resolver.resolve(phone)
            if isinstance(identity, IdentityRecord):
                patient_id = identity.patient_id
                sender_id = identity.sender_id
                sender_role = SenderRole(identity.sender_role)
                permissions = identity.permissions
            elif isinstance(identity, AmbiguousIdentity):
                # Multiple patients — use first match for now
                # Phase 7 will add disambiguation
                record = identity.records[0]
                patient_id = record.patient_id
                sender_id = record.sender_id
                sender_role = SenderRole(record.sender_role)
                permissions = record.permissions
                logger.warning(
                    "Ambiguous identity for %s — defaulting to patient %s",
                    phone, patient_id,
                )

        return self._build_base_envelope(
            event_type=event_type,
            patient_id=patient_id,
            sender_id=sender_id,
            sender_role=sender_role,
            payload={
                "text": text,
                "phone": phone,
                "attachments": attachments,
                "channel": self.channel_name,
                "raw_session": session_path,
                "permissions": permissions,
            },
        )

    def build_dialogflow_response(
        self, responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Build a Dialogflow CX fulfillment response from Gateway responses.

        Dialogflow expects:
        {
          "fulfillmentResponse": {
            "messages": [
              {"text": {"text": ["message1"]}},
              {"text": {"text": ["message2"]}}
            ]
          }
        }
        """
        messages = []
        for resp in responses:
            text = resp.get("message", "")
            if text:
                messages.append({"text": {"text": [text]}})

        return {"fulfillmentResponse": {"messages": messages}}

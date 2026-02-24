"""
Twilio Ingest — converts Twilio SMS/WhatsApp webhook bodies into
EventEnvelopes for the Gateway.

Twilio sends a POST with form data for each incoming message.
This ingest handles both SMS and WhatsApp messages from Twilio.

Expected webhook body (Twilio incoming message):
{
  "MessageSid": "SM...",
  "AccountSid": "AC...",
  "From": "+447700900001",       # or "whatsapp:+447700900001"
  "To": "+441234567890",         # your Twilio number
  "Body": "Message text",
  "NumMedia": "1",
  "MediaUrl0": "https://...",
  "MediaContentType0": "image/jpeg"
}
"""

from __future__ import annotations

import logging
import re
from typing import Any

from medforce.gateway.channels import ChannelIngest
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.ingest.twilio")


class TwilioSMSIngest(ChannelIngest):
    """Converts Twilio incoming SMS/WhatsApp webhook into EventEnvelope."""

    channel_name = "sms"

    def __init__(self, identity_resolver=None) -> None:
        self._identity_resolver = identity_resolver

    async def to_envelope(self, raw_input: dict[str, Any]) -> EventEnvelope:
        """
        Parse Twilio webhook body into an EventEnvelope.

        Resolution flow:
          1. Extract sender phone (strip whatsapp: prefix if present)
          2. Extract message text
          3. Extract media URLs
          4. Use IdentityResolver to map phone → identity
          5. Build EventEnvelope
        """
        # Extract sender phone
        from_number = raw_input.get("From", "")
        is_whatsapp = from_number.startswith("whatsapp:")
        phone = from_number.replace("whatsapp:", "").strip()

        # Determine channel based on source
        channel = "dialogflow_whatsapp" if is_whatsapp else "sms"

        # Extract message
        text = raw_input.get("Body", "")

        # Extract media
        num_media = int(raw_input.get("NumMedia", "0") or "0")
        media_urls = []
        for i in range(num_media):
            url = raw_input.get(f"MediaUrl{i}", "")
            if url:
                media_urls.append(url)

        # Determine event type
        event_type = EventType.USER_MESSAGE
        if media_urls:
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
                "attachments": media_urls,
                "channel": channel,
                "is_whatsapp": is_whatsapp,
                "message_sid": raw_input.get("MessageSid", ""),
                "permissions": permissions,
            },
        )

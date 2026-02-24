"""
Email Ingest — converts SendGrid inbound parse webhook bodies into
EventEnvelopes for the Gateway.

Used primarily for GP responses: GPs reply to emails sent by the
GP Communication Handler, and SendGrid's Inbound Parse forwards
the reply to this webhook.

The reply-to address encodes the patient_id:
  gp-reply+PT-1234@medforce.app → patient_id = "PT-1234"

Expected webhook body (SendGrid Inbound Parse — parsed format):
{
  "from": "dr.patel@greenfields.nhs.uk",
  "to": "gp-reply+PT-1234@medforce.app",
  "subject": "Re: MedForce — Lab Results Requested",
  "text": "Plain text body of the GP's reply",
  "html": "<html>...</html>",
  "attachments": "2",
  "attachment1": <file>,
  "attachment2": <file>,
  "envelope": '{"from":"dr.patel@nhs.uk","to":["gp-reply+PT-1234@medforce.app"]}'
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from medforce.gateway.channels import ChannelIngest
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.ingest.email")

# Regex to extract patient_id from reply-to address
REPLY_TO_PATTERN = re.compile(r"gp-reply\+([A-Za-z0-9\-_]+)@")


class EmailIngest(ChannelIngest):
    """Converts SendGrid inbound parse webhook into EventEnvelope."""

    channel_name = "email"

    def __init__(self, identity_resolver=None) -> None:
        self._identity_resolver = identity_resolver

    async def to_envelope(self, raw_input: dict[str, Any]) -> EventEnvelope:
        """
        Parse SendGrid inbound email into an EventEnvelope.

        Resolution flow:
          1. Extract patient_id from reply-to address
          2. Extract sender email (GP's email)
          3. Use IdentityResolver to verify GP identity
          4. Extract message body and attachments
          5. Determine event type (GP_RESPONSE or DOCUMENT_UPLOADED)
        """
        # Extract sender
        sender_email = raw_input.get("from", "")
        # Clean email — may be "Dr. Patel <dr.patel@nhs.uk>"
        email_match = re.search(r"<([^>]+)>", sender_email)
        if email_match:
            sender_email = email_match.group(1)
        sender_email = sender_email.strip().lower()

        # Extract patient_id from reply-to address
        to_address = raw_input.get("to", "")
        patient_id = ""
        reply_match = REPLY_TO_PATTERN.search(to_address)
        if reply_match:
            patient_id = reply_match.group(1)

        # Extract message body — prefer plain text over HTML
        text = raw_input.get("text", "")
        if not text:
            text = raw_input.get("html", "")

        subject = raw_input.get("subject", "")

        # Extract attachment count
        attachment_count = int(raw_input.get("attachments", "0") or "0")
        attachment_refs = [
            f"email_attachment_{i + 1}"
            for i in range(attachment_count)
        ]

        # Determine event type
        if attachment_count > 0:
            event_type = EventType.DOCUMENT_UPLOADED
        else:
            event_type = EventType.GP_RESPONSE

        # Resolve identity
        sender_id = f"GP-{sender_email}"
        sender_role = SenderRole.GP

        if self._identity_resolver and sender_email:
            from medforce.gateway.handlers.identity_resolver import (
                IdentityRecord,
            )

            identity = self._identity_resolver.resolve(sender_email)
            if isinstance(identity, IdentityRecord):
                patient_id = patient_id or identity.patient_id
                sender_id = identity.sender_id
                sender_role = SenderRole(identity.sender_role)

        return self._build_base_envelope(
            event_type=event_type,
            patient_id=patient_id,
            sender_id=sender_id,
            sender_role=sender_role,
            payload={
                "text": text,
                "subject": subject,
                "sender_email": sender_email,
                "attachments": attachment_refs,
                "attachment_count": attachment_count,
                "channel": self.channel_name,
                "source": "gp_email_reply",
            },
        )

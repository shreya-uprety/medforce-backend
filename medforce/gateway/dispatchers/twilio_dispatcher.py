"""
Twilio SMS Dispatcher — delivers responses via Twilio SMS API.

Alternative to Dialogflow for SMS-only delivery. Can run alongside
Dialogflow (Dialogflow handles WhatsApp, Twilio handles SMS).

Configuration (environment variables):
  TWILIO_ACCOUNT_SID   — Twilio account SID
  TWILIO_AUTH_TOKEN     — Twilio auth token
  TWILIO_FROM_NUMBER   — Twilio phone number (e.g., "+441234567890")
"""

from __future__ import annotations

import logging
import os

from medforce.gateway.channels import (
    AgentResponse,
    ChannelDispatcher,
    DeliveryResult,
)

logger = logging.getLogger("gateway.dispatchers.twilio")


class TwilioSMSDispatcher(ChannelDispatcher):
    """Delivers responses via Twilio SMS API."""

    channel_name = "sms"

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
    ) -> None:
        self._account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self._auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")
        self._from_number = from_number or os.getenv("TWILIO_FROM_NUMBER", "")
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Twilio client."""
        if self._client is None:
            if not self._account_sid or not self._auth_token:
                logger.warning(
                    "Twilio credentials not set — SMS dispatcher in stub mode"
                )
                return None
            try:
                from twilio.rest import Client

                self._client = Client(self._account_sid, self._auth_token)
                logger.info("Twilio client initialized")
            except ImportError:
                logger.warning(
                    "twilio package not installed — "
                    "SMS dispatcher will operate in stub mode"
                )
            except Exception as exc:
                logger.error("Failed to initialize Twilio client: %s", exc)
        return self._client

    async def send(self, response: AgentResponse) -> DeliveryResult:
        """Send an SMS via Twilio."""
        to_number = response.metadata.get("phone", "")

        if not to_number:
            logger.warning("Twilio dispatch: no 'phone' in metadata")
            return DeliveryResult(
                success=False,
                channel=self.channel_name,
                recipient=response.recipient,
                error="No recipient phone number",
            )

        # Truncate message if over SMS limit (160 chars for single,
        # 1600 for concatenated). Twilio handles concatenation automatically.
        message_text = response.message
        if len(message_text) > 1600:
            message_text = message_text[:1597] + "..."

        client = self._get_client()
        if client is None:
            logger.info(
                "Twilio stub: SMS → %s: %s",
                to_number, message_text[:80],
            )
            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
                error="stub_mode",
            )

        try:
            sms = client.messages.create(
                body=message_text,
                from_=self._from_number,
                to=to_number,
            )

            logger.info(
                "Twilio SMS sent: SID=%s → %s",
                sms.sid, to_number,
            )
            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
            )

        except Exception as exc:
            logger.error("Twilio SMS send error: %s", exc)
            return DeliveryResult(
                success=False,
                channel=self.channel_name,
                recipient=response.recipient,
                error=str(exc),
            )

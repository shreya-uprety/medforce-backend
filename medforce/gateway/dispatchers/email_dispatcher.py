"""
Email Dispatcher — delivers responses via SendGrid for GP communication.

GP queries, reminders, and follow-ups are sent as emails. The GP Communication
Handler generates the content; this dispatcher handles delivery.

Configuration (environment variables):
  SENDGRID_API_KEY     — SendGrid API key
  SENDGRID_FROM_EMAIL  — Sender email (e.g., "noreply@medforce.app")
  SENDGRID_FROM_NAME   — Sender display name (default: "MedForce Clinical")
"""

from __future__ import annotations

import logging
import os

from medforce.gateway.channels import (
    AgentResponse,
    ChannelDispatcher,
    DeliveryResult,
)

logger = logging.getLogger("gateway.dispatchers.email")


class EmailDispatcher(ChannelDispatcher):
    """Delivers responses via SendGrid email API."""

    channel_name = "email"

    def __init__(
        self,
        api_key: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("SENDGRID_API_KEY", "")
        self._from_email = from_email or os.getenv(
            "SENDGRID_FROM_EMAIL", "noreply@medforce.app"
        )
        self._from_name = from_name or os.getenv(
            "SENDGRID_FROM_NAME", "MedForce Clinical"
        )
        self._client = None

    def _get_client(self):
        """Lazy-initialize the SendGrid client."""
        if self._client is None:
            if not self._api_key:
                logger.warning(
                    "SENDGRID_API_KEY not set — email dispatcher in stub mode"
                )
                return None
            try:
                from sendgrid import SendGridAPIClient

                self._client = SendGridAPIClient(self._api_key)
                logger.info("SendGrid client initialized")
            except ImportError:
                logger.warning(
                    "sendgrid package not installed — "
                    "email dispatcher will operate in stub mode"
                )
            except Exception as exc:
                logger.error("Failed to initialize SendGrid client: %s", exc)
        return self._client

    async def send(self, response: AgentResponse) -> DeliveryResult:
        """Send an email via SendGrid."""
        to_email = response.metadata.get("to", "")
        subject = response.metadata.get(
            "subject", "MedForce — Clinical Update"
        )
        reply_to = response.metadata.get("reply_to", "")

        if not to_email:
            logger.warning("Email dispatch: no 'to' address in metadata")
            return DeliveryResult(
                success=False,
                channel=self.channel_name,
                recipient=response.recipient,
                error="No recipient email address",
            )

        client = self._get_client()
        if client is None:
            # Stub mode
            logger.info(
                "Email stub: %s → %s: %s",
                subject, to_email, response.message[:80],
            )
            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
                error="stub_mode",
            )

        try:
            from sendgrid.helpers.mail import (
                Email,
                Mail,
                ReplyTo,
                To,
            )

            message = Mail(
                from_email=Email(self._from_email, self._from_name),
                to_emails=To(to_email),
                subject=subject,
                plain_text_content=response.message,
            )

            if reply_to:
                message.reply_to = ReplyTo(reply_to)

            # Add attachments if any
            if response.attachments:
                for attachment_ref in response.attachments:
                    logger.info(
                        "Email attachment reference: %s (not attached — "
                        "attachment resolution not yet implemented)",
                        attachment_ref,
                    )

            sg_response = client.send(message)
            status_code = sg_response.status_code

            if 200 <= status_code < 300:
                logger.info(
                    "Email sent: %s → %s (status=%d)",
                    subject, to_email, status_code,
                )
                return DeliveryResult(
                    success=True,
                    channel=self.channel_name,
                    recipient=response.recipient,
                )
            else:
                logger.error(
                    "Email send failed: status=%d body=%s",
                    status_code, sg_response.body,
                )
                return DeliveryResult(
                    success=False,
                    channel=self.channel_name,
                    recipient=response.recipient,
                    error=f"SendGrid returned status {status_code}",
                )

        except Exception as exc:
            logger.error("Email send error: %s", exc)
            return DeliveryResult(
                success=False,
                channel=self.channel_name,
                recipient=response.recipient,
                error=str(exc),
            )

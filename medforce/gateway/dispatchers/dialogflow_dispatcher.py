"""
Dialogflow Dispatcher — delivers responses via Dialogflow CX → WhatsApp/SMS.

Sends outbound messages through the Dialogflow CX Sessions API, which then
routes to WhatsApp Business API or SMS via configured integrations.

Handles two message types:
  - Proactive messages: Uses WhatsApp-approved templates (heartbeat reminders,
    booking confirmations, deterioration alerts)
  - Reactive messages: Reply within existing conversation session

Configuration (environment variables):
  DIALOGFLOW_PROJECT_ID    — GCP project ID
  DIALOGFLOW_LOCATION      — Dialogflow CX location (e.g., "europe-west2")
  DIALOGFLOW_AGENT_ID      — Dialogflow CX agent ID
  DIALOGFLOW_ENVIRONMENT   — "draft" or production environment ID
"""

from __future__ import annotations

import logging
import os
from typing import Any

from medforce.gateway.channels import (
    AgentResponse,
    ChannelDispatcher,
    DeliveryResult,
)

logger = logging.getLogger("gateway.dispatchers.dialogflow")

# WhatsApp-approved template IDs for proactive messages
WHATSAPP_TEMPLATES = {
    "heartbeat_reminder": "medforce_checkin_reminder",
    "booking_confirmation": "medforce_booking_confirmed",
    "deterioration_alert": "medforce_urgent_update",
    "gp_query_update": "medforce_gp_update",
    "default": "medforce_general_notification",
}


class DialogflowDispatcher(ChannelDispatcher):
    """Delivers responses via Dialogflow CX → WhatsApp Business / SMS."""

    channel_name = "dialogflow_whatsapp"

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        agent_id: str | None = None,
        environment: str | None = None,
    ) -> None:
        self._project_id = project_id or os.getenv("DIALOGFLOW_PROJECT_ID", "")
        self._location = location or os.getenv("DIALOGFLOW_LOCATION", "europe-west2")
        self._agent_id = agent_id or os.getenv("DIALOGFLOW_AGENT_ID", "")
        self._environment = environment or os.getenv("DIALOGFLOW_ENVIRONMENT", "draft")
        self._client = None

    @property
    def _session_path_prefix(self) -> str:
        return (
            f"projects/{self._project_id}/locations/{self._location}"
            f"/agents/{self._agent_id}/environments/{self._environment}"
        )

    def _get_client(self):
        """Lazy-initialize the Dialogflow CX SessionsClient."""
        if self._client is None:
            try:
                from google.cloud.dialogflowcx_v3 import SessionsClient

                self._client = SessionsClient()
                logger.info("Dialogflow CX client initialized")
            except ImportError:
                logger.warning(
                    "google-cloud-dialogflow-cx not installed — "
                    "Dialogflow dispatcher will operate in stub mode"
                )
            except Exception as exc:
                logger.error("Failed to initialize Dialogflow client: %s", exc)
        return self._client

    async def send(self, response: AgentResponse) -> DeliveryResult:
        """Deliver a response via Dialogflow → WhatsApp/SMS."""
        recipient_phone = response.metadata.get("phone", "")
        is_proactive = response.metadata.get("proactive", False)

        if is_proactive:
            return await self._send_template(response, recipient_phone)
        return await self._send_session_message(response, recipient_phone)

    async def _send_template(
        self, response: AgentResponse, phone: str
    ) -> DeliveryResult:
        """Send a proactive WhatsApp template message."""
        template_key = response.metadata.get("template_key", "default")
        template_id = WHATSAPP_TEMPLATES.get(template_key, WHATSAPP_TEMPLATES["default"])
        template_params = response.metadata.get("template_params", {})

        client = self._get_client()
        if client is None:
            # Stub mode — log and succeed for testing
            logger.info(
                "Dialogflow stub: template '%s' → %s: %s",
                template_id, phone, response.message[:80],
            )
            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
                error="stub_mode",
            )

        try:
            # Build session path using phone as session ID
            session_id = phone.replace("+", "").replace(" ", "")
            session_path = f"{self._session_path_prefix}/sessions/{session_id}"

            from google.cloud.dialogflowcx_v3 import (
                DetectIntentRequest,
                QueryInput,
                TextInput,
            )

            # Send the template trigger through Dialogflow
            text_input = TextInput(text=f"__template:{template_id}")
            query_input = QueryInput(text=text_input, language_code="en")

            request = DetectIntentRequest(
                session=session_path,
                query_input=query_input,
                query_params={
                    "parameters": {
                        "template_id": template_id,
                        "template_params": template_params,
                        "message_text": response.message,
                        "recipient_phone": phone,
                    }
                },
            )

            df_response = client.detect_intent(request=request)
            logger.info(
                "Dialogflow template sent: %s → %s",
                template_id, phone,
            )

            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
            )

        except Exception as exc:
            logger.error("Dialogflow template send failed: %s", exc)
            return DeliveryResult(
                success=False,
                channel=self.channel_name,
                recipient=response.recipient,
                error=str(exc),
            )

    async def _send_session_message(
        self, response: AgentResponse, phone: str
    ) -> DeliveryResult:
        """Send a reactive message within an existing Dialogflow session."""
        client = self._get_client()
        if client is None:
            logger.info(
                "Dialogflow stub: session msg → %s: %s",
                phone or response.recipient,
                response.message[:80],
            )
            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
                error="stub_mode",
            )

        try:
            session_id = phone.replace("+", "").replace(" ", "")
            session_path = f"{self._session_path_prefix}/sessions/{session_id}"

            from google.cloud.dialogflowcx_v3 import (
                DetectIntentRequest,
                QueryInput,
                TextInput,
            )

            text_input = TextInput(text=response.message)
            query_input = QueryInput(text=text_input, language_code="en")

            request = DetectIntentRequest(
                session=session_path,
                query_input=query_input,
            )

            df_response = client.detect_intent(request=request)
            logger.info("Dialogflow session message sent → %s", phone)

            return DeliveryResult(
                success=True,
                channel=self.channel_name,
                recipient=response.recipient,
            )

        except Exception as exc:
            logger.error("Dialogflow session send failed: %s", exc)
            return DeliveryResult(
                success=False,
                channel=self.channel_name,
                recipient=response.recipient,
                error=str(exc),
            )

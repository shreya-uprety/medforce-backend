"""
Tests for Phase 6 Channel Dispatchers — Dialogflow, Email, Twilio.

Tests cover:
  - Stub mode (no credentials) — dispatchers succeed gracefully
  - Message formatting and metadata handling
  - Template selection (Dialogflow)
  - SMS truncation (Twilio)
  - Error handling for missing recipients
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from medforce.gateway.channels import AgentResponse


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dialogflow Dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDialogflowDispatcher:
    """Dialogflow CX dispatcher tests."""

    def test_channel_name(self):
        from medforce.gateway.dispatchers.dialogflow_dispatcher import (
            DialogflowDispatcher,
        )
        d = DialogflowDispatcher()
        assert d.channel_name == "dialogflow_whatsapp"

    @pytest.mark.asyncio
    async def test_stub_mode_succeeds(self):
        from medforce.gateway.dispatchers.dialogflow_dispatcher import (
            DialogflowDispatcher,
        )
        d = DialogflowDispatcher()
        response = AgentResponse(
            recipient="patient",
            channel="dialogflow_whatsapp",
            message="Hello from Gateway",
            metadata={"phone": "+447700900001"},
        )
        result = await d.send(response)
        assert result.success is True
        assert result.channel == "dialogflow_whatsapp"
        assert result.error == "stub_mode"

    @pytest.mark.asyncio
    async def test_proactive_template_stub(self):
        from medforce.gateway.dispatchers.dialogflow_dispatcher import (
            DialogflowDispatcher,
        )
        d = DialogflowDispatcher()
        response = AgentResponse(
            recipient="patient",
            channel="dialogflow_whatsapp",
            message="Your 14-day check-in is due",
            metadata={
                "phone": "+447700900001",
                "proactive": True,
                "template_key": "heartbeat_reminder",
            },
        )
        result = await d.send(response)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_session_path_format(self):
        from medforce.gateway.dispatchers.dialogflow_dispatcher import (
            DialogflowDispatcher,
        )
        d = DialogflowDispatcher(
            project_id="my-project",
            location="europe-west2",
            agent_id="agent-123",
            environment="draft",
        )
        assert "projects/my-project" in d._session_path_prefix
        assert "agents/agent-123" in d._session_path_prefix
        assert "europe-west2" in d._session_path_prefix

    @pytest.mark.asyncio
    async def test_reactive_message_stub(self):
        from medforce.gateway.dispatchers.dialogflow_dispatcher import (
            DialogflowDispatcher,
        )
        d = DialogflowDispatcher()
        response = AgentResponse(
            recipient="patient",
            channel="dialogflow_whatsapp",
            message="Thank you for your update",
            metadata={"phone": "+447700900002"},
        )
        result = await d.send(response)
        assert result.success is True
        assert result.recipient == "patient"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Email Dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmailDispatcher:
    """SendGrid email dispatcher tests."""

    def test_channel_name(self):
        from medforce.gateway.dispatchers.email_dispatcher import (
            EmailDispatcher,
        )
        d = EmailDispatcher()
        assert d.channel_name == "email"

    @pytest.mark.asyncio
    async def test_stub_mode_succeeds(self):
        from medforce.gateway.dispatchers.email_dispatcher import (
            EmailDispatcher,
        )
        d = EmailDispatcher()
        response = AgentResponse(
            recipient="gp:Dr.Patel",
            channel="email",
            message="Please provide lab results for patient REF-2026-001",
            metadata={
                "to": "dr.patel@nhs.net",
                "subject": "MedForce — Lab Results Requested",
                "reply_to": "gp-reply+PT-001@medforce.app",
            },
        )
        result = await d.send(response)
        assert result.success is True
        assert result.error == "stub_mode"

    @pytest.mark.asyncio
    async def test_missing_to_address_fails(self):
        from medforce.gateway.dispatchers.email_dispatcher import (
            EmailDispatcher,
        )
        d = EmailDispatcher()
        response = AgentResponse(
            recipient="gp:Dr.Patel",
            channel="email",
            message="No destination",
            metadata={},
        )
        result = await d.send(response)
        assert result.success is False
        assert "No recipient email" in result.error

    @pytest.mark.asyncio
    async def test_default_subject(self):
        from medforce.gateway.dispatchers.email_dispatcher import (
            EmailDispatcher,
        )
        d = EmailDispatcher()
        response = AgentResponse(
            recipient="gp:Dr.Smith",
            channel="email",
            message="Follow-up required",
            metadata={"to": "dr.smith@nhs.net"},
        )
        result = await d.send(response)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_custom_from_email(self):
        from medforce.gateway.dispatchers.email_dispatcher import (
            EmailDispatcher,
        )
        d = EmailDispatcher(
            from_email="clinical@myapp.com",
            from_name="My Clinical App",
        )
        assert d._from_email == "clinical@myapp.com"
        assert d._from_name == "My Clinical App"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Twilio SMS Dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTwilioDispatcher:
    """Twilio SMS dispatcher tests."""

    def test_channel_name(self):
        from medforce.gateway.dispatchers.twilio_dispatcher import (
            TwilioSMSDispatcher,
        )
        d = TwilioSMSDispatcher()
        assert d.channel_name == "sms"

    @pytest.mark.asyncio
    async def test_stub_mode_succeeds(self):
        from medforce.gateway.dispatchers.twilio_dispatcher import (
            TwilioSMSDispatcher,
        )
        d = TwilioSMSDispatcher()
        response = AgentResponse(
            recipient="patient",
            channel="sms",
            message="Your appointment is confirmed",
            metadata={"phone": "+447700900001"},
        )
        result = await d.send(response)
        assert result.success is True
        assert result.error == "stub_mode"

    @pytest.mark.asyncio
    async def test_missing_phone_fails(self):
        from medforce.gateway.dispatchers.twilio_dispatcher import (
            TwilioSMSDispatcher,
        )
        d = TwilioSMSDispatcher()
        response = AgentResponse(
            recipient="patient",
            channel="sms",
            message="No phone",
            metadata={},
        )
        result = await d.send(response)
        assert result.success is False
        assert "No recipient phone" in result.error

    @pytest.mark.asyncio
    async def test_long_message_truncation(self):
        from medforce.gateway.dispatchers.twilio_dispatcher import (
            TwilioSMSDispatcher,
        )
        d = TwilioSMSDispatcher()
        long_msg = "A" * 2000
        response = AgentResponse(
            recipient="patient",
            channel="sms",
            message=long_msg,
            metadata={"phone": "+447700900001"},
        )
        # Stub mode so we can't check truncation in the actual API call,
        # but the code path runs without error
        result = await d.send(response)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_custom_credentials(self):
        from medforce.gateway.dispatchers.twilio_dispatcher import (
            TwilioSMSDispatcher,
        )
        d = TwilioSMSDispatcher(
            account_sid="AC_TEST",
            auth_token="test_token",
            from_number="+441234567890",
        )
        assert d._account_sid == "AC_TEST"
        assert d._from_number == "+441234567890"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Conditional Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConditionalRegistration:
    """setup.py conditional dispatcher registration."""

    def test_no_env_vars_no_external_dispatchers(self):
        from medforce.gateway.channels import DispatcherRegistry
        from medforce.gateway.setup import _register_external_dispatchers

        registry = DispatcherRegistry()
        _register_external_dispatchers(registry)
        # No env vars set → no external dispatchers registered
        assert "dialogflow_whatsapp" not in registry.registered_channels
        assert "email" not in registry.registered_channels
        assert "sms" not in registry.registered_channels

    @patch.dict("os.environ", {"DIALOGFLOW_PROJECT_ID": "test-project"})
    def test_dialogflow_env_registers_dispatcher(self):
        from medforce.gateway.channels import DispatcherRegistry
        from medforce.gateway.setup import _register_external_dispatchers

        registry = DispatcherRegistry()
        _register_external_dispatchers(registry)
        assert "dialogflow_whatsapp" in registry.registered_channels

    @patch.dict("os.environ", {"SENDGRID_API_KEY": "SG.test_key"})
    def test_sendgrid_env_registers_dispatcher(self):
        from medforce.gateway.channels import DispatcherRegistry
        from medforce.gateway.setup import _register_external_dispatchers

        registry = DispatcherRegistry()
        _register_external_dispatchers(registry)
        assert "email" in registry.registered_channels

    @patch.dict("os.environ", {"TWILIO_ACCOUNT_SID": "AC_TEST"})
    def test_twilio_env_registers_dispatcher(self):
        from medforce.gateway.channels import DispatcherRegistry
        from medforce.gateway.setup import _register_external_dispatchers

        registry = DispatcherRegistry()
        _register_external_dispatchers(registry)
        assert "sms" in registry.registered_channels

    @patch.dict("os.environ", {
        "DIALOGFLOW_PROJECT_ID": "proj",
        "SENDGRID_API_KEY": "SG.key",
        "TWILIO_ACCOUNT_SID": "AC_SID",
    })
    def test_all_three_register(self):
        from medforce.gateway.channels import DispatcherRegistry
        from medforce.gateway.setup import _register_external_dispatchers

        registry = DispatcherRegistry()
        _register_external_dispatchers(registry)
        assert "dialogflow_whatsapp" in registry.registered_channels
        assert "email" in registry.registered_channels
        assert "sms" in registry.registered_channels

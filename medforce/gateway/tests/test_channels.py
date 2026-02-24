"""
Comprehensive tests for Channel Dispatcher and Ingest abstractions.

Tests cover:
  - DispatcherRegistry registration, lookup, unregistration
  - Dispatch routing to correct channel
  - Fallback when no dispatcher registered
  - Error handling in dispatchers
  - Bulk dispatch
  - AgentResponse model
  - DeliveryResult model
  - Concrete dispatchers: WebSocketDispatcher, TestHarnessDispatcher
  - Patient scenarios with multi-channel responses
"""

import pytest

from medforce.gateway.channels import (
    AgentResponse,
    ChannelDispatcher,
    ChannelIngest,
    DeliveryResult,
    DispatcherRegistry,
)
from medforce.gateway.dispatchers.test_harness_dispatcher import TestHarnessDispatcher
from medforce.gateway.dispatchers.websocket_dispatcher import WebSocketDispatcher
from medforce.gateway.events import EventEnvelope, EventType, SenderRole


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentResponse Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentResponse:

    def test_basic_response(self):
        r = AgentResponse(
            recipient="patient",
            channel="websocket",
            message="Hello John",
        )
        assert r.recipient == "patient"
        assert r.channel == "websocket"
        assert r.message == "Hello John"
        assert r.attachments == []
        assert r.metadata == {}

    def test_response_with_metadata(self):
        r = AgentResponse(
            recipient="gp:Dr.Patel",
            channel="email",
            message="Lab results requested",
            metadata={
                "subject": "MedForce — Lab Results Request",
                "to": "dr.patel@nhs.uk",
            },
        )
        assert r.metadata["subject"] == "MedForce — Lab Results Request"

    def test_response_with_attachments(self):
        r = AgentResponse(
            recipient="helper:HELPER-001",
            channel="whatsapp",
            message="Booking confirmed",
            attachments=["confirmation.pdf"],
        )
        assert len(r.attachments) == 1

    def test_json_round_trip(self):
        original = AgentResponse(
            recipient="patient",
            channel="websocket",
            message="Test round trip",
            metadata={"key": "value"},
        )
        restored = AgentResponse.model_validate_json(original.model_dump_json())
        assert restored.message == original.message
        assert restored.metadata == original.metadata


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DeliveryResult Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeliveryResult:

    def test_success_result(self):
        r = DeliveryResult(success=True, channel="websocket", recipient="patient")
        assert r.success
        assert r.error is None

    def test_failure_result(self):
        r = DeliveryResult(
            success=False,
            channel="email",
            recipient="gp:Dr.Patel",
            error="SMTP connection failed",
        )
        assert not r.success
        assert "SMTP" in r.error

    def test_timestamp_auto_generated(self):
        r = DeliveryResult(success=True, channel="test", recipient="patient")
        assert r.timestamp is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DispatcherRegistry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDispatcherRegistry:

    def test_register_and_lookup(self):
        reg = DispatcherRegistry()
        ws = WebSocketDispatcher()
        reg.register(ws)
        assert reg.get("websocket") is ws
        assert "websocket" in reg.registered_channels

    def test_lookup_missing_returns_none(self):
        reg = DispatcherRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = DispatcherRegistry()
        ws = WebSocketDispatcher()
        reg.register(ws)
        reg.unregister("websocket")
        assert reg.get("websocket") is None
        assert "websocket" not in reg.registered_channels

    def test_multiple_dispatchers(self):
        reg = DispatcherRegistry()
        ws = WebSocketDispatcher()
        th = TestHarnessDispatcher()
        reg.register(ws)
        reg.register(th)
        assert len(reg.registered_channels) == 2
        assert reg.get("websocket") is ws
        assert reg.get("test_harness") is th

    @pytest.mark.asyncio
    async def test_dispatch_to_correct_channel(self):
        reg = DispatcherRegistry()
        ws = WebSocketDispatcher()
        reg.register(ws)

        response = AgentResponse(
            recipient="patient",
            channel="websocket",
            message="Hello",
        )
        result = await reg.dispatch(response)
        assert result.success
        assert result.channel == "websocket"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_channel_returns_failure(self):
        reg = DispatcherRegistry()
        # No dispatchers registered

        response = AgentResponse(
            recipient="patient",
            channel="whatsapp",
            message="Hello",
        )
        result = await reg.dispatch(response)
        assert not result.success
        assert "No dispatcher" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_all(self):
        reg = DispatcherRegistry()
        reg.register(WebSocketDispatcher())
        reg.register(TestHarnessDispatcher())

        responses = [
            AgentResponse(recipient="patient", channel="websocket", message="Hello patient"),
            AgentResponse(
                recipient="helper:HELPER-001",
                channel="test_harness",
                message="Hello helper",
                metadata={"patient_id": "PT-1"},
            ),
        ]
        results = await reg.dispatch_all(responses)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dispatch_all_partial_failure(self):
        """Some channels available, some not."""
        reg = DispatcherRegistry()
        reg.register(WebSocketDispatcher())
        # No email dispatcher

        responses = [
            AgentResponse(recipient="patient", channel="websocket", message="OK"),
            AgentResponse(recipient="gp:Dr.Patel", channel="email", message="Query"),
        ]
        results = await reg.dispatch_all(responses)
        assert results[0].success  # websocket worked
        assert not results[1].success  # email failed (no dispatcher)

    @pytest.mark.asyncio
    async def test_dispatch_handles_exception_in_dispatcher(self):
        """Dispatcher that throws should not crash the registry."""

        class BrokenDispatcher(ChannelDispatcher):
            channel_name = "broken"
            async def send(self, response):
                raise RuntimeError("Connection lost")

        reg = DispatcherRegistry()
        reg.register(BrokenDispatcher())

        response = AgentResponse(recipient="patient", channel="broken", message="Test")
        result = await reg.dispatch(response)
        assert not result.success
        assert "Connection lost" in result.error


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WebSocketDispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWebSocketDispatcher:

    @pytest.mark.asyncio
    async def test_send_returns_success(self):
        ws = WebSocketDispatcher()
        result = await ws.send(
            AgentResponse(recipient="patient", channel="websocket", message="Hello")
        )
        assert result.success
        assert result.channel == "websocket"

    def test_channel_name(self):
        assert WebSocketDispatcher.channel_name == "websocket"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TestHarnessDispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTestHarnessDispatcher:

    @pytest.mark.asyncio
    async def test_stores_responses(self):
        th = TestHarnessDispatcher()
        r = AgentResponse(
            recipient="patient",
            channel="test_harness",
            message="Hello from test",
            metadata={"patient_id": "PT-100"},
        )
        result = await th.send(r)
        assert result.success

        stored = th.get_responses("PT-100")
        assert len(stored) == 1
        assert stored[0].message == "Hello from test"

    @pytest.mark.asyncio
    async def test_multiple_responses_for_patient(self):
        th = TestHarnessDispatcher()
        for i in range(3):
            await th.send(AgentResponse(
                recipient="patient",
                channel="test_harness",
                message=f"Message {i}",
                metadata={"patient_id": "PT-200"},
            ))
        assert len(th.get_responses("PT-200")) == 3

    @pytest.mark.asyncio
    async def test_clear_specific_patient(self):
        th = TestHarnessDispatcher()
        await th.send(AgentResponse(
            recipient="patient", channel="test_harness",
            message="A", metadata={"patient_id": "PT-A"},
        ))
        await th.send(AgentResponse(
            recipient="patient", channel="test_harness",
            message="B", metadata={"patient_id": "PT-B"},
        ))
        th.clear("PT-A")
        assert len(th.get_responses("PT-A")) == 0
        assert len(th.get_responses("PT-B")) == 1

    @pytest.mark.asyncio
    async def test_clear_all(self):
        th = TestHarnessDispatcher()
        await th.send(AgentResponse(
            recipient="patient", channel="test_harness",
            message="A", metadata={"patient_id": "PT-A"},
        ))
        th.clear()
        assert len(th.get_responses("PT-A")) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Multi-Channel Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiChannelScenarios:

    @pytest.mark.asyncio
    async def test_scenario_patient_and_helper_different_channels(self):
        """Patient on WebSocket, helper on test_harness (simulating WhatsApp)."""
        reg = DispatcherRegistry()
        reg.register(WebSocketDispatcher())
        reg.register(TestHarnessDispatcher())

        responses = [
            AgentResponse(
                recipient="patient",
                channel="websocket",
                message="Your results are in, John.",
            ),
            AgentResponse(
                recipient="helper:Sarah",
                channel="test_harness",
                message="Sarah, John's results have been processed.",
                metadata={"patient_id": "PT-JOHN"},
            ),
        ]
        results = await reg.dispatch_all(responses)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_scenario_gp_email_not_available_yet(self):
        """GP response needs email channel which isn't registered (Phase 6)."""
        reg = DispatcherRegistry()
        reg.register(WebSocketDispatcher())

        responses = [
            AgentResponse(
                recipient="patient",
                channel="websocket",
                message="We've contacted your GP.",
            ),
            AgentResponse(
                recipient="gp:Dr.Patel",
                channel="email",
                message="Lab results requested...",
                metadata={"subject": "MedForce Query", "to": "dr.patel@nhs.uk"},
            ),
        ]
        results = await reg.dispatch_all(responses)
        assert results[0].success  # patient via websocket
        assert not results[1].success  # GP email not available
        assert "No dispatcher" in results[1].error

    @pytest.mark.asyncio
    async def test_scenario_deterioration_multi_notify(self):
        """Deterioration alert goes to patient, helper, and GP (3 channels)."""
        reg = DispatcherRegistry()
        reg.register(WebSocketDispatcher())
        th = TestHarnessDispatcher()
        reg.register(th)

        responses = [
            AgentResponse(
                recipient="patient",
                channel="websocket",
                message="We've detected a concern. Reassessment initiated.",
            ),
            AgentResponse(
                recipient="helper:Peter",
                channel="test_harness",
                message="Alert: Helen's condition needs reassessment.",
                metadata={"patient_id": "PT-HELEN"},
            ),
            AgentResponse(
                recipient="gp:Dr.Brown",
                channel="email",
                message="Deterioration alert for Helen Morris.",
            ),
        ]
        results = await reg.dispatch_all(responses)
        assert results[0].success  # patient OK
        assert results[1].success  # helper OK
        assert not results[2].success  # GP email not available

        # Helper got stored in test harness
        stored = th.get_responses("PT-HELEN")
        assert len(stored) == 1

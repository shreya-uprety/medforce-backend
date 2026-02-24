"""
WebSocket Dispatcher — delivers responses via connected WebSocket sessions.

This is the primary dispatcher during Phases 1-5, using the existing
WebSocket infrastructure in medforce.agents.websocket_agent.
"""

from __future__ import annotations

import logging

from medforce.gateway.channels import (
    AgentResponse,
    ChannelDispatcher,
    DeliveryResult,
)

logger = logging.getLogger("gateway.dispatchers.websocket")


class WebSocketDispatcher(ChannelDispatcher):
    """Push messages to connected WebSocket sessions."""

    channel_name = "websocket"

    def __init__(self) -> None:
        # Will hold reference to the WebSocket session registry once
        # the Gateway is wired into the existing WebSocket router (Phase 2).
        self._session_registry: dict | None = None

    async def send(self, response: AgentResponse) -> DeliveryResult:
        # Phase 1: stub — logs the response for now.
        # Phase 2 will wire this to the real WebSocket session push.
        logger.info(
            "WebSocket dispatch → %s: %s",
            response.recipient,
            response.message[:80] if response.message else "(empty)",
        )
        return DeliveryResult(
            success=True,
            channel=self.channel_name,
            recipient=response.recipient,
        )

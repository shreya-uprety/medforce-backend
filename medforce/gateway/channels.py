"""
Channel Abstractions — Inbound ingestion and outbound dispatch.

These ABCs decouple the Gateway and agents from any specific messaging
channel.  Adding Dialogflow, WhatsApp, SMS, or email in Phase 6 is just:
  1. Implement a ChannelDispatcher subclass  (~50 lines)
  2. Implement a ChannelIngest subclass      (~30 lines)
  3. Add a webhook endpoint                  (~15 lines)
  4. Register the dispatcher in setup.py     (1 line)
Zero changes to Gateway, agents, diary, or queue.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from medforce.gateway.events import (
    EventEnvelope,
    EventType,
    SenderRole,
)

logger = logging.getLogger("gateway.channels")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OUTBOUND — delivering responses to recipients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AgentResponse(BaseModel):
    """A message an agent wants to send to a specific recipient."""

    recipient: str          # "patient", "helper:HELPER-001", "gp:Dr.Patel"
    channel: str            # Must match a registered ChannelDispatcher.channel_name
    message: str = ""
    attachments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # e.g. {"subject": "...", "template_id": "...", "proactive": True}


class DeliveryResult(BaseModel):
    """Outcome of a single message delivery attempt."""

    success: bool
    channel: str
    recipient: str
    error: Optional[str] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ChannelDispatcher(ABC):
    """Abstract outbound channel — delivers AgentResponses to a channel."""

    channel_name: str = ""  # overridden by subclasses

    @abstractmethod
    async def send(self, response: AgentResponse) -> DeliveryResult:
        """Deliver a single response. Must not raise — return DeliveryResult."""

    async def send_bulk(self, responses: list[AgentResponse]) -> list[DeliveryResult]:
        """Deliver multiple responses.  Default: sequential send()."""
        results = []
        for r in responses:
            results.append(await self.send(r))
        return results


class DispatcherRegistry:
    """
    Registry of active ChannelDispatchers.

    The Gateway calls ``dispatch()`` or ``dispatch_all()`` — it never
    talks to a specific channel directly.
    """

    def __init__(self) -> None:
        self._dispatchers: dict[str, ChannelDispatcher] = {}

    def register(self, dispatcher: ChannelDispatcher) -> None:
        name = dispatcher.channel_name
        self._dispatchers[name] = dispatcher
        logger.info("Registered channel dispatcher: %s", name)

    def unregister(self, channel_name: str) -> None:
        self._dispatchers.pop(channel_name, None)

    def get(self, channel_name: str) -> ChannelDispatcher | None:
        return self._dispatchers.get(channel_name)

    @property
    def registered_channels(self) -> list[str]:
        return list(self._dispatchers.keys())

    async def dispatch(self, response: AgentResponse) -> DeliveryResult:
        """Route one response to the correct dispatcher (with single retry)."""
        dispatcher = self.get(response.channel)
        if dispatcher is None:
            logger.warning(
                "No dispatcher for channel '%s' — response stored only",
                response.channel,
            )
            return DeliveryResult(
                success=False,
                channel=response.channel,
                recipient=response.recipient,
                error=f"No dispatcher registered for channel '{response.channel}'",
            )
        for attempt in range(2):
            try:
                result = await dispatcher.send(response)
                if result.success or attempt == 1:
                    return result
                logger.warning(
                    "Dispatch failed for %s on %s (attempt 1) — retrying",
                    response.recipient, response.channel,
                )
                await asyncio.sleep(0.5)
            except Exception as exc:
                if attempt == 0:
                    logger.warning(
                        "Dispatcher '%s' error (attempt 1): %s — retrying",
                        response.channel, exc,
                    )
                    await asyncio.sleep(0.5)
                else:
                    logger.error(
                        "Dispatcher '%s' error after retry: %s",
                        response.channel, exc,
                    )
                    return DeliveryResult(
                        success=False,
                        channel=response.channel,
                        recipient=response.recipient,
                        error=str(exc),
                    )
        # Should not reach here, but handle gracefully
        return DeliveryResult(
            success=False,
            channel=response.channel,
            recipient=response.recipient,
            error="Dispatch failed after retry",
        )

    async def dispatch_all(
        self, responses: list[AgentResponse]
    ) -> list[DeliveryResult]:
        """Dispatch every response in a result set."""
        results: list[DeliveryResult] = []
        for response in responses:
            results.append(await self.dispatch(response))
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INBOUND — converting channel-specific input into EventEnvelopes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChannelIngest(ABC):
    """Abstract inbound channel — converts raw input into an EventEnvelope."""

    channel_name: str = ""

    @abstractmethod
    async def to_envelope(self, raw_input: dict[str, Any]) -> EventEnvelope:
        """Parse channel-specific data into a standard EventEnvelope."""

    def _build_base_envelope(
        self,
        *,
        event_type: EventType = EventType.USER_MESSAGE,
        patient_id: str,
        sender_id: str,
        sender_role: SenderRole,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> EventEnvelope:
        """Shared helper for subclasses."""
        return EventEnvelope(
            event_id=str(uuid4()),
            event_type=event_type,
            patient_id=patient_id,
            payload={**payload, "channel": self.channel_name},
            source=self.channel_name,
            sender_id=sender_id,
            sender_role=sender_role,
            correlation_id=correlation_id,
            timestamp=datetime.now(timezone.utc),
        )

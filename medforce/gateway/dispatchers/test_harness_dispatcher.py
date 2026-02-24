"""
Test Harness Dispatcher — stores responses in memory for the HTML test
harness to poll via GET /api/gateway/events/{patient_id}.

Used during development and testing (Phase 5).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from medforce.gateway.channels import (
    AgentResponse,
    ChannelDispatcher,
    DeliveryResult,
)

logger = logging.getLogger("gateway.dispatchers.test_harness")


class TestHarnessDispatcher(ChannelDispatcher):
    """Stores responses in memory for test harness polling."""

    channel_name = "test_harness"

    def __init__(self) -> None:
        # patient_id → list of AgentResponses
        self._response_log: dict[str, list[AgentResponse]] = defaultdict(list)

    async def send(self, response: AgentResponse) -> DeliveryResult:
        # Extract patient_id from recipient if possible
        patient_id = response.metadata.get("patient_id", "unknown")
        self._response_log[patient_id].append(response)
        logger.debug(
            "Test harness stored response for %s → %s",
            patient_id,
            response.recipient,
        )
        return DeliveryResult(
            success=True,
            channel=self.channel_name,
            recipient=response.recipient,
        )

    def get_responses(self, patient_id: str) -> list[AgentResponse]:
        """Retrieve all stored responses for a patient (test harness polls this)."""
        return list(self._response_log.get(patient_id, []))

    def clear(self, patient_id: str | None = None) -> None:
        """Clear stored responses. If patient_id is None, clear everything."""
        if patient_id:
            self._response_log.pop(patient_id, None)
        else:
            self._response_log.clear()

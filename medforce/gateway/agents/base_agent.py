"""
Base Agent Contract — the universal interface every specialist agent implements.

The Gateway calls ``agent.process(event, diary)`` and receives an ``AgentResult``
containing the updated diary, any new events to emit, and responses to send.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import PatientDiary
from medforce.gateway.events import EventEnvelope


@dataclass
class AgentResult:
    """
    Everything an agent returns after processing an event.

    - updated_diary: the diary with all mutations applied
    - emitted_events: new events for the Gateway to loop back (handoffs, alerts, queries)
    - responses: messages to deliver via DispatcherRegistry
    """

    updated_diary: PatientDiary
    emitted_events: list[EventEnvelope] = field(default_factory=list)
    responses: list[AgentResponse] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Every agent must implement ``process()``.  The Gateway guarantees:
      - The diary is loaded and current before calling process()
      - The diary will be saved after process() returns
      - Emitted events will be looped back through the Gateway
      - Responses will be dispatched via the DispatcherRegistry
    """

    agent_name: str = ""  # overridden by subclasses

    @abstractmethod
    async def process(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """
        Process a single event against the patient diary.

        Args:
            event: The incoming event to handle
            diary: Current patient diary (loaded from GCS)

        Returns:
            AgentResult with updated diary, emitted events, and responses
        """

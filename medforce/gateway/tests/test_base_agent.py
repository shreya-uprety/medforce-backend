"""
Tests for the Base Agent contract and AgentResult.
"""

import pytest

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import PatientDiary
from medforce.gateway.events import EventEnvelope, EventType, SenderRole


class DummyAgent(BaseAgent):
    """Concrete implementation for testing the ABC."""

    agent_name = "dummy"

    async def process(self, event, diary):
        return AgentResult(
            updated_diary=diary,
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel="websocket",
                    message="Hello from dummy agent",
                )
            ],
        )


class FailingAgent(BaseAgent):
    """Agent that raises an error."""

    agent_name = "failing"

    async def process(self, event, diary):
        raise ValueError("Simulated agent error")


class HandoffAgent(BaseAgent):
    """Agent that emits a handoff event."""

    agent_name = "handoff"

    async def process(self, event, diary):
        handoff = EventEnvelope.handoff(
            event_type=EventType.INTAKE_COMPLETE,
            patient_id=event.patient_id,
            source_agent="handoff",
        )
        return AgentResult(
            updated_diary=diary,
            emitted_events=[handoff],
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel="websocket",
                    message="Handing off to clinical",
                )
            ],
        )


@pytest.fixture
def diary():
    return PatientDiary.create_new("PT-TEST")


@pytest.fixture
def event():
    return EventEnvelope.user_message("PT-TEST", "Hello")


# ── Tests ──


class TestAgentResult:
    def test_agent_result_defaults(self):
        diary = PatientDiary.create_new("PT-001")
        result = AgentResult(updated_diary=diary)
        assert result.updated_diary is diary
        assert result.emitted_events == []
        assert result.responses == []

    def test_agent_result_with_responses(self):
        diary = PatientDiary.create_new("PT-001")
        resp = AgentResponse(
            recipient="patient",
            channel="websocket",
            message="Hello!",
        )
        result = AgentResult(updated_diary=diary, responses=[resp])
        assert len(result.responses) == 1
        assert result.responses[0].message == "Hello!"

    def test_agent_result_with_emitted_events(self):
        diary = PatientDiary.create_new("PT-001")
        handoff = EventEnvelope.handoff(
            EventType.INTAKE_COMPLETE,
            "PT-001",
            source_agent="test",
        )
        result = AgentResult(updated_diary=diary, emitted_events=[handoff])
        assert len(result.emitted_events) == 1
        assert result.emitted_events[0].event_type == EventType.INTAKE_COMPLETE


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_dummy_agent_process(self, event, diary):
        agent = DummyAgent()
        result = await agent.process(event, diary)
        assert result.updated_diary is diary
        assert len(result.responses) == 1
        assert "Hello from dummy" in result.responses[0].message

    @pytest.mark.asyncio
    async def test_failing_agent_raises(self, event, diary):
        agent = FailingAgent()
        with pytest.raises(ValueError, match="Simulated"):
            await agent.process(event, diary)

    @pytest.mark.asyncio
    async def test_handoff_agent_emits_event(self, event, diary):
        agent = HandoffAgent()
        result = await agent.process(event, diary)
        assert len(result.emitted_events) == 1
        assert result.emitted_events[0].event_type == EventType.INTAKE_COMPLETE
        assert len(result.responses) == 1

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseAgent()

    def test_agent_name_attribute(self):
        agent = DummyAgent()
        assert agent.agent_name == "dummy"

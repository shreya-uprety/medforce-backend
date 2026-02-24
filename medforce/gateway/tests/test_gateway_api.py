"""
Tests for the Gateway API endpoints.

Covers:
  - POST /api/gateway/emit — event submission
  - GET /api/gateway/diary/{id} — diary retrieval
  - GET /api/gateway/events/{id} — event log
  - GET /api/gateway/status — health check
  - Error handling and validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import AgentResponse, DispatcherRegistry, DeliveryResult
from medforce.gateway.diary import DiaryNotFoundError, DiaryStore, PatientDiary, Phase
from medforce.gateway.events import EventType, SenderRole
from medforce.gateway.gateway import Gateway
from medforce.gateway.permissions import PermissionChecker


# ── Helpers ──


class StubAgent(BaseAgent):
    agent_name = "stub"

    async def process(self, event, diary):
        return AgentResult(
            updated_diary=diary,
            responses=[
                AgentResponse(
                    recipient="patient",
                    channel="websocket",
                    message="Stub response",
                    metadata={"patient_id": event.patient_id},
                )
            ],
        )


class MockDiaryStore:
    def __init__(self):
        self._diaries = {}

    def load(self, patient_id):
        if patient_id not in self._diaries:
            raise DiaryNotFoundError(f"Not found: {patient_id}")
        diary, gen = self._diaries[patient_id]
        return diary, gen

    def save(self, patient_id, diary, generation=None):
        new_gen = (generation or 0) + 1
        self._diaries[patient_id] = (diary, new_gen)
        return new_gen

    def create(self, patient_id, correlation_id=None):
        diary = PatientDiary.create_new(patient_id, correlation_id=correlation_id)
        gen = self.save(patient_id, diary)
        return diary, gen

    def seed(self, patient_id, diary, generation=1):
        self._diaries[patient_id] = (diary, generation)


def _create_test_gateway():
    """Create a fully wired test Gateway."""
    diary_store = MockDiaryStore()
    registry = DispatcherRegistry()

    mock_ws = MagicMock()
    mock_ws.channel_name = "websocket"
    mock_ws.send = AsyncMock(return_value=DeliveryResult(
        success=True, channel="websocket", recipient="patient"
    ))
    registry.register(mock_ws)

    gateway = Gateway(
        diary_store=diary_store,
        dispatcher_registry=registry,
    )
    gateway.register_agent("intake", StubAgent())
    gateway.register_agent("clinical", StubAgent())
    gateway.register_agent("booking", StubAgent())
    gateway.register_agent("monitoring", StubAgent())

    return gateway, diary_store, registry


# Module-level fixtures to mock the setup module
_test_gateway = None
_test_diary_store = None
_test_registry = None


@pytest.fixture(autouse=True)
def setup_test_gateway():
    """Patch the gateway.setup module to use test instances."""
    global _test_gateway, _test_diary_store, _test_registry
    _test_gateway, _test_diary_store, _test_registry = _create_test_gateway()

    # The mock queue manager calls process_event directly so tests
    # can emit and immediately inspect diary / event log.
    async def _mock_enqueue(env):
        await _test_gateway.process_event(env)

    mock_queue = MagicMock(
        active_count=0,
        active_patients=[],
        enqueue=AsyncMock(side_effect=_mock_enqueue),
    )

    with patch("medforce.gateway.setup._gateway", _test_gateway), \
         patch("medforce.gateway.setup._diary_store", _test_diary_store), \
         patch("medforce.gateway.setup._dispatcher_registry", _test_registry), \
         patch("medforce.gateway.setup._queue_manager", mock_queue):
        yield


@pytest.fixture
def client():
    """FastAPI test client with gateway router only."""
    from fastapi import FastAPI
    from medforce.routers.gateway_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── POST /api/gateway/emit ──


class TestEmitEndpoint:
    def test_emit_user_message(self, client):
        resp = client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
            "patient_id": "PT-001",
            "payload": {"text": "Hello", "channel": "websocket"},
            "sender_id": "PATIENT",
            "sender_role": "patient",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "PT-001" in data["message"]

    def test_emit_invalid_event_type(self, client):
        resp = client.post("/api/gateway/emit", json={
            "event_type": "INVALID_TYPE",
            "patient_id": "PT-001",
        })
        assert resp.status_code == 400
        assert "Invalid event_type" in resp.json()["detail"]

    def test_emit_invalid_sender_role(self, client):
        resp = client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
            "patient_id": "PT-001",
            "sender_role": "alien",
        })
        assert resp.status_code == 400
        assert "Invalid sender_role" in resp.json()["detail"]

    def test_emit_heartbeat(self, client):
        resp = client.post("/api/gateway/emit", json={
            "event_type": "HEARTBEAT",
            "patient_id": "PT-001",
            "sender_role": "system",
            "payload": {"days_since_appointment": 14},
        })
        assert resp.status_code == 200

    def test_emit_with_correlation_id(self, client):
        resp = client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
            "patient_id": "PT-001",
            "correlation_id": "CORR-123",
            "payload": {"text": "Hello"},
        })
        assert resp.status_code == 200

    def test_emit_missing_patient_id(self, client):
        resp = client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
        })
        assert resp.status_code == 422  # Pydantic validation error


# ── GET /api/gateway/diary/{id} ──


class TestDiaryEndpoint:
    def test_get_existing_diary(self, client):
        # First create a diary by emitting an event
        client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
            "patient_id": "PT-001",
            "payload": {"text": "Hello"},
        })

        resp = client.get("/api/gateway/diary/PT-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patient_id"] == "PT-001"
        assert "diary" in data
        assert data["diary"]["header"]["patient_id"] == "PT-001"

    def test_get_nonexistent_diary(self, client):
        resp = client.get("/api/gateway/diary/PT-NONEXISTENT")
        assert resp.status_code == 404

    def test_diary_contains_all_sections(self, client):
        client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
            "patient_id": "PT-002",
            "payload": {"text": "Hello"},
        })

        resp = client.get("/api/gateway/diary/PT-002")
        data = resp.json()["diary"]
        assert "header" in data
        assert "intake" in data
        assert "clinical" in data
        assert "booking" in data
        assert "monitoring" in data
        assert "helper_registry" in data
        assert "gp_channel" in data
        assert "conversation_log" in data


# ── GET /api/gateway/events/{id} ──


class TestEventsEndpoint:
    def test_get_events_after_emit(self, client):
        client.post("/api/gateway/emit", json={
            "event_type": "USER_MESSAGE",
            "patient_id": "PT-001",
            "payload": {"text": "Hello"},
        })

        resp = client.get("/api/gateway/events/PT-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patient_id"] == "PT-001"
        assert data["count"] > 0
        assert len(data["events"]) > 0

    def test_get_events_empty(self, client):
        resp = client.get("/api/gateway/events/PT-NOBODY")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_events_with_limit(self, client):
        for i in range(5):
            client.post("/api/gateway/emit", json={
                "event_type": "USER_MESSAGE",
                "patient_id": "PT-MULTI",
                "payload": {"text": f"Message {i}"},
            })

        resp = client.get("/api/gateway/events/PT-MULTI?limit=2")
        data = resp.json()
        assert len(data["events"]) == 2


# ── GET /api/gateway/status ──


class TestStatusEndpoint:
    def test_status_ok(self, client):
        resp = client.get("/api/gateway/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "registered_agents" in data
        assert "registered_channels" in data

    def test_status_shows_agents(self, client):
        resp = client.get("/api/gateway/status")
        data = resp.json()
        assert "intake" in data["registered_agents"]
        assert "clinical" in data["registered_agents"]

    def test_status_shows_channels(self, client):
        resp = client.get("/api/gateway/status")
        data = resp.json()
        assert "websocket" in data["registered_channels"]

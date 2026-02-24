"""
Tests for the GP Communication Handler — query generation, reminders,
7-day fallback, and email channel integration.
"""

import pytest
from datetime import datetime, timedelta, timezone

from medforce.gateway.handlers.gp_comms import GPCommunicationHandler
from medforce.gateway.diary import (
    GPChannel,
    GPQuery,
    IntakeSection,
    PatientDiary,
    Phase,
)
from medforce.gateway.events import EventEnvelope, EventType, SenderRole


# ── Fixtures ──


def make_diary(
    patient_id: str = "PT-300",
    gp_name: str = "Dr. Patel",
    gp_email: str = "dr.patel@nhs.net",
) -> PatientDiary:
    """Create a test diary with GP channel info."""
    diary = PatientDiary.create_new(patient_id)
    diary.header.current_phase = Phase.CLINICAL
    diary.intake.name = "Alice Brown"
    diary.intake.gp_name = gp_name
    diary.gp_channel = GPChannel(
        gp_name=gp_name,
        gp_email=gp_email,
    )
    return diary


def make_gp_query_event(
    patient_id: str = "PT-300",
    query_type: str = "missing_lab_results",
    requested_data: list | None = None,
) -> EventEnvelope:
    return EventEnvelope.handoff(
        event_type=EventType.GP_QUERY,
        patient_id=patient_id,
        source_agent="clinical",
        payload={
            "query_type": query_type,
            "reason": "Missing lab results for clinical assessment",
            "requested_data": requested_data or ["Recent blood test results", "Current medication list"],
            "channel": "websocket",
        },
    )


def make_gp_reminder_event(patient_id: str = "PT-300") -> EventEnvelope:
    return EventEnvelope.handoff(
        event_type=EventType.GP_REMINDER,
        patient_id=patient_id,
        source_agent="system",
        payload={"channel": "websocket"},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GP Query Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPQuery:
    """GP_QUERY event → email generation."""

    @pytest.mark.asyncio
    async def test_generates_query_email(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        # Should have 2 responses: email to GP + notification to patient
        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_email_goes_to_gp(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        gp_response = [r for r in result.responses if r.channel == "email"]
        assert len(gp_response) == 1
        assert gp_response[0].recipient == "gp:Dr. Patel"

    @pytest.mark.asyncio
    async def test_email_metadata_correct(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        gp_response = [r for r in result.responses if r.channel == "email"][0]
        assert gp_response.metadata["to"] == "dr.patel@nhs.net"
        assert "MedForce" in gp_response.metadata["subject"]
        assert "query_id" in gp_response.metadata
        assert "reply_to" in gp_response.metadata

    @pytest.mark.asyncio
    async def test_email_body_contains_patient_info(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        gp_response = [r for r in result.responses if r.channel == "email"][0]
        assert "Alice Brown" in gp_response.message
        assert "PT-300" in gp_response.message
        assert "Dear Dr. Patel" in gp_response.message

    @pytest.mark.asyncio
    async def test_email_body_contains_requested_data(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event(
            requested_data=["Recent LFTs", "Hepatitis serology"]
        )

        result = await handler.process(event, diary)

        gp_response = [r for r in result.responses if r.channel == "email"][0]
        assert "Recent LFTs" in gp_response.message
        assert "Hepatitis serology" in gp_response.message

    @pytest.mark.asyncio
    async def test_patient_notification_sent(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        patient_response = [r for r in result.responses if r.recipient == "patient"]
        assert len(patient_response) == 1
        assert "gp" in patient_response[0].message.lower()

    @pytest.mark.asyncio
    async def test_records_query_in_diary(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        assert len(result.updated_diary.gp_channel.queries) == 1
        query = result.updated_diary.gp_channel.queries[0]
        assert query.status == "pending"
        assert query.query_id.startswith("GPQ-")

    @pytest.mark.asyncio
    async def test_no_email_when_gp_email_missing(self):
        handler = GPCommunicationHandler()
        diary = make_diary(gp_email="")
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        # Should only have patient notification, no email
        email_responses = [r for r in result.responses if r.channel == "email"]
        assert len(email_responses) == 0
        # But patient should still be notified
        patient_responses = [r for r in result.responses if r.recipient == "patient"]
        assert len(patient_responses) == 1

    @pytest.mark.asyncio
    async def test_query_without_specific_data(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = EventEnvelope.handoff(
            event_type=EventType.GP_QUERY,
            patient_id="PT-300",
            source_agent="clinical",
            payload={
                "query_type": "general",
                "reason": "Need additional clinical information",
                "channel": "websocket",
            },
        )

        result = await handler.process(event, diary)

        gp_response = [r for r in result.responses if r.channel == "email"]
        assert len(gp_response) == 1
        assert "additional clinical information" in gp_response[0].message.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GP Reminder Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPReminder:
    """GP_REMINDER event → follow-up reminders."""

    @pytest.mark.asyncio
    async def test_sends_reminder_for_pending_query(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(query_id="GPQ-001", status="pending"),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        # Should send a reminder email
        email_responses = [r for r in result.responses if r.channel == "email"]
        assert len(email_responses) == 1
        assert "REMINDER" in email_responses[0].metadata["subject"]

    @pytest.mark.asyncio
    async def test_reminder_marks_timestamp(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(query_id="GPQ-001", status="pending"),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        query = result.updated_diary.gp_channel.queries[0]
        assert query.reminder_sent is not None

    @pytest.mark.asyncio
    async def test_no_reminder_for_responded_query(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(query_id="GPQ-001", status="responded"),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        # Should NOT send any reminder
        email_responses = [r for r in result.responses if r.channel == "email"]
        assert len(email_responses) == 0

    @pytest.mark.asyncio
    async def test_no_double_reminder(self):
        """Already reminded query should NOT get another reminder."""
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(
                query_id="GPQ-001",
                status="pending",
                reminder_sent=datetime.now(timezone.utc) - timedelta(days=1),
                sent=datetime.now(timezone.utc) - timedelta(days=3),
            ),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        email_responses = [r for r in result.responses if r.channel == "email"]
        assert len(email_responses) == 0

    @pytest.mark.asyncio
    async def test_7_day_non_responsive_fallback(self):
        """After 7 days with no response, mark GP as non_responsive."""
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(
                query_id="GPQ-001",
                status="pending",
                reminder_sent=datetime.now(timezone.utc) - timedelta(days=5),
                sent=datetime.now(timezone.utc) - timedelta(days=8),
            ),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        query = result.updated_diary.gp_channel.queries[0]
        assert query.status == "non_responsive"

    @pytest.mark.asyncio
    async def test_reminder_email_content(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(query_id="GPQ-001", status="pending"),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        email = [r for r in result.responses if r.channel == "email"][0]
        assert "reminder" in email.message.lower()
        assert "GPQ-001" in email.message
        assert "Dear Dr. Patel" in email.message

    @pytest.mark.asyncio
    async def test_multiple_pending_queries_first_reminder(self):
        """Multiple pending queries — should remind the un-reminded one."""
        handler = GPCommunicationHandler()
        diary = make_diary()
        diary.gp_channel.queries = [
            GPQuery(
                query_id="GPQ-001",
                status="pending",
                reminder_sent=datetime.now(timezone.utc),
                sent=datetime.now(timezone.utc) - timedelta(days=2),
            ),
            GPQuery(query_id="GPQ-002", status="pending"),
        ]
        event = make_gp_reminder_event()

        result = await handler.process(event, diary)

        # Should remind GPQ-002 (no reminder_sent yet)
        email_responses = [r for r in result.responses if r.channel == "email"]
        assert len(email_responses) == 1
        assert "GPQ-002" in email_responses[0].message


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Unexpected Event
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUnexpectedEvent:
    """GPCommunicationHandler with unrecognized events."""

    @pytest.mark.asyncio
    async def test_unexpected_event_returns_unchanged_diary(self):
        handler = GPCommunicationHandler()
        diary = make_diary()
        event = EventEnvelope(
            event_type=EventType.USER_MESSAGE,
            patient_id="PT-300",
            payload={"text": "hello"},
            sender_role=SenderRole.PATIENT,
        )

        result = await handler.process(event, diary)

        assert result.updated_diary == diary
        assert len(result.responses) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPCommsScenarios:
    """Realistic GP communication scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_full_query_cycle(self):
        """Query → Reminder → GP responds."""
        handler = GPCommunicationHandler()
        diary = make_diary()

        # Step 1: Send initial query
        query_event = make_gp_query_event(
            requested_data=["Full blood count", "Liver function tests"]
        )
        result1 = await handler.process(query_event, diary)
        diary = result1.updated_diary

        assert len(diary.gp_channel.queries) == 1
        assert diary.gp_channel.queries[0].status == "pending"

        # Step 2: Send reminder (after 48 hours)
        reminder_event = make_gp_reminder_event()
        result2 = await handler.process(reminder_event, diary)
        diary = result2.updated_diary

        assert diary.gp_channel.queries[0].reminder_sent is not None

    @pytest.mark.asyncio
    async def test_scenario_gp_no_email(self):
        """GP has no email — query is recorded but no email sent."""
        handler = GPCommunicationHandler()
        diary = make_diary(gp_email="")
        event = make_gp_query_event()

        result = await handler.process(event, diary)

        # Query should still be recorded
        assert len(result.updated_diary.gp_channel.queries) == 1
        # No email sent, but patient still notified
        assert any(r.recipient == "patient" for r in result.responses)
        assert not any(r.channel == "email" for r in result.responses)

"""
Tests for Phase 6 Channel Ingest — Dialogflow, Email, Twilio.

Tests cover:
  - Envelope creation from raw webhook data
  - Identity resolution integration
  - Phone/email extraction and normalisation
  - Event type determination (USER_MESSAGE vs DOCUMENT_UPLOADED)
  - Dialogflow response formatting
  - GP reply-to address parsing
  - WhatsApp vs SMS detection (Twilio)
"""

import pytest
from unittest.mock import MagicMock

from medforce.gateway.events import EventType, SenderRole
from medforce.gateway.handlers.identity_resolver import (
    AmbiguousIdentity,
    IdentityRecord,
    IdentityResolver,
)


# ── Helper: Mock Identity Resolver ──


def make_resolver_with_patient(
    phone: str = "+447700900001",
    patient_id: str = "PT-600",
    role: str = "patient",
) -> IdentityResolver:
    resolver = IdentityResolver()
    resolver._index[resolver._normalise(phone)] = [
        IdentityRecord(
            patient_id=patient_id,
            sender_id="PATIENT" if role == "patient" else f"HELPER-{patient_id}",
            sender_role=role,
            name="Test Patient",
            permissions=["full_access"],
        )
    ]
    return resolver


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dialogflow Ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDialogflowIngest:
    """Dialogflow CX webhook → EventEnvelope."""

    @pytest.mark.asyncio
    async def test_basic_message(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        ingest = DialogflowIngest()
        envelope = await ingest.to_envelope({
            "text": "I have a question about my results",
            "sessionInfo": {
                "session": "projects/p/locations/l/agents/a/sessions/447700900001",
                "parameters": {"phone": "+447700900001"},
            },
        })
        assert envelope.event_type == EventType.USER_MESSAGE
        assert envelope.payload["text"] == "I have a question about my results"
        assert envelope.payload["phone"] == "+447700900001"

    @pytest.mark.asyncio
    async def test_with_identity_resolution(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        resolver = make_resolver_with_patient("+447700900001", "PT-600")
        ingest = DialogflowIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "text": "Hello",
            "sessionInfo": {"parameters": {"phone": "+447700900001"}},
        })
        assert envelope.patient_id == "PT-600"
        assert envelope.sender_role == SenderRole.PATIENT

    @pytest.mark.asyncio
    async def test_unknown_sender_empty_patient_id(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        resolver = IdentityResolver()  # empty index
        ingest = DialogflowIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "text": "Who am I?",
            "sessionInfo": {"parameters": {"phone": "+449999999999"}},
        })
        assert envelope.patient_id == ""

    @pytest.mark.asyncio
    async def test_media_attachment_triggers_document_event(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        ingest = DialogflowIngest()
        envelope = await ingest.to_envelope({
            "text": "Here are my lab results",
            "media": [
                {"url": "https://example.com/lab_results.pdf"},
            ],
            "sessionInfo": {"parameters": {"phone": "+447700900001"}},
        })
        assert envelope.event_type == EventType.DOCUMENT_UPLOADED
        assert len(envelope.payload["attachments"]) == 1

    @pytest.mark.asyncio
    async def test_phone_from_session_id(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        ingest = DialogflowIngest()
        envelope = await ingest.to_envelope({
            "text": "Test",
            "sessionInfo": {
                "session": "projects/p/locations/l/agents/a/sessions/+447700900001",
                "parameters": {},
            },
        })
        assert envelope.payload["phone"] == "+447700900001"

    @pytest.mark.asyncio
    async def test_dialogflow_response_format(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        ingest = DialogflowIngest()
        result = ingest.build_dialogflow_response([
            {"message": "First response"},
            {"message": "Second response"},
        ])
        assert "fulfillmentResponse" in result
        messages = result["fulfillmentResponse"]["messages"]
        assert len(messages) == 2
        assert messages[0]["text"]["text"] == ["First response"]

    @pytest.mark.asyncio
    async def test_empty_response_format(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        ingest = DialogflowIngest()
        result = ingest.build_dialogflow_response([])
        assert result["fulfillmentResponse"]["messages"] == []

    @pytest.mark.asyncio
    async def test_ambiguous_identity_uses_first(self):
        from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest

        resolver = IdentityResolver()
        phone = "+447700900001"
        resolver._index[resolver._normalise(phone)] = [
            IdentityRecord(
                patient_id="PT-A", sender_id="PATIENT",
                sender_role="patient", permissions=["full_access"],
            ),
            IdentityRecord(
                patient_id="PT-B", sender_id="HELPER-B",
                sender_role="helper", permissions=["view_status"],
            ),
        ]
        ingest = DialogflowIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "text": "Hello",
            "sessionInfo": {"parameters": {"phone": phone}},
        })
        assert envelope.patient_id == "PT-A"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Email Ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmailIngest:
    """SendGrid inbound parse webhook → EventEnvelope."""

    @pytest.mark.asyncio
    async def test_gp_reply_basic(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "dr.patel@greenfields.nhs.uk",
            "to": "gp-reply+PT-1234@medforce.app",
            "subject": "Re: MedForce — Lab Results Requested",
            "text": "Please find the lab results attached. ALT was 250.",
        })
        assert envelope.patient_id == "PT-1234"
        assert envelope.event_type == EventType.GP_RESPONSE
        assert envelope.sender_role == SenderRole.GP
        assert "ALT was 250" in envelope.payload["text"]

    @pytest.mark.asyncio
    async def test_extracts_patient_id_from_reply_to(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "dr.smith@nhs.net",
            "to": "gp-reply+PT-TEST-999@medforce.app",
            "text": "Response",
        })
        assert envelope.patient_id == "PT-TEST-999"

    @pytest.mark.asyncio
    async def test_with_attachments_is_document_uploaded(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "dr.kim@nhs.net",
            "to": "gp-reply+PT-500@medforce.app",
            "text": "Labs attached",
            "attachments": "2",
        })
        assert envelope.event_type == EventType.DOCUMENT_UPLOADED
        assert envelope.payload["attachment_count"] == 2
        assert len(envelope.payload["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_no_attachments_is_gp_response(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "dr.jones@nhs.net",
            "to": "gp-reply+PT-300@medforce.app",
            "text": "Patient is on metformin 500mg",
            "attachments": "0",
        })
        assert envelope.event_type == EventType.GP_RESPONSE

    @pytest.mark.asyncio
    async def test_email_in_angle_brackets(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "Dr. Patel <dr.patel@nhs.net>",
            "to": "gp-reply+PT-100@medforce.app",
            "text": "Response",
        })
        assert envelope.payload["sender_email"] == "dr.patel@nhs.net"

    @pytest.mark.asyncio
    async def test_no_reply_to_match(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "someone@example.com",
            "to": "info@medforce.app",
            "text": "General inquiry",
        })
        assert envelope.patient_id == ""

    @pytest.mark.asyncio
    async def test_with_identity_resolution(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        resolver = IdentityResolver()
        resolver._index["dr.smith@nhs.net"] = [
            IdentityRecord(
                patient_id="PT-200",
                sender_id="GP-Dr. Smith",
                sender_role="gp",
                permissions=["respond_to_queries"],
            )
        ]
        ingest = EmailIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "from": "dr.smith@nhs.net",
            "to": "gp-reply+PT-200@medforce.app",
            "text": "Here are the results",
        })
        assert envelope.sender_id == "GP-Dr. Smith"

    @pytest.mark.asyncio
    async def test_subject_extracted(self):
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "dr.test@nhs.net",
            "to": "gp-reply+PT-001@medforce.app",
            "subject": "Re: Lab Results for Alice Green",
            "text": "Results normal",
        })
        assert envelope.payload["subject"] == "Re: Lab Results for Alice Green"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Twilio Ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTwilioIngest:
    """Twilio incoming SMS/WhatsApp webhook → EventEnvelope."""

    @pytest.mark.asyncio
    async def test_basic_sms(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        ingest = TwilioSMSIngest()
        envelope = await ingest.to_envelope({
            "MessageSid": "SM123",
            "From": "+447700900001",
            "To": "+441234567890",
            "Body": "I'm feeling better today",
            "NumMedia": "0",
        })
        assert envelope.event_type == EventType.USER_MESSAGE
        assert envelope.payload["text"] == "I'm feeling better today"
        assert envelope.payload["phone"] == "+447700900001"
        assert envelope.payload["is_whatsapp"] is False
        assert envelope.payload["channel"] == "sms"

    @pytest.mark.asyncio
    async def test_whatsapp_message(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        ingest = TwilioSMSIngest()
        envelope = await ingest.to_envelope({
            "MessageSid": "SM456",
            "From": "whatsapp:+447700900001",
            "Body": "Hello from WhatsApp",
            "NumMedia": "0",
        })
        assert envelope.payload["is_whatsapp"] is True
        assert envelope.payload["phone"] == "+447700900001"
        # Channel is "sms" (Twilio's channel) — is_whatsapp flag distinguishes
        assert envelope.payload["channel"] == "sms"

    @pytest.mark.asyncio
    async def test_with_media(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        ingest = TwilioSMSIngest()
        envelope = await ingest.to_envelope({
            "From": "+447700900001",
            "Body": "Lab results photo",
            "NumMedia": "2",
            "MediaUrl0": "https://api.twilio.com/media/1.jpg",
            "MediaUrl1": "https://api.twilio.com/media/2.pdf",
        })
        assert envelope.event_type == EventType.DOCUMENT_UPLOADED
        assert len(envelope.payload["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_identity_resolution(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        resolver = make_resolver_with_patient("+447700900001", "PT-700")
        ingest = TwilioSMSIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "From": "+447700900001",
            "Body": "Update",
            "NumMedia": "0",
        })
        assert envelope.patient_id == "PT-700"
        assert envelope.sender_role == SenderRole.PATIENT

    @pytest.mark.asyncio
    async def test_unknown_sender(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        resolver = IdentityResolver()
        ingest = TwilioSMSIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "From": "+449999999999",
            "Body": "Who am I?",
            "NumMedia": "0",
        })
        assert envelope.patient_id == ""

    @pytest.mark.asyncio
    async def test_message_sid_preserved(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        ingest = TwilioSMSIngest()
        envelope = await ingest.to_envelope({
            "MessageSid": "SM_UNIQUE_123",
            "From": "+447700900001",
            "Body": "Test",
            "NumMedia": "0",
        })
        assert envelope.payload["message_sid"] == "SM_UNIQUE_123"

    @pytest.mark.asyncio
    async def test_no_media_count(self):
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        ingest = TwilioSMSIngest()
        envelope = await ingest.to_envelope({
            "From": "+447700900001",
            "Body": "Just text",
        })
        assert envelope.event_type == EventType.USER_MESSAGE
        assert envelope.payload["attachments"] == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cross-Channel Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCrossChannelScenarios:
    """End-to-end scenarios spanning ingest + envelope creation."""

    @pytest.mark.asyncio
    async def test_gp_email_reply_with_labs(self):
        """GP replies to email with lab results attached."""
        from medforce.gateway.ingest.email_ingest import EmailIngest

        ingest = EmailIngest()
        envelope = await ingest.to_envelope({
            "from": "Dr. Williams <dr.williams@nhs.net>",
            "to": "gp-reply+PT-CROSS-001@medforce.app",
            "subject": "Re: Lab Results — Carol White",
            "text": "Bilirubin 2.5, ALT 180, Platelets 150",
            "attachments": "1",
        })
        assert envelope.patient_id == "PT-CROSS-001"
        assert envelope.event_type == EventType.DOCUMENT_UPLOADED
        assert envelope.sender_role == SenderRole.GP

    @pytest.mark.asyncio
    async def test_patient_whatsapp_with_photo(self):
        """Patient sends WhatsApp photo of lab results."""
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        resolver = make_resolver_with_patient("+447700900001", "PT-CROSS-002")
        ingest = TwilioSMSIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "From": "whatsapp:+447700900001",
            "Body": "Here are my blood test results",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/photo.jpg",
        })
        assert envelope.patient_id == "PT-CROSS-002"
        assert envelope.event_type == EventType.DOCUMENT_UPLOADED
        assert envelope.payload["is_whatsapp"] is True

    @pytest.mark.asyncio
    async def test_helper_sms_message(self):
        """Helper sends SMS on behalf of patient."""
        from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest

        resolver = IdentityResolver()
        resolver._index[resolver._normalise("+447700900010")] = [
            IdentityRecord(
                patient_id="PT-CROSS-003",
                sender_id="HELPER-SARAH",
                sender_role="helper",
                name="Sarah Moore",
                relationship="spouse",
                permissions=["full_access"],
            )
        ]
        ingest = TwilioSMSIngest(identity_resolver=resolver)

        envelope = await ingest.to_envelope({
            "From": "+447700900010",
            "Body": "My husband is feeling worse today",
            "NumMedia": "0",
        })
        assert envelope.patient_id == "PT-CROSS-003"
        assert envelope.sender_role == SenderRole.HELPER

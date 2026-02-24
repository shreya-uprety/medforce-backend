"""
GP Communication Handler — manages outbound queries to GPs and processes responses.

Handles:
  - GP_QUERY: generate and send a professional query email to the patient's GP
  - GP_REMINDER: send a follow-up reminder after 48 hours of no response

Uses the ChannelDispatcher abstraction for delivery — returns AgentResponse
with channel="email".  In Phases 1-5, no email dispatcher is registered so
the response is stored in the diary with delivery_status="pending_channel".
In Phase 6, registering an EmailDispatcher delivers automatically.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from medforce.gateway.agents.base_agent import AgentResult, BaseAgent
from medforce.gateway.channels import AgentResponse
from medforce.gateway.diary import GPQuery, PatientDiary
from medforce.gateway.events import EventEnvelope, EventType

logger = logging.getLogger("gateway.handlers.gp_comms")


class GPCommunicationHandler(BaseAgent):
    """
    Handles GP query and reminder events.

    Generates professional email content and returns it as an AgentResponse
    with channel="email" — the DispatcherRegistry handles actual delivery.
    """

    agent_name = "gp_comms"

    async def process(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        if event.event_type == EventType.GP_QUERY:
            return await self._handle_query(event, diary)

        if event.event_type == EventType.GP_REMINDER:
            return await self._handle_reminder(event, diary)

        logger.warning(
            "GPCommsHandler received unexpected event: %s", event.event_type.value
        )
        return AgentResult(updated_diary=diary)

    async def _handle_query(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Generate and send a query to the patient's GP."""
        query_type = event.payload.get("query_type", "general")
        query_reason = event.payload.get("reason", "")
        specific_data = event.payload.get("requested_data", [])

        gp_name = diary.gp_channel.gp_name or diary.intake.gp_name or "Doctor"
        gp_email = diary.gp_channel.gp_email or ""
        patient_name = diary.intake.name or "the patient"
        patient_id = diary.header.patient_id

        # Generate query ID
        query_id = f"GPQ-{patient_id}-{uuid.uuid4().hex[:6]}"

        # Build professional email body
        if specific_data:
            data_list = "\n".join(f"  - {item}" for item in specific_data)
            body = (
                f"Dear {gp_name},\n\n"
                f"We are writing regarding your patient {patient_name} "
                f"(Patient ID: {patient_id}), who is currently undergoing "
                f"pre-consultation assessment at MedForce.\n\n"
                f"Our clinical team has identified a need for the following "
                f"information to complete the assessment:\n\n"
                f"{data_list}\n\n"
                f"Could you please provide this information at your earliest "
                f"convenience? You can reply directly to this email or upload "
                f"documents via our secure portal.\n\n"
                f"Query Reference: {query_id}\n\n"
                f"Kind regards,\n"
                f"MedForce Clinical Team"
            )
        else:
            body = (
                f"Dear {gp_name},\n\n"
                f"We are writing regarding your patient {patient_name} "
                f"(Patient ID: {patient_id}), who is currently undergoing "
                f"pre-consultation assessment at MedForce.\n\n"
                f"{query_reason or 'We require additional clinical information to complete the assessment.'}\n\n"
                f"Could you please respond at your earliest convenience?\n\n"
                f"Query Reference: {query_id}\n\n"
                f"Kind regards,\n"
                f"MedForce Clinical Team"
            )

        # Record in diary
        gp_query = GPQuery(
            query_id=query_id,
            query_type=query_type,
            query_text=body,
            status="pending",
        )
        diary.gp_channel.add_query(gp_query)

        # Build response — uses channel="email"
        responses = []

        if gp_email:
            responses.append(
                AgentResponse(
                    recipient=f"gp:{gp_name}",
                    channel="email",
                    message=body,
                    metadata={
                        "patient_id": patient_id,
                        "subject": f"MedForce — Clinical Information Request for {patient_name} ({query_id})",
                        "to": gp_email,
                        "reply_to": f"gp-reply+{patient_id}@medforce.app",
                        "query_id": query_id,
                    },
                )
            )

        # Also notify the patient
        patient_channel = event.payload.get("channel", "websocket")
        responses.append(
            AgentResponse(
                recipient="patient",
                channel=patient_channel,
                message=(
                    f"We've sent a request to your GP ({gp_name}) for some "
                    f"additional clinical information. This is a routine part "
                    f"of the assessment process and won't delay your care. "
                    f"We'll let you know when we hear back."
                ),
                metadata={"patient_id": patient_id},
            )
        )

        logger.info(
            "GP query %s sent for patient %s to %s",
            query_id,
            patient_id,
            gp_name,
        )

        return AgentResult(updated_diary=diary, responses=responses)

    async def _handle_reminder(
        self, event: EventEnvelope, diary: PatientDiary
    ) -> AgentResult:
        """Send a reminder for unanswered GP queries."""
        gp_name = diary.gp_channel.gp_name or diary.intake.gp_name or "Doctor"
        gp_email = diary.gp_channel.gp_email or ""
        patient_name = diary.intake.name or "the patient"
        patient_id = diary.header.patient_id

        responses = []
        reminded = False

        for query in diary.gp_channel.queries:
            if query.status != "pending":
                continue
            if query.reminder_sent is not None:
                # Already reminded — check for 7-day fallback
                sent_dt = query.sent
                if isinstance(sent_dt, str):
                    continue
                days_since = (datetime.now(timezone.utc) - sent_dt).days
                if days_since >= 7:
                    query.status = "non_responsive"
                    logger.info(
                        "GP marked non-responsive for query %s (patient %s)",
                        query.query_id,
                        patient_id,
                    )
                continue

            # Send reminder
            query.reminder_sent = datetime.now(timezone.utc)

            body = (
                f"Dear {gp_name},\n\n"
                f"This is a gentle reminder regarding our previous request "
                f"for clinical information about your patient {patient_name} "
                f"(Patient ID: {patient_id}).\n\n"
                f"Original query reference: {query.query_id}\n\n"
                f"We would appreciate a response at your earliest convenience.\n\n"
                f"Kind regards,\n"
                f"MedForce Clinical Team"
            )

            if gp_email:
                responses.append(
                    AgentResponse(
                        recipient=f"gp:{gp_name}",
                        channel="email",
                        message=body,
                        metadata={
                            "patient_id": patient_id,
                            "subject": f"REMINDER: MedForce — Clinical Information Request ({query.query_id})",
                            "to": gp_email,
                            "reply_to": f"gp-reply+{patient_id}@medforce.app",
                            "query_id": query.query_id,
                        },
                    )
                )
            reminded = True

        if reminded:
            logger.info(
                "GP reminder sent for patient %s to %s", patient_id, gp_name
            )

        return AgentResult(updated_diary=diary, responses=responses)

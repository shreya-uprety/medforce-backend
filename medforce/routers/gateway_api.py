"""
Gateway API — HTTP endpoints for the event-driven Gateway.

Endpoints:
  POST /api/gateway/emit                Submit an event to the Gateway
  POST /api/gateway/upload/{id}         Upload documents (lab reports, imaging, etc.)
  GET  /api/gateway/diary/{id}          Read a patient's diary
  GET  /api/gateway/chat/{id}           Read patient chat history from GCS
  GET  /api/gateway/documents/{id}      List uploaded documents for a patient
  GET  /api/gateway/events/{id}         Read event log for a patient
  GET  /api/gateway/status              Health + active queue info
  GET  /api/gateway/responses/{id}      Read test harness responses
  POST /api/gateway/scenario/load       Seed diary with test scenario data
  DELETE /api/gateway/reset/{id}        Clear diary + events for a patient
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from medforce.gateway.diary import DiaryNotFoundError
from medforce.gateway.events import EventEnvelope, EventType, SenderRole

logger = logging.getLogger("gateway.api")

router = APIRouter(prefix="/api/gateway", tags=["gateway"])

# Strong references to background tasks to prevent GC before completion
_background_tasks: set[asyncio.Task] = set()


# ── Request / Response Models ──


class EmitEventRequest(BaseModel):
    """Request body for POST /api/gateway/emit."""

    event_type: str
    patient_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    sender_id: str = ""
    sender_role: str = "patient"
    source: str = ""
    correlation_id: str | None = None


class EmitEventResponse(BaseModel):
    """Response for POST /api/gateway/emit."""

    success: bool
    event_id: str = ""
    message: str = ""


class FileAttachment(BaseModel):
    """A single file attachment, Base64-encoded."""

    filename: str
    content_base64: str  # Base64-encoded file content (with or without data URI prefix)


class UploadDocumentsRequest(BaseModel):
    """Request body for POST /api/gateway/upload/{patient_id}."""

    attachments: list[FileAttachment]
    channel: str = "websocket"
    sender_role: str = "patient"


class GatewayStatusResponse(BaseModel):
    """Response for GET /api/gateway/status."""

    status: str = "ok"
    active_queues: int = 0
    active_patients: list[str] = Field(default_factory=list)
    registered_agents: list[str] = Field(default_factory=list)
    registered_channels: list[str] = Field(default_factory=list)


# ── Endpoints ──


@router.post("/emit", response_model=EmitEventResponse)
async def emit_event(request: EmitEventRequest):
    """
    Submit an event to the Gateway for processing.

    The event is validated, wrapped in an EventEnvelope, and enqueued
    via the PatientQueueManager for serialized per-patient processing.
    """
    from medforce.gateway.setup import get_gateway, get_queue_manager

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    # Validate event type
    try:
        event_type = EventType(request.event_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type: {request.event_type}. "
                   f"Valid types: {[e.value for e in EventType]}",
        )

    # Validate sender role
    try:
        sender_role = SenderRole(request.sender_role)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sender_role: {request.sender_role}. "
                   f"Valid roles: {[r.value for r in SenderRole]}",
        )

    # Build envelope
    envelope = EventEnvelope(
        event_type=event_type,
        patient_id=request.patient_id,
        payload=request.payload,
        sender_id=request.sender_id or request.patient_id,
        sender_role=sender_role,
        source=request.source or "api",
        correlation_id=request.correlation_id,
    )

    # Route through the per-patient queue for serialized processing
    queue_manager = get_queue_manager()
    if queue_manager is not None:
        await queue_manager.enqueue(envelope)
    else:
        # Fallback: direct background processing if queue manager unavailable
        async def _process_in_background(gw, env):
            try:
                t0 = time.monotonic()
                await gw.process_event(env)
                elapsed = time.monotonic() - t0
                logger.info(
                    "Event %s for %s processed in %.2fs",
                    env.event_type.value, env.patient_id, elapsed,
                )
            except Exception as exc:
                logger.error("Error processing event %s: %s", env.event_id, exc, exc_info=True)

        task = asyncio.create_task(_process_in_background(gateway, envelope))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return EmitEventResponse(
        success=True,
        event_id=envelope.event_id,
        message=f"Event {event_type.value} accepted for patient {request.patient_id}",
    )


@router.post("/upload/{patient_id}")
async def upload_documents(patient_id: str, request: UploadDocumentsRequest):
    """
    Upload documents (lab reports, imaging, referral letters, etc.) for a patient.

    Accepts Base64-encoded files, stores them in GCS at
    patient_data/{patient_id}/raw_data/{filename}, and emits a
    DOCUMENT_UPLOADED event through the Gateway for each file.
    """
    from medforce.dependencies import get_gcs
    from medforce.gateway.setup import get_gateway

    gcs = get_gcs()
    gateway = get_gateway()

    if gcs is None:
        raise HTTPException(status_code=503, detail="GCS not initialized")
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    uploaded = []
    errors = []

    CONTENT_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".json": "application/json",
    }

    import hashlib

    for att in request.attachments:
        try:
            # Decode Base64 content (handle data URI prefix)
            encoded = att.content_base64
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            file_bytes = base64.b64decode(encoded)

            # P3: Compute content hash for deduplication
            content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

            # Determine content type from extension
            ext = ""
            if "." in att.filename:
                ext = "." + att.filename.rsplit(".", 1)[1].lower()
            content_type = CONTENT_TYPES.get(ext, "application/octet-stream")

            # Store in GCS: patient_data/{patient_id}/raw_data/{filename}
            gcs_path = f"patient_data/{patient_id}/raw_data/{att.filename}"
            await asyncio.to_thread(
                gcs.create_file_from_string,
                file_bytes, gcs_path, content_type,
            )

            # Emit DOCUMENT_UPLOADED event through the gateway
            try:
                sender_role = SenderRole(request.sender_role)
            except ValueError:
                sender_role = SenderRole.PATIENT

            envelope = EventEnvelope(
                event_type=EventType.DOCUMENT_UPLOADED,
                patient_id=patient_id,
                sender_id=patient_id,
                sender_role=sender_role,
                payload={
                    "file_ref": f"gs://clinic_sim_dev/{gcs_path}",
                    "type": _detect_document_type(att.filename),
                    "channel": request.channel,
                    "filename": att.filename,
                    "content_hash": content_hash,
                },
            )
            await gateway.process_event(envelope)

            uploaded.append({
                "filename": att.filename,
                "gcs_path": gcs_path,
                "content_type": content_type,
                "size_bytes": len(file_bytes),
            })

        except Exception as exc:
            logger.error("Failed to upload %s: %s", att.filename, exc)
            errors.append({"filename": att.filename, "error": str(exc)})

    return {
        "success": len(errors) == 0,
        "patient_id": patient_id,
        "uploaded": uploaded,
        "errors": errors,
    }


def _detect_document_type(filename: str) -> str:
    """Infer document type from filename."""
    name_lower = filename.lower()
    if any(kw in name_lower for kw in ["lab", "blood", "test_result"]):
        return "lab_results"
    if any(kw in name_lower for kw in ["xray", "x-ray", "scan", "mri", "ct", "imaging", "ultrasound"]):
        return "imaging"
    if any(kw in name_lower for kw in ["referral", "letter", "ref"]):
        return "referral"
    if any(kw in name_lower for kw in ["nhs", "screenshot"]):
        return "nhs_screenshot"
    return "document"


@router.get("/chat/{patient_id}")
async def get_chat_history(patient_id: str):
    """
    Read the patient's chat history from GCS.

    Returns the conversation stored at patient_data/{patient_id}/pre_consultation_chat.json.
    """
    from medforce.dependencies import get_gcs

    gcs = get_gcs()
    if gcs is None:
        raise HTTPException(status_code=503, detail="GCS not initialized")

    file_path = f"patient_data/{patient_id}/pre_consultation_chat.json"
    content = await asyncio.to_thread(gcs.read_file_as_string, file_path)

    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"No chat history found for patient {patient_id}",
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse chat history")


@router.get("/documents/{patient_id}")
async def list_documents(patient_id: str):
    """
    List all uploaded documents for a patient.

    Lists files in GCS at patient_data/{patient_id}/raw_data/.
    """
    from medforce.dependencies import get_gcs

    gcs = get_gcs()
    if gcs is None:
        raise HTTPException(status_code=503, detail="GCS not initialized")

    folder = f"patient_data/{patient_id}/raw_data"
    try:
        files = await asyncio.to_thread(gcs.list_files, folder)
        return {
            "patient_id": patient_id,
            "count": len(files),
            "documents": files,
            "gcs_prefix": f"gs://clinic_sim_dev/{folder}/",
        }
    except Exception as exc:
        logger.error("Failed to list documents for %s: %s", patient_id, exc)
        return {"patient_id": patient_id, "count": 0, "documents": []}


@router.get("/diary/{patient_id}")
async def get_diary(patient_id: str):
    """Read a patient's current diary state."""
    from medforce.gateway.setup import get_diary_store

    diary_store = get_diary_store()
    if diary_store is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        diary, generation = await asyncio.to_thread(
            diary_store.load, patient_id
        )
        return {
            "patient_id": patient_id,
            "generation": generation,
            "diary": diary.model_dump(mode="json"),
        }
    except DiaryNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No diary found for patient {patient_id}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/events/{patient_id}")
async def get_events(patient_id: str, limit: int = 50):
    """Read the event log for a patient."""
    from medforce.gateway.setup import get_gateway

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    events = gateway.get_event_log(patient_id=patient_id, limit=limit)
    return {
        "patient_id": patient_id,
        "count": len(events),
        "events": events,
    }


@router.get("/status", response_model=GatewayStatusResponse)
async def gateway_status():
    """Health check + active queue info for the Gateway."""
    from medforce.gateway.setup import (
        get_dispatcher_registry,
        get_gateway,
        get_queue_manager,
    )

    gateway = get_gateway()
    queue_manager = get_queue_manager()
    dispatcher_registry = get_dispatcher_registry()

    if gateway is None:
        return GatewayStatusResponse(status="not_initialized")

    return GatewayStatusResponse(
        status="ok",
        active_queues=queue_manager.active_count if queue_manager else 0,
        active_patients=queue_manager.active_patients if queue_manager else [],
        registered_agents=gateway.registered_agents,
        registered_channels=(
            dispatcher_registry.registered_channels
            if dispatcher_registry
            else []
        ),
    )


@router.get("/health")
async def gateway_health():
    """P2: Detailed health check — verifies agents, diary store, channels."""
    from medforce.gateway.setup import get_gateway

    gateway = get_gateway()
    if gateway is None:
        return {"healthy": False, "reason": "Gateway not initialized"}

    return gateway.health_check()


@router.get("/metrics")
async def gateway_metrics():
    """P2: Observability metrics — processing times, error rates, DLQ size."""
    from medforce.gateway.setup import get_gateway

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    return gateway.get_metrics()


@router.get("/dlq")
async def gateway_dlq(limit: int = 50):
    """P2: Dead Letter Queue — failed events for ops review and replay."""
    from medforce.gateway.setup import get_gateway

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    entries = gateway.get_dlq(limit=limit)
    return {"count": len(entries), "entries": entries}


# ── Test Harness Endpoints ──


@router.get("/responses/{patient_id}")
async def get_responses(patient_id: str):
    """Read test harness stored responses for a patient."""
    from medforce.gateway.setup import get_dispatcher_registry

    registry = get_dispatcher_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    harness = None
    for dispatcher in registry._dispatchers.values():
        if hasattr(dispatcher, "get_responses"):
            harness = dispatcher
            break

    if harness is None:
        return {"patient_id": patient_id, "count": 0, "responses": []}

    responses = harness.get_responses(patient_id)
    return {
        "patient_id": patient_id,
        "count": len(responses),
        "responses": [
            {
                "recipient": r.recipient,
                "channel": r.channel,
                "message": r.message,
                "metadata": r.metadata,
            }
            for r in responses
        ],
    }


class ScenarioLoadRequest(BaseModel):
    """Request body for POST /api/gateway/scenario/load."""

    scenario: str  # scenario name/ID
    patient_id: str = "PT-TEST-001"


# Test scenario data definitions
SCENARIOS = {
    "happy_path_low_risk": {
        "description": "Happy Path — Solo Patient, Low Risk",
        "intake": {
            "name": "Alice Green", "dob": "1985-06-15",
            "nhs_number": "1234567890", "phone": "07700900001",
            "email": "alice@example.com", "gp_name": "Dr. Williams",
            "gp_practice": "Elm Street Practice", "address": "10 Elm Street",
        },
        "clinical": {
            "chief_complaint": "routine health check",
            "medical_history": ["none significant"],
            "current_medications": [],
        },
        "risk_level": "low",
    },
    "urgent_high_risk": {
        "description": "Urgent — Patient + Spouse, High Risk",
        "intake": {
            "name": "Bob Harris", "dob": "1970-03-22",
            "nhs_number": "9876543210", "phone": "07700900002",
            "email": "bob@example.com", "gp_name": "Dr. Patel",
            "gp_practice": "Oak Road Surgery",
        },
        "clinical": {
            "chief_complaint": "severe jaundice and abdominal pain",
            "medical_history": ["chronic liver disease", "diabetes"],
            "current_medications": ["metformin 500mg", "warfarin 5mg"],
            "red_flags": ["jaundice", "ascites"],
        },
        "risk_level": "high",
        "lab_values": {"bilirubin": 7.0, "ALT": 600, "platelets": 45},
    },
    "gp_query_required": {
        "description": "Missing Info — GP Query Required",
        "intake": {
            "name": "Carol White", "dob": "1992-11-08",
            "nhs_number": "5555555555", "phone": "07700900003",
            "gp_name": "Dr. Smith", "gp_practice": "Pine Medical Centre",
        },
        "clinical": {
            "chief_complaint": "elevated liver enzymes on routine blood test",
            "medical_history": ["hypertension"],
        },
        "gp_channel": {"gp_name": "Dr. Smith", "gp_email": "dr.smith@nhs.net"},
        "risk_level": "medium",
    },
    "backward_loop": {
        "description": "Backward Loop — Missing Medications",
        "intake": {
            "name": "David Brown", "dob": "1988-01-30",
            "nhs_number": "1111111111",
            "gp_name": "Dr. Jones",
        },
        "clinical": {
            "chief_complaint": "fatigue and abdominal discomfort",
        },
        "risk_level": "medium",
        "missing_phone": True,
    },
    "deterioration_escalation": {
        "description": "Deterioration Escalation (Month 3)",
        "intake": {
            "name": "Eve Taylor", "dob": "1975-09-12",
            "nhs_number": "2222222222", "phone": "07700900005",
            "gp_name": "Dr. Brown",
        },
        "clinical": {
            "chief_complaint": "chronic hepatitis monitoring",
            "medical_history": ["hepatitis C", "previous liver biopsy"],
            "current_medications": ["ribavirin"],
        },
        "risk_level": "medium",
        "monitoring": {
            "baseline": {"bilirubin": 2.5, "ALT": 150, "platelets": 180},
            "appointment_date": "2026-01-01",
        },
    },
    "multi_helper": {
        "description": "Multi-Helper Permission Conflict",
        "intake": {
            "name": "Frank Moore", "dob": "1960-04-18",
            "nhs_number": "3333333333", "phone": "07700900006",
            "gp_name": "Dr. Lee",
        },
        "helpers": [
            {"id": "helper-sarah", "name": "Sarah Moore", "relationship": "spouse",
             "permissions": ["full_access"], "verified": True},
            {"id": "helper-jim", "name": "Jim Moore", "relationship": "friend",
             "permissions": ["view_status"], "verified": True},
        ],
        "risk_level": "low",
    },
    "gp_non_responsive": {
        "description": "GP Non-Responsive",
        "intake": {
            "name": "Grace Chen", "dob": "1995-07-25",
            "nhs_number": "4444444444", "phone": "07700900007",
            "gp_name": "Dr. Kim",
        },
        "gp_channel": {"gp_name": "Dr. Kim", "gp_email": "dr.kim@nhs.net"},
        "risk_level": "medium",
    },
    "unknown_sender": {
        "description": "Unknown Sender",
        "intake": {
            "name": "Test Patient", "dob": "2000-01-01",
            "nhs_number": "9999999999", "phone": "07700900008",
            "gp_name": "Dr. Test",
        },
        "risk_level": "none",
    },
}


@router.post("/scenario/load")
async def load_scenario(request: ScenarioLoadRequest):
    """Seed a patient diary with test scenario data."""
    from medforce.gateway.diary import (
        ClinicalDocument,
        GPChannel,
        HelperEntry,
        PatientDiary,
        Phase,
        RiskLevel,
    )
    from medforce.gateway.setup import get_diary_store

    diary_store = get_diary_store()
    if diary_store is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    scenario = SCENARIOS.get(request.scenario)
    if scenario is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario: {request.scenario}. "
                   f"Available: {list(SCENARIOS.keys())}",
        )

    # Create diary
    diary = PatientDiary.create_new(request.patient_id)

    # Apply intake data
    intake_data = scenario.get("intake", {})
    for field, value in intake_data.items():
        if hasattr(diary.intake, field):
            setattr(diary.intake, field, value)
            diary.intake.mark_field_collected(field, value)

    # Apply clinical data
    clinical_data = scenario.get("clinical", {})
    for field, value in clinical_data.items():
        if hasattr(diary.clinical, field):
            setattr(diary.clinical, field, value)

    # Set risk level
    risk_str = scenario.get("risk_level", "none")
    try:
        diary.header.risk_level = RiskLevel(risk_str)
        diary.clinical.risk_level = RiskLevel(risk_str)
    except ValueError:
        pass

    # Lab values as document
    lab_values = scenario.get("lab_values")
    if lab_values:
        diary.clinical.documents.append(
            ClinicalDocument(
                type="lab_results", source="scenario",
                processed=True, extracted_values=lab_values,
            )
        )

    # GP channel
    gp_data = scenario.get("gp_channel")
    if gp_data:
        diary.gp_channel = GPChannel(**gp_data)

    # Helpers
    helpers = scenario.get("helpers", [])
    for h in helpers:
        diary.helper_registry.add_helper(HelperEntry(**h))

    # Monitoring pre-setup
    monitoring_data = scenario.get("monitoring")
    if monitoring_data:
        diary.monitoring.baseline = monitoring_data.get("baseline", {})
        diary.monitoring.appointment_date = monitoring_data.get("appointment_date")
        diary.monitoring.monitoring_active = True
        diary.header.current_phase = Phase.MONITORING

    # Missing phone scenario
    if scenario.get("missing_phone"):
        diary.intake.phone = None

    # Save
    try:
        await asyncio.to_thread(diary_store.save, request.patient_id, diary)
    except Exception:
        await asyncio.to_thread(diary_store.create, request.patient_id)
        await asyncio.to_thread(diary_store.save, request.patient_id, diary)

    return {
        "success": True,
        "patient_id": request.patient_id,
        "scenario": request.scenario,
        "description": scenario.get("description", ""),
        "phase": diary.header.current_phase.value,
    }


@router.delete("/reset/{patient_id}")
async def reset_patient(patient_id: str):
    """Clear diary + events + test harness responses for a patient."""
    from medforce.gateway.setup import get_diary_store, get_dispatcher_registry, get_gateway

    diary_store = get_diary_store()
    gateway = get_gateway()

    if diary_store is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    deleted = await asyncio.to_thread(diary_store.delete, patient_id)

    # Clear event log for this patient
    if gateway:
        gateway._event_log = [
            e for e in gateway._event_log
            if e.get("patient_id") != patient_id
        ]

    # Clear test harness responses for this patient
    registry = get_dispatcher_registry()
    if registry:
        for dispatcher in registry._dispatchers.values():
            if hasattr(dispatcher, "clear"):
                dispatcher.clear(patient_id)

    return {
        "success": True,
        "patient_id": patient_id,
        "diary_deleted": deleted,
    }


@router.get("/scenarios")
async def list_scenarios():
    """List all available test scenarios."""
    return {
        "scenarios": [
            {"id": k, "description": v.get("description", "")}
            for k, v in SCENARIOS.items()
        ]
    }


# ── Phase 6: Channel Webhook Endpoints ──


@router.post("/dialogflow-webhook")
async def dialogflow_webhook(request_body: dict[str, Any]):
    """
    Dialogflow CX fulfillment webhook.

    Receives incoming WhatsApp/SMS messages relayed through Dialogflow CX,
    converts them to EventEnvelopes, processes through the Gateway, and
    returns Dialogflow-formatted responses.
    """
    from medforce.gateway.ingest.dialogflow_ingest import DialogflowIngest
    from medforce.gateway.setup import get_gateway, get_identity_resolver

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    identity_resolver = get_identity_resolver()
    ingest = DialogflowIngest(identity_resolver=identity_resolver)

    try:
        envelope = await ingest.to_envelope(request_body)
    except Exception as exc:
        logger.error("Dialogflow ingest error: %s", exc, exc_info=True)
        return ingest.build_dialogflow_response([{
            "message": "Sorry, we couldn't process your message. Please try again."
        }])

    if not envelope.patient_id:
        return ingest.build_dialogflow_response([{
            "message": (
                "We couldn't identify your account. Please contact us "
                "directly or ask your clinic for assistance."
            )
        }])

    try:
        result = await gateway.process_event(envelope)
        # Convert AgentResponses to Dialogflow format
        responses = [
            {"message": r.message}
            for r in (result or [])
            if hasattr(r, "message")
        ]
        if not responses:
            responses = [{"message": "Thank you, we've received your message."}]
        return ingest.build_dialogflow_response(responses)
    except Exception as exc:
        logger.error("Gateway processing error: %s", exc, exc_info=True)
        return ingest.build_dialogflow_response([{
            "message": "We're experiencing a temporary issue. Please try again shortly."
        }])


@router.post("/email-inbound")
async def email_inbound(request_body: dict[str, Any]):
    """
    SendGrid Inbound Parse webhook.

    Receives GP email replies, converts to EventEnvelopes, and processes
    through the Gateway. GP responses are routed to the Clinical Agent.
    """
    from medforce.gateway.ingest.email_ingest import EmailIngest
    from medforce.gateway.setup import get_gateway, get_identity_resolver

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    identity_resolver = get_identity_resolver()
    ingest = EmailIngest(identity_resolver=identity_resolver)

    try:
        envelope = await ingest.to_envelope(request_body)
    except Exception as exc:
        logger.error("Email ingest error: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse email: {exc}")

    if not envelope.patient_id:
        logger.warning(
            "Email inbound: could not resolve patient_id from %s",
            request_body.get("to", "unknown"),
        )
        return {"success": False, "error": "Could not resolve patient"}

    try:
        await gateway.process_event(envelope)
        return {
            "success": True,
            "patient_id": envelope.patient_id,
            "event_type": envelope.event_type.value,
        }
    except Exception as exc:
        logger.error("Gateway processing error: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


@router.post("/twilio-webhook")
async def twilio_webhook(request_body: dict[str, Any]):
    """
    Twilio incoming SMS/WhatsApp webhook.

    Receives SMS or WhatsApp messages from Twilio, converts to
    EventEnvelopes, and processes through the Gateway.

    Returns TwiML response (empty — Gateway handles async responses
    via the TwilioSMSDispatcher).
    """
    from medforce.gateway.ingest.twilio_ingest import TwilioSMSIngest
    from medforce.gateway.setup import get_gateway, get_identity_resolver

    gateway = get_gateway()
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    identity_resolver = get_identity_resolver()
    ingest = TwilioSMSIngest(identity_resolver=identity_resolver)

    try:
        envelope = await ingest.to_envelope(request_body)
    except Exception as exc:
        logger.error("Twilio ingest error: %s", exc, exc_info=True)
        # Return empty TwiML — don't send error to user's phone
        return {"twiml": "<Response></Response>"}

    if not envelope.patient_id:
        logger.warning(
            "Twilio inbound: unrecognised sender %s",
            request_body.get("From", "unknown"),
        )
        # Return empty TwiML — unknown senders get no response
        return {"twiml": "<Response></Response>"}

    try:
        await gateway.process_event(envelope)
        return {
            "success": True,
            "patient_id": envelope.patient_id,
            "twiml": "<Response></Response>",
        }
    except Exception as exc:
        logger.error("Gateway processing error: %s", exc, exc_info=True)
        return {"twiml": "<Response></Response>"}

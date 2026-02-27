"""
Patient Diary — Single source of truth for a patient's clinical journey.

One structured JSON document per patient, stored in GCS.  Every agent
reads the diary before acting and writes to it after acting.

Storage path: gs://{bucket}/patient_diaries/patient_{id}/diary.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("gateway.diary")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Phase(str, Enum):
    INTAKE = "intake"
    CLINICAL = "clinical"
    BOOKING = "booking"
    MONITORING = "monitoring"
    CLOSED = "closed"


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClinicalSubPhase(str, Enum):
    NOT_STARTED = "not_started"
    ANALYZING_REFERRAL = "analyzing_referral"
    ASKING_QUESTIONS = "asking_questions"
    COLLECTING_DOCUMENTS = "collecting_documents"
    SCORING_RISK = "scoring_risk"
    COMPLETE = "complete"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sub-models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DiaryHeader(BaseModel):
    patient_id: str
    current_phase: Phase = Phase.INTAKE
    risk_level: RiskLevel = RiskLevel.NONE
    created: datetime = Field(default_factory=_now)
    last_updated: datetime = Field(default_factory=_now)
    correlation_id: Optional[str] = None
    # P0: Phase transition recovery — tracks when the current phase was entered
    phase_entered_at: datetime = Field(default_factory=_now)


class IntakeSection(BaseModel):
    # Responder identification — who is filling this in?
    responder_type: Optional[str] = None  # "patient" or "helper"
    responder_name: Optional[str] = None  # helper name if not patient
    responder_relationship: Optional[str] = None  # e.g. "spouse", "carer"

    # Standard UK clinic patient demographics
    name: Optional[str] = None
    dob: Optional[str] = None
    nhs_number: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    next_of_kin: Optional[str] = None
    gp_practice: Optional[str] = None
    gp_name: Optional[str] = None
    contact_preference: Optional[str] = None  # "email", "sms", "phone", "websocket"
    consent_gp_contact: Optional[bool] = None
    referral_letter_ref: Optional[str] = None
    fields_collected: list[str] = Field(default_factory=list)
    fields_missing: list[str] = Field(default_factory=lambda: [
        "name", "dob", "nhs_number", "address", "phone",
        "email", "next_of_kin", "gp_practice", "gp_name",
        "contact_preference",
    ])
    intake_complete: bool = False

    # All fields that must be collected before Intake is done
    REQUIRED_FIELDS: ClassVar[list[str]] = [
        "name", "dob", "nhs_number", "phone", "gp_name",
        "contact_preference",
    ]

    def mark_field_collected(self, field: str, value: str) -> None:
        setattr(self, field, value)
        if field not in self.fields_collected:
            self.fields_collected.append(field)
        if field in self.fields_missing:
            self.fields_missing.remove(field)

    def get_missing_required(self) -> list[str]:
        return [f for f in self.REQUIRED_FIELDS if f not in self.fields_collected]

    def is_complete(self) -> bool:
        return len(self.get_missing_required()) == 0


class HelperEntry(BaseModel):
    id: str
    name: str
    relationship: str = ""  # spouse, child, friend, carer
    channel: str = "websocket"
    contact: str = ""  # phone or email
    permissions: list[str] = Field(default_factory=list)
    verified: bool = False
    added: datetime = Field(default_factory=_now)


class HelperRegistry(BaseModel):
    helpers: list[HelperEntry] = Field(default_factory=list)
    pending_verifications: list[str] = Field(default_factory=list)

    def add_helper(self, helper: HelperEntry) -> None:
        self.helpers.append(helper)
        if not helper.verified:
            self.pending_verifications.append(helper.id)

    def verify_helper(self, helper_id: str) -> bool:
        for h in self.helpers:
            if h.id == helper_id:
                h.verified = True
                if helper_id in self.pending_verifications:
                    self.pending_verifications.remove(helper_id)
                return True
        return False

    def get_helper(self, helper_id: str) -> HelperEntry | None:
        for h in self.helpers:
            if h.id == helper_id:
                return h
        return None

    def get_helper_by_contact(self, contact: str) -> HelperEntry | None:
        for h in self.helpers:
            if h.contact == contact:
                return h
        return None

    def get_helpers_with_permission(self, permission: str) -> list[HelperEntry]:
        return [
            h for h in self.helpers
            if h.verified and permission in h.permissions
        ]

    def remove_helper(self, helper_id: str) -> bool:
        for i, h in enumerate(self.helpers):
            if h.id == helper_id:
                self.helpers.pop(i)
                if helper_id in self.pending_verifications:
                    self.pending_verifications.remove(helper_id)
                return True
        return False


class GPQuery(BaseModel):
    query_id: str
    query_type: str = ""  # missing_lab_results, incomplete_meds, etc.
    query_text: str = ""
    sent: datetime = Field(default_factory=_now)
    reminder_sent: Optional[datetime] = None
    status: str = "pending"  # pending, responded, non_responsive
    response_received: Optional[datetime] = None
    attachments_received: list[str] = Field(default_factory=list)


class GPChannel(BaseModel):
    gp_name: Optional[str] = None
    gp_email: Optional[str] = None
    gp_practice: Optional[str] = None
    referral_id: Optional[str] = None
    queries: list[GPQuery] = Field(default_factory=list)
    last_contacted: Optional[datetime] = None

    def add_query(self, query: GPQuery) -> None:
        self.queries.append(query)
        self.last_contacted = query.sent

    def has_pending_queries(self) -> bool:
        return any(q.status == "pending" for q in self.queries)

    def get_pending_queries(self) -> list[GPQuery]:
        return [q for q in self.queries if q.status == "pending"]


class ClinicalQuestion(BaseModel):
    question: str
    answer: Optional[str] = None
    answered_by: Optional[str] = None  # "patient", "helper:Sarah"
    timestamp: Optional[datetime] = None
    is_followup: bool = False  # True if this is an adaptive follow-up to a plan question


class ClinicalDocument(BaseModel):
    type: str = ""  # lab_results, imaging, referral, nhs_screenshot
    source: str = ""  # "patient", "helper:Sarah", "gp:Dr.Patel"
    file_ref: str = ""
    processed: bool = False
    extracted_values: dict[str, Any] = Field(default_factory=dict)
    # P3: Content hash for deduplication (prevents same doc being processed twice)
    content_hash: Optional[str] = None


class ClinicalSection(BaseModel):
    chief_complaint: Optional[str] = None
    medical_history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    questions_asked: list[ClinicalQuestion] = Field(default_factory=list)
    documents: list[ClinicalDocument] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.NONE
    risk_reasoning: Optional[str] = None
    risk_method: Optional[str] = None  # "deterministic_rule: bilirubin > 5"
    sub_phase: ClinicalSubPhase = ClinicalSubPhase.NOT_STARTED
    sub_phase_history: list[str] = Field(default_factory=list)
    backward_loop_count: int = 0

    # Referral letter analysis — extracted by LLM
    referral_analysis: dict[str, Any] = Field(default_factory=dict)
    # Condition context — identified condition for personalized questioning
    condition_context: Optional[str] = None  # e.g. "cirrhosis", "MASH", "hepatitis"
    # Referral narrative — 200-300 word clinical prose preserving all referral detail
    referral_narrative: Optional[str] = None
    # Lifestyle factors relevant to condition
    lifestyle_factors: dict[str, Any] = Field(default_factory=dict)
    # Pain assessment
    pain_level: Optional[int] = None  # 0-10 scale
    pain_location: Optional[str] = None
    # Personalized questions generated by LLM, ranked by clinical importance
    generated_questions: list[str] = Field(default_factory=list)
    # Adaptive questioning — how many times questions have been regenerated (cap at 5)
    question_generation_count: int = 0
    # Track whether meds/allergies have been addressed (asked or answered)
    meds_addressed: bool = False
    allergies_addressed: bool = False
    # Adaptive follow-up: True after a plan question is asked, cleared after evaluation
    awaiting_followup: bool = False
    # Sequential document collection
    pending_document_requests: list[str] = Field(default_factory=list)
    documents_requested: list[str] = Field(default_factory=list)

    def advance_sub_phase(self, new_phase: ClinicalSubPhase) -> None:
        self.sub_phase = new_phase
        if new_phase.value not in self.sub_phase_history:
            self.sub_phase_history.append(new_phase.value)

    def has_document_hash(self, content_hash: str) -> bool:
        """P3: Check if a document with this content hash already exists."""
        return any(
            d.content_hash == content_hash
            for d in self.documents
            if d.content_hash
        )


class SlotOption(BaseModel):
    date: str
    time: str
    provider: str = ""
    hold_id: str = ""


class BookingSection(BaseModel):
    eligible_window: Optional[str] = None  # "48 hours (HIGH risk)"
    slots_offered: list[SlotOption] = Field(default_factory=list)
    slots_rejected: list[SlotOption] = Field(default_factory=list)  # previously rejected slots
    slot_selected: Optional[SlotOption] = None
    booked_by: Optional[str] = None  # "patient", "helper:Sarah"
    appointment_id: Optional[str] = None
    location: Optional[str] = None
    pre_appointment_instructions: list[str] = Field(default_factory=list)
    confirmed: bool = False
    rescheduled_from: list[dict] = Field(default_factory=list)


class MonitoringEntry(BaseModel):
    date: str
    type: str = ""  # heartbeat_14d, patient_message, lab_update
    action: str = ""
    detail: str = ""
    new_values: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)


class DeteriorationQuestion(BaseModel):
    """A question asked during deterioration assessment."""
    question: str
    answer: Optional[str] = None
    category: str = ""  # "description", "new_symptoms", "severity", "functional"


class DeteriorationAssessment(BaseModel):
    """Tracks an interactive deterioration assessment during monitoring."""
    active: bool = False
    detected_symptoms: list[str] = Field(default_factory=list)
    trigger_message: str = ""
    questions: list[DeteriorationQuestion] = Field(default_factory=list)
    assessment_complete: bool = False
    severity: Optional[str] = None  # "mild", "moderate", "severe", "emergency"
    recommendation: Optional[str] = None  # "continue_monitoring", "bring_forward", "urgent_referral", "emergency"
    reasoning: Optional[str] = None
    started: Optional[datetime] = None


class ScheduledQuestion(BaseModel):
    """A personalized monitoring question scheduled for delivery."""
    question: str
    day: int  # days after booking to send
    priority: int = 0  # higher = more important
    category: str = ""  # "symptom", "lifestyle", "medication", "labs"
    sent: bool = False
    response: Optional[str] = None


class CommunicationPlan(BaseModel):
    """Risk-stratified monitoring communication plan."""
    risk_level: str = "low"
    total_messages: int = 4
    check_in_days: list[int] = Field(default_factory=list)
    questions: list[ScheduledQuestion] = Field(default_factory=list)
    generated: bool = False


class MonitoringSection(BaseModel):
    monitoring_active: bool = False
    baseline: dict[str, Any] = Field(default_factory=dict)
    entries: list[MonitoringEntry] = Field(default_factory=list)
    alerts_fired: list[str] = Field(default_factory=list)
    next_scheduled_check: Optional[str] = None
    appointment_date: Optional[str] = None
    # Risk-stratified communication plan
    communication_plan: CommunicationPlan = Field(default_factory=CommunicationPlan)
    # Interactive deterioration assessment
    deterioration_assessment: DeteriorationAssessment = Field(default_factory=DeteriorationAssessment)

    MAX_ENTRIES: ClassVar[int] = 50

    def add_entry(self, entry: MonitoringEntry) -> None:
        self.entries.append(entry)
        # Cap at MAX_ENTRIES
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries = self.entries[-self.MAX_ENTRIES:]


class CrossPhaseState(BaseModel):
    """Tracks an active cross-phase conversation (e.g. clinical follow-up during booking)."""
    active: bool = False
    target_agent: str = ""          # "clinical" or "intake"
    pending_phase: str = ""         # phase to return to (e.g. "booking")
    follow_up_question: str | None = None
    awaiting_response: bool = False
    original_text: str = ""
    started: datetime | None = None


class ConversationEntry(BaseModel):
    timestamp: datetime = Field(default_factory=_now)
    direction: str = ""  # "AGENT→PATIENT", "PATIENT→AGENT", "SYSTEM", etc.
    channel: str = ""
    message: str = ""
    delivery_status: str = "delivered"  # delivered, pending_channel, failed
    chat_channel: str = "pre_consultation"  # "pre_consultation" or "monitoring"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Top-level Diary Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PatientDiary(BaseModel):
    header: DiaryHeader
    intake: IntakeSection = Field(default_factory=IntakeSection)
    helper_registry: HelperRegistry = Field(default_factory=HelperRegistry)
    gp_channel: GPChannel = Field(default_factory=GPChannel)
    clinical: ClinicalSection = Field(default_factory=ClinicalSection)
    booking: BookingSection = Field(default_factory=BookingSection)
    monitoring: MonitoringSection = Field(default_factory=MonitoringSection)
    conversation_log: list[ConversationEntry] = Field(default_factory=list)
    # Cross-phase content routing — audit trail of cross-phase data extractions
    cross_phase_extractions: list[dict[str, Any]] = Field(default_factory=list)
    # Cross-phase interactive state (e.g. clinical follow-up question during booking)
    cross_phase_state: CrossPhaseState = Field(default_factory=CrossPhaseState)

    MAX_CONVERSATION_LOG: ClassVar[int] = 100

    def add_conversation(self, entry: ConversationEntry) -> None:
        self.conversation_log.append(entry)
        if len(self.conversation_log) > self.MAX_CONVERSATION_LOG:
            self.conversation_log = self.conversation_log[-self.MAX_CONVERSATION_LOG:]

    def get_conversation(self, chat_channel: str | None = None) -> list[ConversationEntry]:
        """Return conversation entries, optionally filtered by chat_channel."""
        if chat_channel is None:
            return list(self.conversation_log)
        return [e for e in self.conversation_log if e.chat_channel == chat_channel]

    def touch(self) -> None:
        """Update last_updated timestamp."""
        self.header.last_updated = datetime.now(timezone.utc)

    @classmethod
    def create_new(cls, patient_id: str, correlation_id: str | None = None) -> PatientDiary:
        """Factory for a fresh diary in the intake phase."""
        return cls(
            header=DiaryHeader(
                patient_id=patient_id,
                correlation_id=correlation_id,
            ),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Diary Store — GCS persistence layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DiaryNotFoundError(Exception):
    pass


class DiaryConcurrencyError(Exception):
    """Raised when optimistic locking fails (diary was modified by another process)."""
    pass


class DiaryStore:
    """
    Persists PatientDiary objects to GCS via GCSBucketManager.

    Uses GCS generation-match for optimistic locking: when loading, we
    capture the blob generation number.  When saving, we require that the
    generation hasn't changed — if it has, another process wrote first
    and we raise DiaryConcurrencyError.
    """

    DIARY_PREFIX = "patient_diaries"

    def __init__(self, gcs_bucket_manager) -> None:
        self._gcs = gcs_bucket_manager

    def _blob_path(self, patient_id: str) -> str:
        return f"{self.DIARY_PREFIX}/patient_{patient_id}/diary.json"

    # HTTP timeout for individual GCS operations (seconds)
    GCS_TIMEOUT = 30

    def load(self, patient_id: str) -> tuple[PatientDiary, int]:
        """
        Load a diary from GCS.

        Returns (diary, generation) where generation is used for
        optimistic locking on save.
        """
        path = self._blob_path(patient_id)
        try:
            self._gcs._ensure_initialized()
            blob = self._gcs.bucket.blob(path)
            content = blob.download_as_text(timeout=self.GCS_TIMEOUT)
            generation = blob.generation or 0
            data = json.loads(content)
            diary = PatientDiary.model_validate(data)
            return diary, generation
        except DiaryNotFoundError:
            raise
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e).lower()
            if "notfound" in err_type.lower() or "not found" in err_msg or "notfound" in err_msg:
                raise DiaryNotFoundError(
                    f"No diary found for patient {patient_id}"
                ) from e
            raise

    def save(
        self, patient_id: str, diary: PatientDiary, generation: int | None = None
    ) -> int:
        """
        Save diary to GCS.  Returns the new generation number.

        If ``generation`` is provided, uses if_generation_match for
        optimistic locking.  Pass None to force-write (e.g. on create).
        """
        diary.touch()
        path = self._blob_path(patient_id)
        content = diary.model_dump_json(indent=2)

        try:
            self._gcs._ensure_initialized()
            blob = self._gcs.bucket.blob(path)

            if generation is not None:
                blob.upload_from_string(
                    content,
                    content_type="application/json",
                    if_generation_match=generation,
                    timeout=self.GCS_TIMEOUT,
                )
            else:
                blob.upload_from_string(
                    content,
                    content_type="application/json",
                    timeout=self.GCS_TIMEOUT,
                )

            blob.reload(timeout=self.GCS_TIMEOUT)
            return blob.generation or 0
        except Exception as e:
            if "conditionNotMet" in str(e) or "Precondition" in str(e):
                raise DiaryConcurrencyError(
                    f"Diary for {patient_id} was modified by another process"
                ) from e
            raise

    def create(self, patient_id: str, correlation_id: str | None = None) -> tuple[PatientDiary, int]:
        """Create a brand-new diary and persist it.  Returns (diary, generation)."""
        diary = PatientDiary.create_new(patient_id, correlation_id=correlation_id)
        gen = self.save(patient_id, diary, generation=None)
        logger.info("Created new diary for patient %s", patient_id)
        return diary, gen

    def exists(self, patient_id: str) -> bool:
        path = self._blob_path(patient_id)
        try:
            self._gcs._ensure_initialized()
            blob = self._gcs.bucket.blob(path)
            return blob.exists(timeout=self.GCS_TIMEOUT)
        except Exception:
            return False

    def delete(self, patient_id: str) -> bool:
        path = self._blob_path(patient_id)
        return self._gcs.delete_file(path)

    def list_all_patient_ids(self) -> list[str]:
        """List all patient IDs that have diaries."""
        try:
            files = self._gcs.list_files(self.DIARY_PREFIX)
            patient_ids = []
            for f in files:
                # Folder names look like "patient_PT-1234/"
                if f.startswith("patient_") and f.endswith("/"):
                    pid = f[len("patient_"):-1]
                    patient_ids.append(pid)
            return patient_ids
        except Exception as e:
            logger.error("Failed to list patient diaries: %s", e)
            return []

    def list_monitoring_patients(self) -> list[str]:
        """
        Find all patients in the monitoring phase with monitoring_active=True.

        Used by HeartbeatScheduler on startup to recover monitored patients.
        """
        all_ids = self.list_all_patient_ids()
        monitoring = []
        for pid in all_ids:
            try:
                diary, _ = self.load(pid)
                if (
                    diary.header.current_phase == Phase.MONITORING
                    and diary.monitoring.monitoring_active
                ):
                    monitoring.append(pid)
            except Exception:
                continue
        return monitoring

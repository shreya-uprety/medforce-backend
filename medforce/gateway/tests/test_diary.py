"""
Comprehensive tests for the Patient Diary model and DiaryStore.

Tests cover:
  - Diary creation with defaults
  - All sub-section models (Intake, Helper, GP, Clinical, Booking, Monitoring)
  - Field collection tracking
  - Helper registry (add, verify, remove, permissions)
  - GP query management
  - Clinical sub-phase advancement
  - Conversation log capping
  - Monitoring entry capping
  - JSON serialisation round-trip
  - DiaryStore with mocked GCS
  - Multiple patient scenarios
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import pytest

from medforce.gateway.diary import (
    BookingSection,
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSection,
    ClinicalSubPhase,
    ConversationEntry,
    DiaryHeader,
    DiaryNotFoundError,
    DiaryStore,
    GPChannel,
    GPQuery,
    HelperEntry,
    HelperRegistry,
    IntakeSection,
    MonitoringEntry,
    MonitoringSection,
    PatientDiary,
    Phase,
    RiskLevel,
    SlotOption,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Diary Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDiaryHeader:

    def test_defaults(self):
        h = DiaryHeader(patient_id="PT-1")
        assert h.current_phase == Phase.INTAKE
        assert h.risk_level == RiskLevel.NONE
        assert h.created is not None
        assert h.last_updated is not None

    def test_all_phases(self):
        for phase in Phase:
            h = DiaryHeader(patient_id="PT-1", current_phase=phase)
            assert h.current_phase == phase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Intake Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIntakeSection:

    def test_initial_missing_fields(self):
        intake = IntakeSection()
        missing = intake.get_missing_required()
        assert "name" in missing
        assert "dob" in missing
        assert "nhs_number" in missing
        assert "phone" in missing
        assert "gp_name" in missing

    def test_mark_field_collected(self):
        intake = IntakeSection()
        intake.mark_field_collected("name", "John Smith")
        assert intake.name == "John Smith"
        assert "name" in intake.fields_collected
        assert "name" not in intake.fields_missing

    def test_is_complete(self):
        intake = IntakeSection()
        assert not intake.is_complete()

        for field in IntakeSection.REQUIRED_FIELDS:
            intake.mark_field_collected(field, f"value_{field}")

        assert intake.is_complete()
        assert len(intake.get_missing_required()) == 0

    def test_partial_completion(self):
        intake = IntakeSection()
        intake.mark_field_collected("name", "John Smith")
        intake.mark_field_collected("dob", "1975-05-12")
        assert not intake.is_complete()
        missing = intake.get_missing_required()
        assert "nhs_number" in missing
        assert "phone" in missing
        assert "gp_name" in missing

    def test_duplicate_mark_is_idempotent(self):
        intake = IntakeSection()
        intake.mark_field_collected("name", "John")
        intake.mark_field_collected("name", "John Smith")  # update
        assert intake.name == "John Smith"
        assert intake.fields_collected.count("name") == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHelperRegistry:

    def _make_helper(self, **kwargs):
        defaults = {
            "id": "HELPER-001",
            "name": "Sarah Smith",
            "relationship": "spouse",
            "channel": "whatsapp",
            "contact": "+447700900462",
            "permissions": ["view_status", "upload_documents", "book_appointments"],
            "verified": False,
        }
        defaults.update(kwargs)
        return HelperEntry(**defaults)

    def test_add_helper(self):
        reg = HelperRegistry()
        helper = self._make_helper()
        reg.add_helper(helper)
        assert len(reg.helpers) == 1
        assert "HELPER-001" in reg.pending_verifications

    def test_verify_helper(self):
        reg = HelperRegistry()
        reg.add_helper(self._make_helper())
        assert reg.verify_helper("HELPER-001")
        assert reg.helpers[0].verified
        assert "HELPER-001" not in reg.pending_verifications

    def test_verify_nonexistent_helper(self):
        reg = HelperRegistry()
        assert not reg.verify_helper("HELPER-999")

    def test_get_helper_by_id(self):
        reg = HelperRegistry()
        reg.add_helper(self._make_helper())
        h = reg.get_helper("HELPER-001")
        assert h is not None
        assert h.name == "Sarah Smith"

    def test_get_helper_by_contact(self):
        reg = HelperRegistry()
        reg.add_helper(self._make_helper())
        h = reg.get_helper_by_contact("+447700900462")
        assert h is not None

    def test_get_helpers_with_permission(self):
        reg = HelperRegistry()
        sarah = self._make_helper(verified=True)
        reg.add_helper(sarah)
        friend = self._make_helper(
            id="HELPER-002",
            name="Emma",
            relationship="friend",
            contact="+447700900999",
            permissions=["view_status"],
            verified=True,
        )
        reg.add_helper(friend)

        # Both can view status
        viewers = reg.get_helpers_with_permission("view_status")
        assert len(viewers) == 2

        # Only Sarah can book
        bookers = reg.get_helpers_with_permission("book_appointments")
        assert len(bookers) == 1
        assert bookers[0].name == "Sarah Smith"

    def test_unverified_helper_excluded_from_permissions(self):
        reg = HelperRegistry()
        reg.add_helper(self._make_helper(verified=False))
        result = reg.get_helpers_with_permission("view_status")
        assert len(result) == 0  # not verified yet

    def test_remove_helper(self):
        reg = HelperRegistry()
        reg.add_helper(self._make_helper())
        assert reg.remove_helper("HELPER-001")
        assert len(reg.helpers) == 0

    def test_multiple_helpers(self):
        reg = HelperRegistry()
        reg.add_helper(self._make_helper(id="H1", name="Sarah", contact="+441"))
        reg.add_helper(self._make_helper(id="H2", name="Michael", contact="+442"))
        reg.add_helper(self._make_helper(id="H3", name="Emma", contact="+443"))
        assert len(reg.helpers) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GP Channel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPChannel:

    def test_add_query(self):
        gp = GPChannel(gp_name="Dr. Patel", gp_email="dr.patel@nhs.uk")
        q = GPQuery(query_id="GPQ-001", query_type="missing_lab_results")
        gp.add_query(q)
        assert len(gp.queries) == 1
        assert gp.has_pending_queries()

    def test_no_pending_queries(self):
        gp = GPChannel()
        assert not gp.has_pending_queries()

    def test_responded_query_not_pending(self):
        gp = GPChannel()
        q = GPQuery(query_id="GPQ-001", status="responded")
        gp.add_query(q)
        assert not gp.has_pending_queries()

    def test_multiple_queries(self):
        gp = GPChannel()
        gp.add_query(GPQuery(query_id="Q1", status="responded"))
        gp.add_query(GPQuery(query_id="Q2", status="pending"))
        pending = gp.get_pending_queries()
        assert len(pending) == 1
        assert pending[0].query_id == "Q2"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clinical Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClinicalSection:

    def test_sub_phase_advancement(self):
        cs = ClinicalSection()
        assert cs.sub_phase == ClinicalSubPhase.NOT_STARTED

        cs.advance_sub_phase(ClinicalSubPhase.ANALYZING_REFERRAL)
        assert cs.sub_phase == ClinicalSubPhase.ANALYZING_REFERRAL
        assert "analyzing_referral" in cs.sub_phase_history

        cs.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        assert cs.sub_phase == ClinicalSubPhase.ASKING_QUESTIONS
        assert len(cs.sub_phase_history) == 2

    def test_sub_phase_history_no_duplicates(self):
        cs = ClinicalSection()
        cs.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        cs.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        assert cs.sub_phase_history.count("asking_questions") == 1

    def test_clinical_questions(self):
        cs = ClinicalSection()
        cs.questions_asked.append(ClinicalQuestion(
            question="How many units of alcohol per week?",
            answer="About 20",
            answered_by="patient",
        ))
        assert len(cs.questions_asked) == 1

    def test_clinical_documents(self):
        cs = ClinicalSection()
        cs.documents.append(ClinicalDocument(
            type="lab_results",
            source="helper:Sarah",
            file_ref="gs://bucket/lab.jpg",
            processed=True,
            extracted_values={"ALT": 340, "bilirubin": 6.2},
        ))
        assert cs.documents[0].extracted_values["ALT"] == 340

    def test_backward_loop_counter(self):
        cs = ClinicalSection()
        assert cs.backward_loop_count == 0
        cs.backward_loop_count += 1
        assert cs.backward_loop_count == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Booking Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBookingSection:

    def test_slot_selection(self):
        bs = BookingSection()
        bs.slots_offered = [
            SlotOption(date="2026-02-18", time="10:00", provider="Dr. Williams"),
            SlotOption(date="2026-02-18", time="14:00", provider="Dr. Chen"),
        ]
        bs.slot_selected = bs.slots_offered[0]
        bs.confirmed = True
        bs.booked_by = "helper:Sarah"
        assert bs.confirmed
        assert bs.slot_selected.provider == "Dr. Williams"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monitoring Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMonitoringSection:

    def test_add_entry(self):
        ms = MonitoringSection(monitoring_active=True)
        ms.add_entry(MonitoringEntry(
            date="2026-03-03",
            type="heartbeat_14d",
            action="sent_followup_reminder",
        ))
        assert len(ms.entries) == 1

    def test_entry_capping(self):
        ms = MonitoringSection()
        for i in range(60):
            ms.add_entry(MonitoringEntry(date=f"2026-01-{i+1:02d}", type="test"))
        assert len(ms.entries) == 50  # capped at MAX_ENTRIES

    def test_baseline_storage(self):
        ms = MonitoringSection()
        ms.baseline = {"ALT": 340, "AST": 280, "bilirubin": 6.2, "platelets": 95}
        assert ms.baseline["ALT"] == 340


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Diary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientDiary:

    def test_create_new(self):
        diary = PatientDiary.create_new("PT-1234", correlation_id="journey-001")
        assert diary.header.patient_id == "PT-1234"
        assert diary.header.current_phase == Phase.INTAKE
        assert diary.header.correlation_id == "journey-001"

    def test_conversation_log_capping(self):
        diary = PatientDiary.create_new("PT-CAP")
        for i in range(120):
            diary.add_conversation(ConversationEntry(
                direction="AGENT→PATIENT",
                message=f"Message {i}",
            ))
        assert len(diary.conversation_log) == 100  # capped

    def test_touch_updates_timestamp(self):
        diary = PatientDiary.create_new("PT-TOUCH")
        old_ts = diary.header.last_updated
        import time
        time.sleep(0.01)
        diary.touch()
        assert diary.header.last_updated >= old_ts

    def test_full_json_round_trip(self):
        """Build a realistic diary and serialise/deserialise."""
        diary = PatientDiary.create_new("PT-FULL")

        # Fill intake
        diary.intake.mark_field_collected("name", "John Smith")
        diary.intake.mark_field_collected("dob", "1975-05-12")
        diary.intake.mark_field_collected("nhs_number", "123-456-7890")
        diary.intake.mark_field_collected("phone", "+447700900461")
        diary.intake.mark_field_collected("gp_name", "Dr. Patel")

        # Add a helper
        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-001",
            name="Sarah Smith",
            relationship="spouse",
            contact="+447700900462",
            permissions=["view_status", "upload_documents", "book_appointments"],
            verified=True,
        ))

        # GP channel
        diary.gp_channel.gp_name = "Dr. Patel"
        diary.gp_channel.gp_email = "dr.patel@nhs.uk"

        # Clinical
        diary.clinical.chief_complaint = "RUQ pain, fatigue"
        diary.clinical.risk_level = RiskLevel.HIGH
        diary.clinical.advance_sub_phase(ClinicalSubPhase.COMPLETE)

        # Booking
        diary.booking.slot_selected = SlotOption(
            date="2026-02-18", time="10:00", provider="Dr. Williams"
        )
        diary.booking.confirmed = True

        # Monitoring
        diary.monitoring.monitoring_active = True
        diary.monitoring.baseline = {"ALT": 340, "bilirubin": 6.2}

        # Conversation
        diary.add_conversation(ConversationEntry(
            direction="AGENT→PATIENT",
            channel="whatsapp",
            message="Hi John, confirm your DOB?",
        ))

        # Round trip
        json_str = diary.model_dump_json(indent=2)
        restored = PatientDiary.model_validate_json(json_str)

        assert restored.header.patient_id == "PT-FULL"
        assert restored.intake.name == "John Smith"
        assert restored.helper_registry.helpers[0].name == "Sarah Smith"
        assert restored.clinical.risk_level == RiskLevel.HIGH
        assert restored.booking.confirmed
        assert restored.monitoring.baseline["ALT"] == 340
        assert len(restored.conversation_log) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenario Diaries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientScenarios:

    def test_scenario_1_mary_jones_low_risk(self):
        """Mary Jones, 55, solo patient, low risk routine screening."""
        diary = PatientDiary.create_new("PT-MARY")
        diary.intake.mark_field_collected("name", "Mary Jones")
        diary.intake.mark_field_collected("dob", "1970-03-15")
        diary.intake.mark_field_collected("nhs_number", "987-654-3210")
        diary.intake.mark_field_collected("phone", "+447700111222")
        diary.intake.mark_field_collected("gp_name", "Dr. Williams")
        diary.intake.mark_field_collected("contact_preference", "phone")
        assert diary.intake.is_complete()

        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.chief_complaint = "Mildly elevated GGT, routine screening"
        diary.clinical.risk_level = RiskLevel.LOW
        assert diary.clinical.risk_level == RiskLevel.LOW

    def test_scenario_2_david_clarke_high_risk_with_spouse(self):
        """David Clarke, 62, urgent, wife Linda as helper."""
        diary = PatientDiary.create_new("PT-DAVID")
        diary.intake.mark_field_collected("name", "David Clarke")
        diary.intake.mark_field_collected("dob", "1963-08-22")
        diary.intake.mark_field_collected("nhs_number", "555-123-4567")
        diary.intake.mark_field_collected("phone", "+447700333444")
        diary.intake.mark_field_collected("gp_name", "Dr. Patel")

        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-LINDA",
            name="Linda Clarke",
            relationship="spouse",
            contact="+447700555666",
            permissions=["view_status", "upload_documents", "answer_questions",
                         "book_appointments", "receive_alerts"],
            verified=True,
        ))

        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.chief_complaint = "Jaundice, ascites, confusion"
        diary.clinical.red_flags = ["jaundice", "ascites", "confusion"]
        diary.clinical.documents.append(ClinicalDocument(
            type="lab_results",
            source="gp:Dr.Patel",
            extracted_values={"ALT": 580, "bilirubin": 8.3, "platelets": 62},
            processed=True,
        ))
        diary.clinical.risk_level = RiskLevel.HIGH
        diary.clinical.risk_method = "deterministic_rule: bilirubin > 5"

        assert diary.clinical.risk_level == RiskLevel.HIGH
        assert len(diary.helper_registry.helpers) == 1
        bookers = diary.helper_registry.get_helpers_with_permission("book_appointments")
        assert len(bookers) == 1

    def test_scenario_4_robert_taylor_backward_loop(self):
        """Robert Taylor, 71, backward loop for missing medication list."""
        diary = PatientDiary.create_new("PT-ROBERT")
        diary.intake.mark_field_collected("name", "Robert Taylor")
        diary.intake.mark_field_collected("dob", "1954-11-03")
        diary.intake.mark_field_collected("nhs_number", "111-222-3333")
        diary.intake.mark_field_collected("phone", "+447700777888")
        diary.intake.mark_field_collected("gp_name", "Dr. Singh")
        diary.intake.mark_field_collected("contact_preference", "phone")
        diary.intake.intake_complete = True

        # All required fields collected — intake was complete
        assert diary.intake.is_complete()

        # Clinical discovers missing meds
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.advance_sub_phase(ClinicalSubPhase.ASKING_QUESTIONS)
        diary.clinical.backward_loop_count = 1

        # Backward loop: phase goes back to intake, meds tracked in fields_missing
        diary.header.current_phase = Phase.INTAKE
        diary.intake.intake_complete = False
        diary.intake.fields_missing.append("current_medication_list")

        assert diary.clinical.backward_loop_count == 1
        assert diary.header.current_phase == Phase.INTAKE
        assert "current_medication_list" in diary.intake.fields_missing
        # Note: is_complete() checks REQUIRED_FIELDS only (core demographics).
        # Backward loop requests are tracked via fields_missing and the
        # NEEDS_INTAKE_DATA event payload, not by modifying REQUIRED_FIELDS.
        # The Intake Agent reads fields_missing to know what to ask for.

    def test_scenario_5_helen_morris_deterioration(self):
        """Helen Morris, 45, monitoring phase, deterioration detected."""
        diary = PatientDiary.create_new("PT-HELEN")
        diary.header.current_phase = Phase.MONITORING

        diary.monitoring.monitoring_active = True
        diary.monitoring.baseline = {"ALT": 180, "bilirubin": 2.1}
        diary.monitoring.appointment_date = "2025-11-19"

        # Patient reports jaundice 3 months later
        diary.monitoring.add_entry(MonitoringEntry(
            date="2026-02-19",
            type="patient_message",
            action="deterioration_detected",
            detail="Patient reports yellow skin. Jaundice keyword detected.",
        ))
        diary.monitoring.alerts_fired.append("DETERIORATION_ALERT")

        assert diary.monitoring.monitoring_active
        assert len(diary.monitoring.alerts_fired) == 1

    def test_scenario_6_multi_helper_permissions(self):
        """Tom Hughes, 28, mother full access, girlfriend view+upload only."""
        diary = PatientDiary.create_new("PT-TOM")

        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-CAROL",
            name="Carol Hughes",
            relationship="parent",
            contact="+447700100100",
            permissions=["view_status", "upload_documents", "answer_questions",
                         "book_appointments", "receive_alerts"],
            verified=True,
        ))
        diary.helper_registry.add_helper(HelperEntry(
            id="HELPER-EMMA",
            name="Emma",
            relationship="friend",
            contact="+447700200200",
            permissions=["view_status", "upload_documents"],
            verified=True,
        ))

        bookers = diary.helper_registry.get_helpers_with_permission("book_appointments")
        assert len(bookers) == 1
        assert bookers[0].name == "Carol Hughes"

        uploaders = diary.helper_registry.get_helpers_with_permission("upload_documents")
        assert len(uploaders) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DiaryStore (Mocked GCS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDiaryStore:

    def _make_mock_gcs(self):
        """Create a mock GCSBucketManager with in-memory storage."""
        gcs = MagicMock()
        gcs._storage = {}  # our in-memory blob store

        def _ensure_initialized():
            pass
        gcs._ensure_initialized = _ensure_initialized

        # Mock bucket.blob() to return a mock blob
        mock_bucket = MagicMock()
        gcs.bucket = mock_bucket

        class MockBlob:
            def __init__(self, path):
                self.path = path
                self.generation = 1

            def download_as_text(self, timeout=None):
                if self.path not in gcs._storage:
                    raise Exception("NotFound")
                return gcs._storage[self.path]

            def upload_from_string(self, content, content_type=None, if_generation_match=None, timeout=None):
                if if_generation_match is not None:
                    existing_gen = gcs._generations.get(self.path, 0)
                    if existing_gen != if_generation_match:
                        raise Exception("conditionNotMet: generation mismatch")
                gcs._storage[self.path] = content
                gcs._generations[self.path] = gcs._generations.get(self.path, 0) + 1
                self.generation = gcs._generations[self.path]

            def reload(self, timeout=None):
                self.generation = gcs._generations.get(self.path, 1)

            def exists(self, timeout=None):
                return self.path in gcs._storage

        gcs._generations = {}
        mock_bucket.blob = lambda path: MockBlob(path)

        def delete_file(path):
            if path in gcs._storage:
                del gcs._storage[path]
                return True
            return False
        gcs.delete_file = delete_file

        def list_files(prefix):
            result = set()
            for key in gcs._storage:
                if key.startswith(prefix):
                    # Return immediate children (folder names)
                    rest = key[len(prefix):]
                    if rest.startswith("/"):
                        rest = rest[1:]
                    parts = rest.split("/")
                    if len(parts) > 1:
                        result.add(parts[0] + "/")
                    else:
                        result.add(parts[0])
            return list(result)
        gcs.list_files = list_files

        return gcs

    def test_create_and_load(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        diary, gen = store.create("PT-1234")
        assert diary.header.patient_id == "PT-1234"
        assert diary.header.current_phase == Phase.INTAKE

        loaded, loaded_gen = store.load("PT-1234")
        assert loaded.header.patient_id == "PT-1234"

    def test_load_nonexistent_raises(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        with pytest.raises(DiaryNotFoundError):
            store.load("PT-NONEXISTENT")

    def test_save_and_load_preserves_data(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        diary, gen = store.create("PT-SAVE")
        diary.intake.mark_field_collected("name", "Test Patient")
        diary.header.current_phase = Phase.CLINICAL
        diary.clinical.risk_level = RiskLevel.HIGH

        new_gen = store.save("PT-SAVE", diary, generation=None)

        loaded, _ = store.load("PT-SAVE")
        assert loaded.intake.name == "Test Patient"
        assert loaded.header.current_phase == Phase.CLINICAL
        assert loaded.clinical.risk_level == RiskLevel.HIGH

    def test_exists(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        assert not store.exists("PT-NEW")
        store.create("PT-NEW")
        assert store.exists("PT-NEW")

    def test_delete(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        store.create("PT-DEL")
        assert store.exists("PT-DEL")
        store.delete("PT-DEL")
        assert not store.exists("PT-DEL")

    def test_list_all_patient_ids(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        store.create("PT-A")
        store.create("PT-B")
        store.create("PT-C")

        ids = store.list_all_patient_ids()
        assert "PT-A" in ids
        assert "PT-B" in ids
        assert "PT-C" in ids

    def test_list_monitoring_patients(self):
        gcs = self._make_mock_gcs()
        store = DiaryStore(gcs)

        # Create 3 patients, only 1 in monitoring
        d1, _ = store.create("PT-MON1")
        d1.header.current_phase = Phase.MONITORING
        d1.monitoring.monitoring_active = True
        store.save("PT-MON1", d1)

        d2, _ = store.create("PT-MON2")
        d2.header.current_phase = Phase.CLINICAL
        store.save("PT-MON2", d2)

        d3, _ = store.create("PT-MON3")
        d3.header.current_phase = Phase.MONITORING
        d3.monitoring.monitoring_active = False  # inactive
        store.save("PT-MON3", d3)

        monitoring = store.list_monitoring_patients()
        assert "PT-MON1" in monitoring
        assert "PT-MON2" not in monitoring
        assert "PT-MON3" not in monitoring

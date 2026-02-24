"""
Comprehensive tests for the Identity Resolution System.

Tests cover:
  - Patient lookup by phone, email, NHS number
  - Helper lookup across multiple patient diaries
  - GP lookup by email
  - Unknown sender handling
  - Ambiguous identity (helper for multiple patients)
  - UK phone number normalisation (07xxx → +447xxx)
  - Index rebuild from diaries
  - Incremental index update
  - Patient removal from index
  - Multiple patient scenarios from architecture doc
"""

import pytest

from medforce.gateway.diary import (
    GPChannel,
    HelperEntry,
    HelperRegistry,
    IntakeSection,
    PatientDiary,
)
from medforce.gateway.handlers.identity_resolver import (
    AmbiguousIdentity,
    IdentityRecord,
    IdentityResolver,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_diary(
    patient_id: str,
    name: str = "",
    phone: str = "",
    email: str = "",
    nhs_number: str = "",
    helpers: list[HelperEntry] | None = None,
    gp_name: str = "",
    gp_email: str = "",
) -> PatientDiary:
    diary = PatientDiary.create_new(patient_id)
    if name:
        diary.intake.mark_field_collected("name", name)
    if phone:
        diary.intake.mark_field_collected("phone", phone)
    if email:
        diary.intake.mark_field_collected("email", email)
    if nhs_number:
        diary.intake.mark_field_collected("nhs_number", nhs_number)
    if helpers:
        for h in helpers:
            diary.helper_registry.add_helper(h)
    if gp_name:
        diary.gp_channel.gp_name = gp_name
    if gp_email:
        diary.gp_channel.gp_email = gp_email
    return diary


def _sarah_helper(**overrides) -> HelperEntry:
    defaults = dict(
        id="HELPER-001",
        name="Sarah Smith",
        relationship="spouse",
        channel="whatsapp",
        contact="+447700900462",
        permissions=["view_status", "upload_documents", "book_appointments"],
        verified=True,
    )
    defaults.update(overrides)
    return HelperEntry(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Basic Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBasicResolution:

    def test_resolve_patient_by_phone(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1234", name="John", phone="+447700900461")
        resolver.rebuild_from_diaries({"PT-1234": diary})

        result = resolver.resolve("+447700900461")
        assert isinstance(result, IdentityRecord)
        assert result.patient_id == "PT-1234"
        assert result.sender_role == "patient"

    def test_resolve_patient_by_email(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1234", email="john@email.com")
        resolver.rebuild_from_diaries({"PT-1234": diary})

        result = resolver.resolve("john@email.com")
        assert isinstance(result, IdentityRecord)
        assert result.sender_role == "patient"

    def test_resolve_patient_by_nhs_number(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1234", nhs_number="123-456-7890")
        resolver.rebuild_from_diaries({"PT-1234": diary})

        result = resolver.resolve("123-456-7890")
        assert isinstance(result, IdentityRecord)
        assert result.sender_role == "patient"

    def test_resolve_unknown_returns_none(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1234", phone="+447700900461")
        resolver.rebuild_from_diaries({"PT-1234": diary})

        result = resolver.resolve("+447700999888")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHelperResolution:

    def test_resolve_helper_by_contact(self):
        resolver = IdentityResolver()
        diary = _make_diary(
            "PT-1234",
            phone="+447700900461",
            helpers=[_sarah_helper()],
        )
        resolver.rebuild_from_diaries({"PT-1234": diary})

        result = resolver.resolve("+447700900462")
        assert isinstance(result, IdentityRecord)
        assert result.sender_role == "helper"
        assert result.sender_id == "HELPER-001"
        assert result.name == "Sarah Smith"
        assert result.patient_id == "PT-1234"
        assert "book_appointments" in result.permissions

    def test_helper_permissions_preserved(self):
        resolver = IdentityResolver()
        friend = HelperEntry(
            id="HELPER-EMMA",
            name="Emma",
            relationship="friend",
            contact="+447700999111",
            permissions=["view_status"],
            verified=True,
        )
        diary = _make_diary("PT-TOM", helpers=[friend])
        resolver.rebuild_from_diaries({"PT-TOM": diary})

        result = resolver.resolve("+447700999111")
        assert result.permissions == ["view_status"]
        assert result.relationship == "friend"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GP Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGPResolution:

    def test_resolve_gp_by_email(self):
        resolver = IdentityResolver()
        diary = _make_diary(
            "PT-1234",
            gp_name="Dr. Patel",
            gp_email="dr.patel@greenfields.nhs.uk",
        )
        resolver.rebuild_from_diaries({"PT-1234": diary})

        result = resolver.resolve("dr.patel@greenfields.nhs.uk")
        assert isinstance(result, IdentityRecord)
        assert result.sender_role == "gp"
        assert result.patient_id == "PT-1234"
        assert "Dr. Patel" in result.name

    def test_gp_has_correct_permissions(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1", gp_email="dr.chen@nhs.uk", gp_name="Dr. Chen")
        resolver.rebuild_from_diaries({"PT-1": diary})

        result = resolver.resolve("dr.chen@nhs.uk")
        assert "upload_documents" in result.permissions
        assert "respond_to_queries" in result.permissions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phone Number Normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPhoneNormalisation:

    def test_uk_mobile_07_to_plus44(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-UK", phone="07700900461")
        resolver.rebuild_from_diaries({"PT-UK": diary})

        # Lookup with international format
        result = resolver.resolve("+447700900461")
        assert result is not None
        assert result.patient_id == "PT-UK"

    def test_plus44_to_07_lookup(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-UK2", phone="+447700900999")
        resolver.rebuild_from_diaries({"PT-UK2": diary})

        # Lookup with local format
        result = resolver.resolve("07700900999")
        assert result is not None
        assert result.patient_id == "PT-UK2"

    def test_spaces_and_dashes_stripped(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-FMT", phone="+44 7700 900 123")
        resolver.rebuild_from_diaries({"PT-FMT": diary})

        result = resolver.resolve("+447700900123")
        assert result is not None

    def test_case_insensitive_email(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-CASE", email="John.Smith@Email.COM")
        resolver.rebuild_from_diaries({"PT-CASE": diary})

        result = resolver.resolve("john.smith@email.com")
        assert result is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ambiguous Identity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAmbiguousIdentity:

    def test_helper_for_multiple_patients(self):
        """Sarah is helper for both John and Emma."""
        resolver = IdentityResolver()

        diary_john = _make_diary(
            "PT-JOHN",
            helpers=[_sarah_helper(contact="+447700900462")],
        )
        diary_emma = _make_diary(
            "PT-EMMA",
            helpers=[_sarah_helper(
                id="HELPER-002",
                contact="+447700900462",  # same phone
            )],
        )
        resolver.rebuild_from_diaries({
            "PT-JOHN": diary_john,
            "PT-EMMA": diary_emma,
        })

        result = resolver.resolve("+447700900462")
        assert isinstance(result, AmbiguousIdentity)
        assert len(result.records) == 2
        patient_ids = {r.patient_id for r in result.records}
        assert patient_ids == {"PT-JOHN", "PT-EMMA"}

    def test_resolve_for_specific_patient(self):
        """When ambiguous, resolve_for_patient narrows to one."""
        resolver = IdentityResolver()

        diary_john = _make_diary(
            "PT-JOHN",
            helpers=[_sarah_helper(contact="+447700900462")],
        )
        diary_emma = _make_diary(
            "PT-EMMA",
            helpers=[_sarah_helper(id="HELPER-002", contact="+447700900462")],
        )
        resolver.rebuild_from_diaries({
            "PT-JOHN": diary_john,
            "PT-EMMA": diary_emma,
        })

        result = resolver.resolve_for_patient("+447700900462", "PT-JOHN")
        assert isinstance(result, IdentityRecord)
        assert result.patient_id == "PT-JOHN"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Index Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIndexManagement:

    def test_rebuild_returns_count(self):
        resolver = IdentityResolver()
        diary = _make_diary(
            "PT-1",
            name="John",
            phone="+447700900461",
            email="john@email.com",
            nhs_number="123-456-7890",
            helpers=[_sarah_helper()],
            gp_email="dr.patel@nhs.uk",
            gp_name="Dr. Patel",
        )
        count = resolver.rebuild_from_diaries({"PT-1": diary})
        # 3 patient contacts + 1 helper + 1 GP = 5
        assert count == 5

    def test_index_size_and_unique_contacts(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1", phone="+441111", email="a@b.com")
        resolver.rebuild_from_diaries({"PT-1": diary})
        assert resolver.unique_contacts == 2
        assert resolver.index_size == 2

    def test_incremental_update(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1", phone="+441111")
        resolver.rebuild_from_diaries({"PT-1": diary})

        # Add a helper
        diary.helper_registry.add_helper(_sarah_helper(contact="+449999"))
        resolver.update_for_patient("PT-1", diary)

        # Helper should now be findable
        result = resolver.resolve("+449999")
        assert result is not None
        assert result.sender_role == "helper"

    def test_update_replaces_old_contacts(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1", phone="+441111")
        resolver.rebuild_from_diaries({"PT-1": diary})

        # Change phone number
        diary.intake.mark_field_collected("phone", "+442222")
        resolver.update_for_patient("PT-1", diary)

        # Old number should no longer resolve
        assert resolver.resolve("+441111") is None
        # New number should resolve
        assert resolver.resolve("+442222") is not None

    def test_rebuild_clears_old_index(self):
        resolver = IdentityResolver()
        diary1 = _make_diary("PT-OLD", phone="+440000")
        resolver.rebuild_from_diaries({"PT-OLD": diary1})

        # Rebuild with different data
        diary2 = _make_diary("PT-NEW", phone="+449999")
        resolver.rebuild_from_diaries({"PT-NEW": diary2})

        assert resolver.resolve("+440000") is None
        assert resolver.resolve("+449999") is not None

    def test_no_duplicate_entries(self):
        resolver = IdentityResolver()
        diary = _make_diary("PT-1", phone="+441111")
        resolver.rebuild_from_diaries({"PT-1": diary})
        # Rebuild again with same data
        resolver.rebuild_from_diaries({"PT-1": diary})
        result = resolver.resolve("+441111")
        assert isinstance(result, IdentityRecord)  # not ambiguous


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Patient Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientScenarios:

    def _build_full_index(self) -> IdentityResolver:
        """Build an index with multiple patients, helpers, and GPs."""
        resolver = IdentityResolver()

        diaries = {
            # Scenario 1: Mary Jones, solo, no helpers
            "PT-MARY": _make_diary(
                "PT-MARY", name="Mary Jones",
                phone="+447700111111", email="mary@email.com",
                gp_name="Dr. Williams", gp_email="dr.williams@oakfield.nhs.uk",
            ),
            # Scenario 2: David Clarke + wife Linda
            "PT-DAVID": _make_diary(
                "PT-DAVID", name="David Clarke",
                phone="+447700222222",
                helpers=[HelperEntry(
                    id="HELPER-LINDA", name="Linda Clarke",
                    relationship="spouse", contact="+447700222333",
                    permissions=["view_status", "book_appointments", "receive_alerts"],
                    verified=True,
                )],
                gp_name="Dr. Patel", gp_email="dr.patel@greenfields.nhs.uk",
            ),
            # Scenario 4: Robert Taylor + son James
            "PT-ROBERT": _make_diary(
                "PT-ROBERT", name="Robert Taylor",
                phone="+447700444444",
                helpers=[HelperEntry(
                    id="HELPER-JAMES", name="James Taylor",
                    relationship="child", contact="+447700444555",
                    permissions=["view_status", "upload_documents"],
                    verified=True,
                )],
                gp_name="Dr. Singh", gp_email="dr.singh@millroad.nhs.uk",
            ),
            # Scenario 5: Helen Morris + husband Peter
            "PT-HELEN": _make_diary(
                "PT-HELEN", name="Helen Morris",
                phone="+447700555555",
                helpers=[HelperEntry(
                    id="HELPER-PETER", name="Peter Morris",
                    relationship="spouse", contact="+447700555666",
                    permissions=["view_status", "upload_documents",
                                 "book_appointments", "receive_alerts"],
                    verified=True,
                )],
                gp_name="Dr. Brown", gp_email="dr.brown@nhs.uk",
            ),
        }

        resolver.rebuild_from_diaries(diaries)
        return resolver

    def test_identify_patient_mary(self):
        resolver = self._build_full_index()
        result = resolver.resolve("+447700111111")
        assert result.patient_id == "PT-MARY"
        assert result.sender_role == "patient"

    def test_identify_helper_linda(self):
        resolver = self._build_full_index()
        result = resolver.resolve("+447700222333")
        assert result.patient_id == "PT-DAVID"
        assert result.sender_role == "helper"
        assert result.name == "Linda Clarke"
        assert "book_appointments" in result.permissions

    def test_identify_gp_dr_patel(self):
        resolver = self._build_full_index()
        result = resolver.resolve("dr.patel@greenfields.nhs.uk")
        assert result.sender_role == "gp"
        assert result.patient_id == "PT-DAVID"

    def test_identify_helper_james_limited_permissions(self):
        resolver = self._build_full_index()
        result = resolver.resolve("+447700444555")
        assert result.sender_role == "helper"
        assert result.name == "James Taylor"
        assert "upload_documents" in result.permissions
        assert "book_appointments" not in result.permissions

    def test_unknown_number(self):
        """Scenario 8: random unknown number."""
        resolver = self._build_full_index()
        result = resolver.resolve("+447700999888")
        assert result is None

    def test_mary_email_lookup(self):
        resolver = self._build_full_index()
        result = resolver.resolve("mary@email.com")
        assert result.patient_id == "PT-MARY"

    def test_gp_dr_brown_for_helen(self):
        resolver = self._build_full_index()
        result = resolver.resolve("dr.brown@nhs.uk")
        assert result.patient_id == "PT-HELEN"
        assert result.sender_role == "gp"

    def test_scenario_6_multi_helper_tom(self):
        """Tom Hughes with mother Carol (full) and girlfriend Emma (limited)."""
        resolver = IdentityResolver()
        diary = _make_diary(
            "PT-TOM", name="Tom Hughes",
            phone="+447700600600",
            helpers=[
                HelperEntry(
                    id="HELPER-CAROL", name="Carol Hughes",
                    relationship="parent", contact="+447700600700",
                    permissions=["view_status", "upload_documents",
                                 "book_appointments", "receive_alerts"],
                    verified=True,
                ),
                HelperEntry(
                    id="HELPER-EMMA", name="Emma",
                    relationship="friend", contact="+447700600800",
                    permissions=["view_status", "upload_documents"],
                    verified=True,
                ),
            ],
        )
        resolver.rebuild_from_diaries({"PT-TOM": diary})

        carol = resolver.resolve("+447700600700")
        assert carol.name == "Carol Hughes"
        assert "book_appointments" in carol.permissions

        emma = resolver.resolve("+447700600800")
        assert emma.name == "Emma"
        assert "book_appointments" not in emma.permissions

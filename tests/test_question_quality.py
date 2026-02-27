"""
Question & Document Request Quality Tests.

Tests that the clinical agent generates relevant, patient-specific questions
and document requests for known test patients (PT-TEST-001 Arthur Shelby,
PT-TEST-002 Clara Higgins).

Can be run in two modes:
  1. With LLM:  python tests/test_question_quality.py
  2. Without:   python -m pytest tests/test_question_quality.py -v
"""

import asyncio
import json
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.diary import (
    ClinicalSubPhase,
    PatientDiary,
    Phase,
    RiskLevel,
)


# ── Test Patient Factories ──


def _make_arthur_shelby_diary() -> PatientDiary:
    """PT-TEST-001 Arthur Shelby: HCC, AFP 485, 2WW, HIGH risk."""
    diary = PatientDiary.create_new("PT-TEST-001")
    diary.header.current_phase = Phase.CLINICAL
    diary.header.risk_level = RiskLevel.HIGH
    diary.clinical.sub_phase = ClinicalSubPhase.ASKING_QUESTIONS

    diary.intake.name = "Arthur Shelby"
    diary.intake.phone = "07700900100"
    diary.intake.nhs_number = "1234500001"
    diary.intake.dob = "1962-08-14"
    diary.intake.gp_name = "Dr. Bennett"

    diary.clinical.chief_complaint = "Suspected hepatocellular carcinoma (HCC)"
    diary.clinical.condition_context = "HCC / hepatocellular carcinoma"
    diary.clinical.medical_history = [
        "Alcohol-related liver disease",
        "Compensated cirrhosis (diagnosed 2022)",
        "Type 2 diabetes",
        "Hypertension",
    ]
    diary.clinical.current_medications = [
        "Propranolol 40mg BD",
        "Spironolactone 100mg OD",
        "Metformin 500mg BD",
    ]
    diary.clinical.allergies = ["Penicillin — rash"]
    diary.clinical.red_flags = ["weight loss", "right upper quadrant pain", "2-week wait"]

    diary.clinical.referral_analysis = {
        "chief_complaint": "Suspected hepatocellular carcinoma (HCC)",
        "condition_context": "HCC / hepatocellular carcinoma",
        "medical_history": [
            "Alcohol-related liver disease",
            "Compensated cirrhosis",
            "Type 2 diabetes",
        ],
        "current_medications": ["Propranolol 40mg BD", "Spironolactone 100mg OD"],
        "allergies": ["Penicillin — rash"],
        "red_flags": ["weight loss", "right upper quadrant pain"],
        "lab_values": {
            "AFP": "485 kU/L",
            "ALT": "78 U/L",
            "Bilirubin": "28 µmol/L",
            "Albumin": "32 g/L",
            "Platelets": "95 × 10⁹/L",
        },
        "key_findings": (
            "6.5 kg unintentional weight loss over 3 months. "
            "Ultrasound: 4.2 cm hypoechoic lesion in right lobe of liver. "
            "AFP markedly elevated at 485 kU/L. 2-week wait referral."
        ),
    }

    diary.clinical.referral_narrative = (
        "Arthur Shelby is a 63-year-old gentleman with a background of "
        "alcohol-related liver disease and compensated cirrhosis (diagnosed 2022), "
        "referred urgently under the 2-week wait pathway with suspected "
        "hepatocellular carcinoma. He presents with a 3-month history of "
        "6.5 kg unintentional weight loss and right upper quadrant pain. "
        "Ultrasound demonstrates a 4.2 cm hypoechoic lesion in the right lobe "
        "of the liver. AFP is markedly elevated at 485 kU/L. ALT 78 U/L, "
        "bilirubin 28 µmol/L, albumin 32 g/L, platelets 95 × 10⁹/L. "
        "He also has type 2 diabetes managed with metformin and hypertension. "
        "He is on propranolol 40mg BD and spironolactone 100mg OD. "
        "Allergic to penicillin (rash). Please assess urgently."
    )

    return diary


def _make_clara_higgins_diary() -> PatientDiary:
    """PT-TEST-002 Clara Higgins: Asymptomatic Hep C, Genotype 3a, LOW risk."""
    diary = PatientDiary.create_new("PT-TEST-002")
    diary.header.current_phase = Phase.CLINICAL
    diary.header.risk_level = RiskLevel.LOW
    diary.clinical.sub_phase = ClinicalSubPhase.ASKING_QUESTIONS

    diary.intake.name = "Clara Higgins"
    diary.intake.phone = "07700900200"
    diary.intake.nhs_number = "1234500002"
    diary.intake.dob = "1984-03-22"
    diary.intake.gp_name = "Dr. Okafor"

    diary.clinical.chief_complaint = "Hepatitis C — referral for FibroScan and DAA therapy"
    diary.clinical.condition_context = "Hepatitis C"
    diary.clinical.medical_history = [
        "Hepatitis C (diagnosed 2023, incidental finding)",
        "Anxiety disorder",
    ]
    diary.clinical.current_medications = [
        "Sertraline 50mg OD",
    ]
    diary.clinical.allergies = []
    diary.clinical.red_flags = []

    diary.clinical.referral_analysis = {
        "chief_complaint": "Hepatitis C — referral for FibroScan and DAA therapy",
        "condition_context": "Hepatitis C",
        "medical_history": [
            "Hepatitis C (incidental finding on routine bloods)",
            "Anxiety disorder",
        ],
        "current_medications": ["Sertraline 50mg OD"],
        "allergies": [],
        "red_flags": [],
        "lab_values": {
            "HCV RNA": "1.4 × 10⁶ IU/mL",
            "Genotype": "3a",
            "ALT": "52 U/L",
            "Bilirubin": "12 µmol/L",
            "Albumin": "42 g/L",
            "Platelets": "220 × 10⁹/L",
        },
        "key_findings": (
            "Asymptomatic Hepatitis C, Genotype 3a. HCV RNA 1.4 million IU/mL. "
            "Normal liver function apart from mildly elevated ALT. "
            "No signs of advanced fibrosis. FIB-4 score 0.8."
        ),
    }

    diary.clinical.referral_narrative = (
        "Clara Higgins is a 42-year-old woman referred with Hepatitis C "
        "Genotype 3a, diagnosed incidentally on routine blood tests in 2023. "
        "She is currently asymptomatic. HCV RNA viral load is 1.4 million IU/mL. "
        "Liver function is largely normal: ALT 52 U/L (mildly elevated), "
        "bilirubin 12 µmol/L, albumin 42 g/L, platelets 220 × 10⁹/L. "
        "FIB-4 score 0.8 — no evidence of advanced fibrosis. "
        "She has a history of anxiety disorder managed with sertraline 50mg OD. "
        "Referred for FibroScan assessment and consideration of DAA therapy."
    )

    return diary


# ── Pytest Tests (mocked LLM) ──


class TestArthurShelbyQuestions:
    """Question quality for PT-TEST-001 Arthur Shelby (HCC, 2WW, HIGH)."""

    @pytest.mark.asyncio
    async def test_fallback_generates_5_questions(self):
        """Fallback (no LLM) generates exactly 5 referral-aware questions."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_arthur_shelby_diary()

        questions = agent._fallback_question_plan(diary)

        assert len(questions) == 5

    @pytest.mark.asyncio
    async def test_fallback_references_specific_findings(self):
        """Fallback questions reference Arthur's specific clinical data."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_arthur_shelby_diary()

        questions = agent._fallback_question_plan(diary)
        all_text = " ".join(questions).lower()

        # Should reference at least one specific finding from the referral
        specific_refs = [
            "weight loss",       # red flag from referral
            "afp",               # lab value
            "485",               # lab value number
            "yellowing",         # screening question for HCC
            "skin",              # jaundice screening
        ]
        matches = [ref for ref in specific_refs if ref in all_text]
        assert len(matches) >= 2, (
            f"Expected at least 2 specific references, got {matches}. "
            f"Questions: {questions}"
        )

    @pytest.mark.asyncio
    async def test_fallback_not_generic(self):
        """Fallback questions are NOT purely generic template questions."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_arthur_shelby_diary()

        questions = agent._fallback_question_plan(diary)

        # At least one question should be clearly patient-specific
        generic_starters = [
            "could you tell me about the main reason",
            "do you have any existing medical conditions",
            "are you currently taking any medications",
        ]
        generic_count = sum(
            1 for q in questions
            if any(q.lower().startswith(g) for g in generic_starters)
        )
        assert generic_count == 0, (
            f"Found {generic_count} generic questions: {questions}"
        )

    @pytest.mark.asyncio
    async def test_llm_question_plan_stored(self):
        """LLM-generated questions are stored in diary."""
        llm_questions = json.dumps([
            "The 6.5 kg weight loss over 3 months — has it continued or stabilised?",
            "Have you noticed any yellowing of your skin or eyes since your GP visit?",
            "Your referral mentions right upper quadrant pain — has this changed?",
            "How is your energy day-to-day? Can you still manage your usual activities?",
            "What are you most worried about ahead of your consultation?",
        ])
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text=llm_questions)
        )
        agent = ClinicalAgent(llm_client=mock_client)
        diary = _make_arthur_shelby_diary()

        await agent._generate_initial_question_plan(diary)

        assert len(diary.clinical.generated_questions) == 5

    @pytest.mark.asyncio
    async def test_document_requests_fallback(self):
        """Fallback generates 2-3 relevant document requests for HCC."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_arthur_shelby_diary()

        docs = agent._fallback_document_requests(diary)

        assert 2 <= len(docs) <= 3
        all_text = " ".join(docs).lower()
        # HCC patient should get blood tests and imaging
        assert "blood" in all_text or "afp" in all_text
        assert "scan" in all_text or "ct" in all_text or "mri" in all_text

    @pytest.mark.asyncio
    async def test_document_requests_llm(self):
        """LLM-generated document requests stored correctly."""
        llm_docs = json.dumps([
            "blood test results (including AFP/tumour markers)",
            "CT or MRI scan reports",
        ])
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text=llm_docs)
        )
        agent = ClinicalAgent(llm_client=mock_client)
        diary = _make_arthur_shelby_diary()

        docs = await agent._generate_document_requests(diary)

        assert len(docs) == 2
        assert "AFP" in docs[0] or "blood" in docs[0].lower()


class TestClaraHigginsQuestions:
    """Question quality for PT-TEST-002 Clara Higgins (Hep C, LOW)."""

    @pytest.mark.asyncio
    async def test_fallback_generates_5_questions(self):
        """Fallback generates exactly 5 referral-aware questions."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_clara_higgins_diary()

        questions = agent._fallback_question_plan(diary)

        assert len(questions) == 5

    @pytest.mark.asyncio
    async def test_fallback_references_specific_findings(self):
        """Fallback questions reference Clara's specific clinical data."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_clara_higgins_diary()

        questions = agent._fallback_question_plan(diary)
        all_text = " ".join(questions).lower()

        # Should reference specific findings from her referral
        specific_refs = [
            "hcv rna",          # lab value
            "genotype",         # genotype
            "1.4",              # viral load number
            "hepatitis",        # condition
            "yellowing",        # hep C screening
            "dark-coloured",    # hep C screening
        ]
        matches = [ref for ref in specific_refs if ref in all_text]
        assert len(matches) >= 1, (
            f"Expected at least 1 specific reference, got {matches}. "
            f"Questions: {questions}"
        )

    @pytest.mark.asyncio
    async def test_different_from_arthur(self):
        """Clara's questions are meaningfully different from Arthur's."""
        agent = ClinicalAgent(llm_client=None)
        arthur_diary = _make_arthur_shelby_diary()
        clara_diary = _make_clara_higgins_diary()

        arthur_qs = agent._fallback_question_plan(arthur_diary)
        clara_qs = agent._fallback_question_plan(clara_diary)

        # No more than 1 question should be identical
        overlap = set(arthur_qs) & set(clara_qs)
        assert len(overlap) <= 1, (
            f"Too much overlap ({len(overlap)} identical questions): {overlap}"
        )

    @pytest.mark.asyncio
    async def test_fallback_not_generic(self):
        """Fallback questions are NOT purely generic."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_clara_higgins_diary()

        questions = agent._fallback_question_plan(diary)

        generic_starters = [
            "could you tell me about the main reason",
            "do you have any existing medical conditions",
            "are you currently taking any medications",
        ]
        generic_count = sum(
            1 for q in questions
            if any(q.lower().startswith(g) for g in generic_starters)
        )
        assert generic_count == 0, (
            f"Found {generic_count} generic questions: {questions}"
        )

    @pytest.mark.asyncio
    async def test_document_requests_fallback(self):
        """Fallback generates 2-3 relevant document requests for Hep C."""
        agent = ClinicalAgent(llm_client=None)
        diary = _make_clara_higgins_diary()

        docs = agent._fallback_document_requests(diary)

        assert 2 <= len(docs) <= 3
        all_text = " ".join(docs).lower()
        # Hep C patient should get viral load / blood tests
        assert "blood" in all_text or "viral" in all_text

    @pytest.mark.asyncio
    async def test_document_requests_different_from_arthur(self):
        """Clara's document requests differ from Arthur's."""
        agent = ClinicalAgent(llm_client=None)
        arthur_diary = _make_arthur_shelby_diary()
        clara_diary = _make_clara_higgins_diary()

        arthur_docs = agent._fallback_document_requests(arthur_diary)
        clara_docs = agent._fallback_document_requests(clara_diary)

        # At least one document type should differ
        assert set(arthur_docs) != set(clara_docs), (
            f"Documents are identical: Arthur={arthur_docs}, Clara={clara_docs}"
        )


# ── Live LLM Runner (no mocking) ──


async def _run_live():
    """Run with real LLM to see actual generated questions and documents."""
    import os
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as exc:
        print(f"Cannot create Gemini client: {exc}")
        print("Set GOOGLE_API_KEY to run live tests.")
        sys.exit(1)

    agent = ClinicalAgent(llm_client=client)

    for name, factory in [
        ("PT-TEST-001 Arthur Shelby (HCC, 2WW, HIGH)", _make_arthur_shelby_diary),
        ("PT-TEST-002 Clara Higgins (Hep C, LOW)", _make_clara_higgins_diary),
    ]:
        diary = factory()
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        # Generate questions
        await agent._generate_initial_question_plan(diary)
        print(f"\n  Questions ({len(diary.clinical.generated_questions)}):")
        for i, q in enumerate(diary.clinical.generated_questions, 1):
            print(f"    {i}. {q}")

        # Generate document requests
        docs = await agent._generate_document_requests(diary)
        print(f"\n  Document requests ({len(docs)}):")
        for i, d in enumerate(docs, 1):
            print(f"    {i}. {d}")

        # Quick quality checks
        print(f"\n  Quality checks:")
        q_count = len(diary.clinical.generated_questions)
        print(f"    Questions count: {q_count} {'PASS' if q_count == 5 else 'WARN'}")
        d_count = len(docs)
        print(f"    Document count:  {d_count} {'PASS' if 2 <= d_count <= 3 else 'WARN'}")

    print()


if __name__ == "__main__":
    asyncio.run(_run_live())

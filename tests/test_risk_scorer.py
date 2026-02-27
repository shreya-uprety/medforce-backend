"""
Tests for the RiskScorer — deterministic risk assessment.

Covers:
  - Hard rules (lab value thresholds)
  - Keyword rules (red-flag symptoms)
  - LLM fallback heuristics
  - Risk rank ordering
  - Edge cases (missing data, mixed signals)
"""

import pytest
from medforce.gateway.agents.risk_scorer import (
    HARD_RULES,
    KEYWORD_RULES,
    RiskResult,
    RiskScorer,
)
from medforce.gateway.diary import (
    ClinicalDocument,
    ClinicalQuestion,
    ClinicalSection,
    RiskLevel,
)


# ── Fixtures ──


def make_clinical(**kwargs) -> ClinicalSection:
    """Build a ClinicalSection with optional overrides."""
    return ClinicalSection(**kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hard Rule Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHardRules:
    """Deterministic lab-value thresholds — HIGHEST priority."""

    def test_high_bilirubin_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        # Bilirubin > 85 µmol/L → HIGH
        result = scorer.score(clinical, {"bilirubin": 90})
        assert result.risk_level == RiskLevel.HIGH
        assert "deterministic_rule" in result.method
        assert result.confidence == 1.0
        assert len(result.triggered_rules) >= 1

    def test_high_alt_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"ALT": 600})
        assert result.risk_level == RiskLevel.HIGH
        assert "ALT > 500" in result.triggered_rules[0]

    def test_high_ast_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"AST": 550})
        assert result.risk_level == RiskLevel.HIGH

    def test_low_platelets_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"platelets": 40})
        assert result.risk_level == RiskLevel.HIGH
        assert "Platelets" in result.triggered_rules[0]

    def test_high_inr_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"INR": 2.5})
        assert result.risk_level == RiskLevel.HIGH

    def test_high_creatinine_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        # Creatinine > 300 µmol/L → HIGH
        result = scorer.score(clinical, {"creatinine": 350})
        assert result.risk_level == RiskLevel.HIGH

    def test_low_albumin_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        # Albumin < 25 g/L → HIGH
        result = scorer.score(clinical, {"albumin": 20})
        assert result.risk_level == RiskLevel.HIGH

    def test_medium_bilirubin_triggers_medium_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        # Bilirubin 30 µmol/L: > 20 (MEDIUM) but < 85 (HIGH)
        result = scorer.score(clinical, {"bilirubin": 30})
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.confidence == 1.0

    def test_medium_alt_triggers_medium_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"alt": 300})
        assert result.risk_level == RiskLevel.MEDIUM

    def test_medium_platelets_triggers_medium_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"platelets": 80})
        assert result.risk_level == RiskLevel.MEDIUM

    def test_medium_inr_triggers_medium_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"inr": 1.8})
        assert result.risk_level == RiskLevel.MEDIUM

    def test_highest_risk_wins_with_multiple_triggers(self):
        """When multiple rules fire, the highest risk level wins."""
        scorer = RiskScorer()
        clinical = make_clinical()
        # bilirubin 30 → MEDIUM, ALT 600 → HIGH
        result = scorer.score(clinical, {"bilirubin": 30, "ALT": 600})
        assert result.risk_level == RiskLevel.HIGH
        assert len(result.triggered_rules) >= 2

    def test_non_numeric_values_ignored(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {"bilirubin": "not-a-number"})
        # Should fall through to fallback since no numeric rules fire
        assert result.risk_level != RiskLevel.HIGH

    def test_missing_labs_no_hard_rules(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {})
        assert result.risk_level in (RiskLevel.NONE, RiskLevel.LOW)

    def test_normal_values_no_trigger(self):
        """Normal lab values should not trigger hard rules."""
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {
            "bilirubin": 10,  # Normal: < 20 µmol/L
            "ALT": 30,
            "AST": 25,
            "platelets": 250,
            "INR": 1.0,
        })
        # No hard rules fire — should fall through
        assert result.risk_level in (RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM)
        assert "deterministic_rule" not in result.method

    def test_case_insensitive_lab_keys(self):
        """Both 'ALT' and 'alt' should work."""
        scorer = RiskScorer()
        clinical = make_clinical()
        result_upper = scorer.score(clinical, {"ALT": 600})
        result_lower = scorer.score(clinical, {"alt": 600})
        assert result_upper.risk_level == RiskLevel.HIGH
        assert result_lower.risk_level == RiskLevel.HIGH


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Keyword Rule Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKeywordRules:
    """Red-flag symptom keywords — second priority after hard rules."""

    def test_jaundice_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(red_flags=["jaundice"])
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH
        assert "keyword" in result.method
        assert result.confidence == 0.9

    def test_confusion_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(red_flags=["confusion"])
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

    def test_gi_bleeding_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(
            chief_complaint="I have been experiencing gi bleeding"
        )
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

    def test_ascites_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(red_flags=["ascites"])
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

    def test_encephalopathy_triggers_high_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(
            medical_history=["hepatic encephalopathy"]
        )
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

    def test_fatigue_triggers_medium_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(chief_complaint="I have extreme fatigue")
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.MEDIUM

    def test_nausea_triggers_low_risk(self):
        scorer = RiskScorer()
        clinical = make_clinical(chief_complaint="I have nausea")
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.LOW

    def test_keyword_in_question_answer(self):
        """Keywords in Q&A answers should be detected."""
        scorer = RiskScorer()
        q = ClinicalQuestion(question="Any symptoms?", answer="I have jaundice")
        clinical = make_clinical(questions_asked=[q])
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

    def test_multiple_keywords_highest_wins(self):
        """Multiple keywords → highest risk level wins."""
        scorer = RiskScorer()
        clinical = make_clinical(
            chief_complaint="I have nausea and jaundice"
        )
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

    def test_hard_rules_override_keywords(self):
        """Hard rules should take precedence over keywords."""
        scorer = RiskScorer()
        clinical = make_clinical(chief_complaint="I have nausea")
        # Bilirubin > 85 µmol/L → HIGH (hard rule overrides keyword)
        result = scorer.score(clinical, {"bilirubin": 90})
        assert result.risk_level == RiskLevel.HIGH
        assert "deterministic_rule" in result.method


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Fallback / Heuristic Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLLMFallback:
    """Gray-zone heuristics when no deterministic rules fire."""

    def test_no_data_defaults_to_low(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.LOW
        assert "insufficient_data" in result.method

    def test_red_flags_present_gives_medium(self):
        """Red flags that don't match keywords still yield MEDIUM via heuristic."""
        scorer = RiskScorer()
        clinical = make_clinical(
            red_flags=["some_unusual_flag_not_in_keywords"]
        )
        result = scorer.score(clinical, {})
        # The keyword matching might not catch custom red flags,
        # but the LLM fallback checks red_flags list
        assert result.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_labs_present_no_triggers_gives_low_or_medium(self):
        """Labs present without concerning findings → LOW (minimal concern)."""
        scorer = RiskScorer()
        clinical = make_clinical(chief_complaint="general checkup")
        result = scorer.score(clinical, {"random_lab": 50})
        # With updated heuristic: minimal data (just labs, no red flags) → LOW
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_no_concerning_findings_gives_low(self):
        scorer = RiskScorer()
        clinical = make_clinical(
            chief_complaint="routine checkup",
            medical_history=["none"],
        )
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.LOW
        assert "no_concerning_findings" in result.method


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  score_from_extracted_values Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestScoreFromDocuments:
    """Score using lab values from clinical documents."""

    def test_extracts_values_from_documents(self):
        scorer = RiskScorer()
        doc = ClinicalDocument(
            type="lab_results",
            source="gp",
            processed=True,
            # Bilirubin > 85 µmol/L → HIGH
            extracted_values={"bilirubin": 90, "ALT": 300},
        )
        clinical = make_clinical(documents=[doc])
        result = scorer.score_from_extracted_values(clinical)
        assert result.risk_level == RiskLevel.HIGH

    def test_merges_multiple_documents(self):
        scorer = RiskScorer()
        doc1 = ClinicalDocument(
            type="blood_test", processed=True,
            extracted_values={"bilirubin": 1.0},
        )
        doc2 = ClinicalDocument(
            type="lab_results", processed=True,
            extracted_values={"ALT": 600},
        )
        clinical = make_clinical(documents=[doc1, doc2])
        result = scorer.score_from_extracted_values(clinical)
        assert result.risk_level == RiskLevel.HIGH

    def test_empty_documents_fallback(self):
        scorer = RiskScorer()
        clinical = make_clinical()
        result = scorer.score_from_extracted_values(clinical)
        assert result.risk_level == RiskLevel.LOW


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Risk Rank Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskRank:
    """Verify _risk_rank ordering."""

    def test_rank_ordering(self):
        scorer = RiskScorer()
        assert scorer._risk_rank(RiskLevel.NONE) == 0
        assert scorer._risk_rank(RiskLevel.LOW) == 1
        assert scorer._risk_rank(RiskLevel.MEDIUM) == 2
        assert scorer._risk_rank(RiskLevel.HIGH) == 3
        assert scorer._risk_rank(RiskLevel.CRITICAL) == 4

    def test_critical_is_highest(self):
        scorer = RiskScorer()
        assert scorer._risk_rank(RiskLevel.CRITICAL) > scorer._risk_rank(RiskLevel.HIGH)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Patient Scenario Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatientScenarios:
    """Realistic patient scenarios for risk scoring."""

    def test_scenario_severe_liver_disease(self):
        """Patient with severe liver disease — HIGH risk."""
        scorer = RiskScorer()
        clinical = make_clinical(
            chief_complaint="jaundice and abdominal swelling",
            red_flags=["jaundice", "ascites"],
            medical_history=["chronic liver disease"],
        )
        # UK units: bilirubin µmol/L, ALT U/L, albumin g/L
        labs = {"bilirubin": 90, "ALT": 700, "INR": 2.5, "albumin": 20}
        result = scorer.score(clinical, labs)
        assert result.risk_level == RiskLevel.HIGH
        assert result.confidence == 1.0
        assert len(result.triggered_rules) >= 3

    def test_scenario_mild_abnormality(self):
        """Patient with mildly elevated labs — MEDIUM risk."""
        scorer = RiskScorer()
        clinical = make_clinical(
            chief_complaint="routine checkup",
        )
        labs = {"ALT": 250}
        result = scorer.score(clinical, labs)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_scenario_healthy_patient(self):
        """Patient with normal labs — no hard rules fire, labs present → MEDIUM via heuristic."""
        scorer = RiskScorer()
        clinical = make_clinical(
            chief_complaint="routine health check",
            medical_history=["none significant"],
        )
        labs = {"bilirubin": 8, "ALT": 20, "AST": 18, "platelets": 300}
        result = scorer.score(clinical, labs)
        # Labs present but no hard rules fired → heuristic (moderate concerns)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_scenario_gi_bleeding_emergency(self):
        """Patient reporting GI bleeding — HIGH risk via keyword."""
        scorer = RiskScorer()
        q = ClinicalQuestion(
            question="Any concerning symptoms?",
            answer="I have been having melena and hematemesis"
        )
        clinical = make_clinical(questions_asked=[q])
        result = scorer.score(clinical, {})
        assert result.risk_level == RiskLevel.HIGH

"""
Risk Scorer — Deterministic clinical risk assessment.

Hard rules ALWAYS override LLM assessment.  This is the most safety-critical
component in the entire system.

Scoring order:
  1. Check deterministic hard rules (lab values vs thresholds)
  2. Check keyword rules (red-flag symptoms)
  3. Only if NO hard rule fired → use LLM for gray-zone assessment
  4. Log which method determined the score
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from medforce.gateway.diary import ClinicalSection, RiskLevel

logger = logging.getLogger("gateway.agents.risk_scorer")


@dataclass
class RiskResult:
    """Outcome of risk scoring."""

    risk_level: RiskLevel
    method: str  # "deterministic_rule: bilirubin > 5", "keyword: jaundice", "llm_assessment"
    reasoning: str = ""
    triggered_rules: list[str] = field(default_factory=list)
    confidence: float = 1.0  # 1.0 for deterministic, lower for LLM


# ── Hard Rules: lab value thresholds ──
# (parameter, operator, threshold, risk_level, human description)
HARD_RULES: list[tuple[str, str, float, RiskLevel, str]] = [
    ("bilirubin", ">", 5.0, RiskLevel.HIGH, "Bilirubin > 5 mg/dL"),
    ("total_bilirubin", ">", 5.0, RiskLevel.HIGH, "Total bilirubin > 5 mg/dL"),
    ("ALT", ">", 500, RiskLevel.HIGH, "ALT > 500 U/L"),
    ("alt", ">", 500, RiskLevel.HIGH, "ALT > 500 U/L"),
    ("AST", ">", 500, RiskLevel.HIGH, "AST > 500 U/L"),
    ("ast", ">", 500, RiskLevel.HIGH, "AST > 500 U/L"),
    ("platelets", "<", 50, RiskLevel.HIGH, "Platelets < 50 x10^9/L"),
    ("platelet_count", "<", 50, RiskLevel.HIGH, "Platelet count < 50 x10^9/L"),
    ("INR", ">", 2.0, RiskLevel.HIGH, "INR > 2.0"),
    ("inr", ">", 2.0, RiskLevel.HIGH, "INR > 2.0"),
    ("creatinine", ">", 3.0, RiskLevel.HIGH, "Creatinine > 3.0 mg/dL"),
    ("albumin", "<", 2.5, RiskLevel.HIGH, "Albumin < 2.5 g/dL"),
    # Medium-risk rules
    ("bilirubin", ">", 2.0, RiskLevel.MEDIUM, "Bilirubin > 2 mg/dL"),
    ("total_bilirubin", ">", 2.0, RiskLevel.MEDIUM, "Total bilirubin > 2 mg/dL"),
    ("ALT", ">", 200, RiskLevel.MEDIUM, "ALT > 200 U/L"),
    ("alt", ">", 200, RiskLevel.MEDIUM, "ALT > 200 U/L"),
    ("AST", ">", 200, RiskLevel.MEDIUM, "AST > 200 U/L"),
    ("ast", ">", 200, RiskLevel.MEDIUM, "AST > 200 U/L"),
    ("platelets", "<", 100, RiskLevel.MEDIUM, "Platelets < 100 x10^9/L"),
    ("INR", ">", 1.5, RiskLevel.MEDIUM, "INR > 1.5"),
    ("inr", ">", 1.5, RiskLevel.MEDIUM, "INR > 1.5"),
]

# ── Keyword Rules: red-flag symptoms ──
KEYWORD_RULES: list[tuple[str, RiskLevel, str]] = [
    ("jaundice", RiskLevel.HIGH, "Jaundice reported"),
    ("confusion", RiskLevel.HIGH, "Confusion / altered mental status"),
    ("encephalopathy", RiskLevel.HIGH, "Encephalopathy"),
    ("gi_bleeding", RiskLevel.HIGH, "GI bleeding"),
    ("gi bleeding", RiskLevel.HIGH, "GI bleeding"),
    ("gastrointestinal bleeding", RiskLevel.HIGH, "GI bleeding"),
    ("ascites", RiskLevel.HIGH, "Ascites"),
    ("variceal", RiskLevel.HIGH, "Variceal bleeding"),
    ("hematemesis", RiskLevel.HIGH, "Hematemesis"),
    ("melena", RiskLevel.HIGH, "Melena"),
    ("hepatic failure", RiskLevel.HIGH, "Hepatic failure"),
    ("liver failure", RiskLevel.HIGH, "Liver failure"),
    ("coagulopathy", RiskLevel.HIGH, "Coagulopathy"),
    # Medium keywords
    ("fatigue", RiskLevel.MEDIUM, "Significant fatigue"),
    ("abdominal pain", RiskLevel.MEDIUM, "Abdominal pain"),
    ("nausea", RiskLevel.LOW, "Nausea"),
    ("itching", RiskLevel.LOW, "Pruritus / itching"),
    ("pruritus", RiskLevel.LOW, "Pruritus"),
]


class RiskScorer:
    """
    Deterministic risk scorer with LLM fallback.

    Usage:
        scorer = RiskScorer()
        result = scorer.score(clinical_section, lab_values)
    """

    def __init__(self, llm_client=None) -> None:
        self._client = llm_client
        self._model_name = os.getenv("CLINICAL_MODEL", "gemini-2.0-flash")

    def score(
        self,
        clinical: ClinicalSection,
        lab_values: dict[str, Any] | None = None,
    ) -> RiskResult:
        """
        Score risk using the three-tier approach.

        Args:
            clinical: ClinicalSection from the patient diary
            lab_values: dict of lab parameter → numeric value

        Returns:
            RiskResult with the determined risk level and method
        """
        labs = lab_values or {}
        triggered: list[str] = []

        # ── Tier 1: Deterministic hard rules (HIGHEST PRIORITY) ──
        highest_risk = RiskLevel.NONE

        for param, op, threshold, risk, desc in HARD_RULES:
            value = labs.get(param)
            if value is None:
                continue

            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                continue

            if op == ">" and numeric_value > threshold:
                triggered.append(desc)
                if self._risk_rank(risk) > self._risk_rank(highest_risk):
                    highest_risk = risk
            elif op == "<" and numeric_value < threshold:
                triggered.append(desc)
                if self._risk_rank(risk) > self._risk_rank(highest_risk):
                    highest_risk = risk

        if highest_risk != RiskLevel.NONE:
            logger.info(
                "Deterministic risk: %s — triggered: %s",
                highest_risk.value,
                triggered,
            )
            return RiskResult(
                risk_level=highest_risk,
                method=f"deterministic_rule: {triggered[0]}",
                reasoning=f"Hard rules triggered: {', '.join(triggered)}",
                triggered_rules=triggered,
                confidence=1.0,
            )

        # ── Tier 2: Keyword rules (red-flag symptoms) ──
        all_text = self._collect_clinical_text(clinical)

        for keyword, risk, desc in KEYWORD_RULES:
            if keyword.lower() in all_text:
                triggered.append(desc)
                if self._risk_rank(risk) > self._risk_rank(highest_risk):
                    highest_risk = risk

        if highest_risk != RiskLevel.NONE:
            logger.info(
                "Keyword risk: %s — triggered: %s",
                highest_risk.value,
                triggered,
            )
            return RiskResult(
                risk_level=highest_risk,
                method=f"keyword: {triggered[0]}",
                reasoning=f"Red-flag keywords detected: {', '.join(triggered)}",
                triggered_rules=triggered,
                confidence=0.9,
            )

        # ── Tier 3: LLM assessment (gray zone only) ──
        # Only reach here if NO deterministic rules fired
        return self._llm_fallback(clinical, labs)

    def score_from_extracted_values(
        self,
        clinical: ClinicalSection,
    ) -> RiskResult:
        """
        Score using lab values already extracted into clinical.documents.

        Collects all extracted_values from clinical documents and passes
        them through the standard scoring pipeline.
        """
        lab_values: dict[str, Any] = {}
        for doc in clinical.documents:
            if doc.extracted_values:
                lab_values.update(doc.extracted_values)
        return self.score(clinical, lab_values)

    # ── Internal ──

    def _collect_clinical_text(self, clinical: ClinicalSection) -> str:
        """Build a searchable text corpus from clinical data."""
        parts = []
        if clinical.chief_complaint:
            parts.append(clinical.chief_complaint)
        parts.extend(clinical.medical_history)
        parts.extend(clinical.red_flags)
        for q in clinical.questions_asked:
            if q.answer:
                parts.append(q.answer)
        return " ".join(parts).lower()

    def _llm_fallback(
        self,
        clinical: ClinicalSection,
        labs: dict[str, Any],
    ) -> RiskResult:
        """
        Use LLM for gray-zone risk assessment.

        This is the ONLY place where LLM output affects risk scoring,
        and it only runs when no deterministic rules fired.
        """
        # If no clinical data at all, default to LOW
        has_data = (
            clinical.chief_complaint
            or clinical.medical_history
            or clinical.red_flags
            or labs
        )

        if not has_data:
            return RiskResult(
                risk_level=RiskLevel.LOW,
                method="default: insufficient_data",
                reasoning="Insufficient clinical data for assessment — defaulting to LOW",
                confidence=0.5,
            )

        # For now (without LLM), use a simple heuristic
        # Phase 3 can wire the actual LLM call here
        if clinical.red_flags:
            return RiskResult(
                risk_level=RiskLevel.MEDIUM,
                method="heuristic: red_flags_present",
                reasoning=f"Red flags noted: {', '.join(clinical.red_flags)}",
                confidence=0.7,
            )

        if labs:
            return RiskResult(
                risk_level=RiskLevel.MEDIUM,
                method="heuristic: labs_present_no_hard_rules",
                reasoning="Lab values present but no hard rules triggered — medium by default",
                confidence=0.6,
            )

        return RiskResult(
            risk_level=RiskLevel.LOW,
            method="heuristic: no_concerning_findings",
            reasoning="No red flags, no abnormal labs — low risk",
            confidence=0.7,
        )

    @staticmethod
    def _risk_rank(level: RiskLevel) -> int:
        """Numeric rank for risk comparison."""
        return {
            RiskLevel.NONE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }.get(level, 0)

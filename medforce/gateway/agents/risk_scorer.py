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
# Values are UNIT-AGNOSTIC: the scorer extracts the numeric part from strings
# like "28 µmol/L" or "485 kU/L".  Thresholds are set for the MOST COMMON
# unit used in UK referral letters (µmol/L for bilirubin, U/L for enzymes,
# x10^9/L for platelets, g/L for albumin).
# (parameter, operator, threshold, risk_level, human description)
HARD_RULES: list[tuple[str, str, float, RiskLevel, str]] = [
    # ── HIGH risk (immediate clinical concern) ──
    # Bilirubin: ≥50 µmol/L is significantly elevated (3x upper normal)
    ("bilirubin", ">", 50, RiskLevel.HIGH, "Bilirubin > 50 µmol/L (severe)"),
    ("total_bilirubin", ">", 50, RiskLevel.HIGH, "Total bilirubin > 50 µmol/L"),
    # Liver enzymes: >500 U/L indicates acute liver injury
    ("ALT", ">", 500, RiskLevel.HIGH, "ALT > 500 U/L"),
    ("alt", ">", 500, RiskLevel.HIGH, "ALT > 500 U/L"),
    ("AST", ">", 500, RiskLevel.HIGH, "AST > 500 U/L"),
    ("ast", ">", 500, RiskLevel.HIGH, "AST > 500 U/L"),
    # Platelets: <50 indicates severe thrombocytopenia
    ("platelets", "<", 50, RiskLevel.HIGH, "Platelets < 50 x10^9/L"),
    ("platelet_count", "<", 50, RiskLevel.HIGH, "Platelet count < 50 x10^9/L"),
    # Coagulation
    ("INR", ">", 2.0, RiskLevel.HIGH, "INR > 2.0"),
    ("inr", ">", 2.0, RiskLevel.HIGH, "INR > 2.0"),
    # Renal
    ("creatinine", ">", 300, RiskLevel.HIGH, "Creatinine > 300 µmol/L"),
    # Albumin: <25 g/L is critically low
    ("albumin", "<", 25, RiskLevel.HIGH, "Albumin < 25 g/L"),
    # Tumour markers: AFP >400 strongly suggests HCC
    ("AFP", ">", 400, RiskLevel.HIGH, "AFP > 400 kU/L (HCC marker)"),
    ("afp", ">", 400, RiskLevel.HIGH, "AFP > 400 kU/L (HCC marker)"),
    ("alpha_fetoprotein", ">", 400, RiskLevel.HIGH, "AFP > 400 kU/L"),

    # ── MEDIUM risk (needs timely attention) ──
    # Bilirubin: >20 µmol/L is above normal
    ("bilirubin", ">", 20, RiskLevel.MEDIUM, "Bilirubin > 20 µmol/L (elevated)"),
    ("total_bilirubin", ">", 20, RiskLevel.MEDIUM, "Total bilirubin > 20 µmol/L"),
    # Liver enzymes: >200 is significantly elevated
    ("ALT", ">", 200, RiskLevel.MEDIUM, "ALT > 200 U/L"),
    ("alt", ">", 200, RiskLevel.MEDIUM, "ALT > 200 U/L"),
    ("AST", ">", 200, RiskLevel.MEDIUM, "AST > 200 U/L"),
    ("ast", ">", 200, RiskLevel.MEDIUM, "AST > 200 U/L"),
    # ALP: >300 is significantly elevated (e.g. cholestatic pattern)
    ("ALP", ">", 300, RiskLevel.MEDIUM, "ALP > 300 U/L"),
    ("alp", ">", 300, RiskLevel.MEDIUM, "ALP > 300 U/L"),
    # GGT: >200 is significantly elevated
    ("GGT", ">", 200, RiskLevel.MEDIUM, "GGT > 200 U/L"),
    ("ggt", ">", 200, RiskLevel.MEDIUM, "GGT > 200 U/L"),
    # Platelets: <100 indicates moderate thrombocytopenia
    ("platelets", "<", 100, RiskLevel.MEDIUM, "Platelets < 100 x10^9/L"),
    # Coagulation
    ("INR", ">", 1.5, RiskLevel.MEDIUM, "INR > 1.5"),
    ("inr", ">", 1.5, RiskLevel.MEDIUM, "INR > 1.5"),
    # Tumour markers: AFP >20 is elevated
    ("AFP", ">", 20, RiskLevel.MEDIUM, "AFP > 20 kU/L (elevated)"),
    ("afp", ">", 20, RiskLevel.MEDIUM, "AFP > 20 kU/L (elevated)"),
    ("alpha_fetoprotein", ">", 20, RiskLevel.MEDIUM, "AFP > 20 kU/L"),
    # FIB-4 score: >3.25 strongly suggests advanced fibrosis
    ("FIB-4", ">", 3.25, RiskLevel.MEDIUM, "FIB-4 > 3.25 (advanced fibrosis)"),
    ("fib_4", ">", 3.25, RiskLevel.MEDIUM, "FIB-4 > 3.25"),
    ("fib4", ">", 3.25, RiskLevel.MEDIUM, "FIB-4 > 3.25"),
]

# ── Keyword Rules: red-flag symptoms ──
KEYWORD_RULES: list[tuple[str, RiskLevel, str]] = [
    # HIGH — oncology / suspected cancer
    ("carcinoma", RiskLevel.HIGH, "Suspected carcinoma"),
    ("cancer", RiskLevel.HIGH, "Suspected cancer"),
    ("malignancy", RiskLevel.HIGH, "Suspected malignancy"),
    ("tumour", RiskLevel.HIGH, "Suspected tumour"),
    ("tumor", RiskLevel.HIGH, "Suspected tumor"),
    ("hcc", RiskLevel.HIGH, "Suspected hepatocellular carcinoma"),
    ("mass", RiskLevel.HIGH, "Suspicious mass identified"),
    ("2-week wait", RiskLevel.HIGH, "2-week wait pathway"),
    ("two week wait", RiskLevel.HIGH, "2-week wait pathway"),
    ("2ww", RiskLevel.HIGH, "2-week wait pathway"),
    # HIGH — hepatic emergencies
    ("jaundice", RiskLevel.HIGH, "Jaundice reported"),
    ("icterus", RiskLevel.HIGH, "Icterus / jaundice"),
    ("confusion", RiskLevel.HIGH, "Confusion / altered mental status"),
    ("encephalopathy", RiskLevel.HIGH, "Encephalopathy"),
    ("altered mental", RiskLevel.HIGH, "Altered mental status"),
    ("gi_bleeding", RiskLevel.HIGH, "GI bleeding"),
    ("gi bleeding", RiskLevel.HIGH, "GI bleeding"),
    ("gastrointestinal bleeding", RiskLevel.HIGH, "GI bleeding"),
    ("ascites", RiskLevel.HIGH, "Ascites"),
    ("variceal", RiskLevel.HIGH, "Variceal bleeding"),
    ("hematemesis", RiskLevel.HIGH, "Hematemesis"),
    ("melena", RiskLevel.HIGH, "Melena"),
    ("hepatic failure", RiskLevel.HIGH, "Hepatic failure"),
    ("liver failure", RiskLevel.HIGH, "Liver failure"),
    ("decompensated", RiskLevel.HIGH, "Decompensated liver disease"),
    ("coagulopathy", RiskLevel.HIGH, "Coagulopathy"),
    ("spider naevi", RiskLevel.HIGH, "Spider naevi (portal hypertension)"),
    ("weight loss", RiskLevel.HIGH, "Unexplained weight loss"),
    # MEDIUM
    ("fatigue", RiskLevel.MEDIUM, "Significant fatigue"),
    ("abdominal pain", RiskLevel.MEDIUM, "Abdominal pain"),
    ("cirrhosis", RiskLevel.MEDIUM, "Cirrhosis"),
    ("fibrosis", RiskLevel.MEDIUM, "Fibrosis"),
    ("splenomegaly", RiskLevel.MEDIUM, "Splenomegaly"),
    ("hepatomegaly", RiskLevel.MEDIUM, "Hepatomegaly"),
    # LOW
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
        labs = self._normalize_lab_keys(lab_values or {})
        triggered: list[str] = []

        # ── Tier 1: Deterministic hard rules ──
        highest_risk = RiskLevel.NONE
        method = ""

        for param, op, threshold, risk, desc in HARD_RULES:
            value = labs.get(param)
            if value is None:
                continue

            numeric_value = self._parse_lab_value(value)
            if numeric_value is None:
                continue

            if op == ">" and numeric_value > threshold:
                triggered.append(desc)
                if self._risk_rank(risk) > self._risk_rank(highest_risk):
                    highest_risk = risk
                    method = f"deterministic_rule: {desc}"
            elif op == "<" and numeric_value < threshold:
                triggered.append(desc)
                if self._risk_rank(risk) > self._risk_rank(highest_risk):
                    highest_risk = risk
                    method = f"deterministic_rule: {desc}"

        # ── Tier 2: Keyword rules (red-flag symptoms) ──
        # ALWAYS check keywords — they can elevate risk above lab rules.
        # E.g., MEDIUM from labs + HIGH from "jaundice" keyword → HIGH.
        all_text = self._collect_clinical_text(clinical)

        for keyword, risk, desc in KEYWORD_RULES:
            if keyword.lower() in all_text:
                triggered.append(desc)
                if self._risk_rank(risk) > self._risk_rank(highest_risk):
                    highest_risk = risk
                    method = f"keyword: {desc}"

        if highest_risk != RiskLevel.NONE:
            # Determine confidence: deterministic = 1.0, keyword = 0.9
            confidence = 1.0 if "deterministic_rule" in method else 0.9
            reasoning_parts = [r for r in triggered]

            logger.info(
                "Risk: %s (method: %s) — triggered: %s",
                highest_risk.value, method, triggered,
            )
            return RiskResult(
                risk_level=highest_risk,
                method=method,
                reasoning=f"Triggered: {', '.join(reasoning_parts)}",
                triggered_rules=triggered,
                confidence=confidence,
            )

        # ── Tier 3: Heuristic assessment (gray zone only) ──
        # Only reach here if NO deterministic or keyword rules fired
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

    @staticmethod
    def _parse_lab_value(value: Any) -> float | None:
        """Extract a numeric value from lab results that may include units.

        Handles formats like:
          - 28              → 28.0
          - "28"            → 28.0
          - "28 µmol/L"    → 28.0
          - "485 kU/L"     → 485.0
          - "1.6 mg/dL"    → 1.6
          - ">500"         → 500.0
          - "<50"          → 50.0
          - "1.4 × 10^6"  → 1400000.0
        """
        if value is None:
            return None

        # Already numeric
        if isinstance(value, (int, float)):
            return float(value)

        s = str(value).strip()
        if not s:
            return None

        import re

        # Handle scientific notation with ×: "1.4 × 10^6"
        sci_match = re.match(r'([<>]?\s*[\d.]+)\s*[×x]\s*10\^?(\d+)', s)
        if sci_match:
            base = float(sci_match.group(1).lstrip('<> '))
            exp = int(sci_match.group(2))
            return base * (10 ** exp)

        # Strip leading < or > (take the value as-is for threshold comparison)
        s = s.lstrip('<> ')

        # Extract first numeric part (integer or decimal)
        num_match = re.match(r'([\d.]+)', s)
        if num_match:
            try:
                return float(num_match.group(1))
            except ValueError:
                return None

        return None

    @staticmethod
    def _normalize_lab_keys(labs: dict[str, Any]) -> dict[str, Any]:
        """Normalize lab parameter keys for consistent matching.

        Handles variations like:
          - "Total Bilirubin" → "total_bilirubin"
          - "Platelet Count" → "platelet_count"
          - "Alpha-fetoprotein" → "alpha_fetoprotein"
          - "FIB-4 Score" → "fib_4"
        """
        normalized: dict[str, Any] = {}
        for key, value in labs.items():
            # Lowercase and replace spaces/hyphens with underscores
            norm_key = key.lower().replace(" ", "_").replace("-", "_")
            # Remove trailing descriptors like "_score", "_level", "_count"
            # but keep specific compound keys
            normalized[norm_key] = value
            # Also store common aliases
            if "bilirubin" in norm_key and "total" not in norm_key:
                normalized["bilirubin"] = value
            if "platelet" in norm_key:
                normalized["platelets"] = value
                normalized["platelet_count"] = value
            if "fetoprotein" in norm_key or norm_key in ("afp", "alpha_fetoprotein"):
                normalized["afp"] = value
                normalized["AFP"] = value
                normalized["alpha_fetoprotein"] = value
            if "fib" in norm_key and "4" in norm_key:
                normalized["fib_4"] = value
                normalized["FIB-4"] = value
                normalized["fib4"] = value
        return normalized

    def _collect_clinical_text(self, clinical: ClinicalSection) -> str:
        """Build a searchable text corpus from clinical data."""
        parts = []
        if clinical.chief_complaint:
            parts.append(clinical.chief_complaint)
        if clinical.condition_context:
            parts.append(clinical.condition_context)
        parts.extend(clinical.medical_history)
        parts.extend(clinical.red_flags)
        for q in clinical.questions_asked:
            if q.answer:
                parts.append(q.answer)
        # Include referral analysis key_findings if available
        if clinical.referral_analysis:
            findings = clinical.referral_analysis.get("key_findings", "")
            if findings:
                parts.append(findings)
        return " ".join(parts).lower()

    def _llm_fallback(
        self,
        clinical: ClinicalSection,
        labs: dict[str, Any],
    ) -> RiskResult:
        """
        Heuristic risk assessment for the gray zone.

        Only runs when no deterministic hard rules or keyword rules fired.
        Uses clinical context to make a more informed assessment than
        a blanket MEDIUM.
        """
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

        # Count concerning factors for a more nuanced assessment
        concern_score = 0
        reasons = []

        # Red flags contribute significantly
        if clinical.red_flags:
            concern_score += len(clinical.red_flags) * 2
            reasons.append(f"Red flags: {', '.join(clinical.red_flags[:3])}")

        # Pain level
        if clinical.pain_level is not None:
            if clinical.pain_level >= 7:
                concern_score += 3
                reasons.append(f"Significant pain ({clinical.pain_level}/10)")
            elif clinical.pain_level >= 4:
                concern_score += 1
                reasons.append(f"Moderate pain ({clinical.pain_level}/10)")

        # Multiple comorbidities
        if len(clinical.medical_history) >= 3:
            concern_score += 2
            reasons.append(f"Multiple comorbidities ({len(clinical.medical_history)})")
        elif clinical.medical_history:
            concern_score += 1

        # Abnormal labs present but below hard thresholds
        if labs:
            concern_score += 1
            reasons.append("Abnormal lab values present")

        # Urgency indicators in referral
        if clinical.referral_analysis:
            urgency = str(clinical.referral_analysis.get("urgency", "")).lower()
            if any(u in urgency for u in ["urgent", "2-week", "2ww", "emergency", "same-day"]):
                concern_score += 4
                reasons.append(f"Urgent referral: {urgency}")

        reasoning = "; ".join(reasons) if reasons else "No specific concerning findings"

        if concern_score >= 5:
            return RiskResult(
                risk_level=RiskLevel.HIGH,
                method="heuristic: multiple_concerns",
                reasoning=reasoning,
                confidence=0.75,
            )
        if concern_score >= 2:
            return RiskResult(
                risk_level=RiskLevel.MEDIUM,
                method="heuristic: moderate_concerns",
                reasoning=reasoning,
                confidence=0.7,
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

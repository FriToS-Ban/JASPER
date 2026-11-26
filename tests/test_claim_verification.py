"""Claim extraction and deterministic verification."""
from __future__ import annotations

import pytest

from hypothesis_agent.research import analytics
from hypothesis_agent.research.claims import extract_claims, verify_text
from hypothesis_agent.research.evidence import EvidenceRegistry
from hypothesis_agent.research.extraction import build_snapshot, extract_evidence
from tests.research_helpers import company_payloads


@pytest.fixture
def registry() -> EvidenceRegistry:
    registry = EvidenceRegistry()
    payloads = company_payloads()
    for slug, key in (("YFINANCE_FINANCIALS", "YFINANCE_FINANCIALS|annual"), ("YFINANCE_CASH_FLOW", "YFINANCE_CASH_FLOW|annual"),
                      ("YFINANCE_COMPANY_INFO", "YFINANCE_COMPANY_INFO")):
        snapshot = build_snapshot(ticker="TEST", tool_slug=slug, parameters={"symbol": "TEST"}, payload=payloads[key],
                                  retrieval_function=f"yfinance.Ticker('TEST').{slug}")
        registry.register_snapshot(snapshot)
        extract_evidence(registry, snapshot, payloads[key], currency="USD")
    analytics.compute_ratio(registry, "TEST", "operating_margin")
    analytics.compute_growth(registry, "TEST", "total_revenue")
    return registry


def _one(registry: EvidenceRegistry, text: str, verdict: str = "SUPPORTED"):
    report = verify_text({"narrative": text}, registry, "TEST", verdict=verdict)
    assert report.claims, f"no claim extracted from: {text}"
    return report.claims[0], report


def test_correct_numeric_claim_is_verified(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Revenue reached $133.1 billion in FY2025.")
    assert claim.metric == "total_revenue" and claim.period_year == 2025
    assert claim.status == "VERIFIED" and claim.matched_evidence


def test_rounding_difference_is_tolerated(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Operating margin was 16% in FY2025.")  # exact value 16.0%
    assert claim.status == "VERIFIED"
    claim, _ = _one(registry, "Revenue was $133 billion in FY2025.")  # 133.1 rounded to integer billions
    assert claim.status == "VERIFIED"


def test_incorrect_claim_with_explicit_period_is_contradicted(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Net income was $20 billion in FY2025.")
    assert claim.status == "CONTRADICTED"
    assert claim.expected_display == "14.000B" and "FY2025" in claim.reason


def test_mismatch_without_period_is_unsupported_not_false(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Revenue was $90 billion.")
    assert claim.status == "UNSUPPORTED"
    assert "not treated as a contradiction" in claim.reason


def test_value_from_other_period_is_partially_verified(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Revenue was $110 billion.")  # FY2023 value, period not stated
    assert claim.status == "PARTIALLY_VERIFIED"


def test_approximate_figure_is_partially_verified(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Revenue was $135 billion in FY2025.")  # 1.4% off, beyond rounding
    assert claim.status == "PARTIALLY_VERIFIED"


def test_missing_source_record_is_unsupported(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Total debt was $60 billion in FY2025.")  # balance sheet never collected
    assert claim.status == "UNSUPPORTED" and claim.missing_evidence


def test_fabricated_citation_is_flagged(registry: EvidenceRegistry) -> None:
    claim, report = _one(registry, "Operating margin was 16% in FY2025 [EV-deadbeef].")
    assert "EV-deadbeef" in report.unresolved_citations
    assert claim.status != "VERIFIED" and "does not exist" in claim.reason


def test_citation_pins_the_compared_record(registry: EvidenceRegistry) -> None:
    fy2023 = next(r for r in registry.series("TEST", "operating_margin") if r.period_year == 2023)
    claim, _ = _one(registry, f"Operating margin was 16% [{fy2023.evidence_id}].")  # cited FY2023 record is 12%
    assert claim.status == "CONTRADICTED"


def test_growth_claim_uses_recomputed_growth(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Revenue grew 10% in FY2025.")
    assert claim.metric == "total_revenue_growth_yoy" and claim.status == "VERIFIED"
    claim, _ = _one(registry, "Net income rose 16.7% in FY2025.")  # never registered: recomputed from raw
    assert claim.status == "VERIFIED" and "recomputed" in (claim.calculation or "")


def test_direction_claims(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Operating cash flow declined in FY2025.")
    assert claim.category == "trend" and claim.status == "VERIFIED"
    claim, _ = _one(registry, "Operating margin contracted in FY2025.")
    assert claim.status == "CONTRADICTED"
    claim, _ = _one(registry, "Free cash flow improved.")  # up then down: depends on the period
    assert claim.status in {"PARTIALLY_VERIFIED", "CONTRADICTED"}


def test_qualitative_claim_is_not_verifiable(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Management execution and brand strength remain excellent.")
    assert claim.category == "qualitative" and claim.status == "NOT_VERIFIABLE"


def test_verdict_statement_checked_against_rule_based_verdict(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Overall, the hypothesis is not supported.", verdict="SUPPORTED")
    assert claim.category == "verdict" and claim.status == "CONTRADICTED"
    claim, _ = _one(registry, "Overall, the hypothesis is supported.", verdict="SUPPORTED")
    assert claim.status == "VERIFIED"


def test_hypothesis_framing_is_not_treated_as_a_claim(registry: EvidenceRegistry) -> None:
    claims = extract_claims({"narrative": "We tested whether margins could keep expanding."})
    assert not [c for c in claims if c.category == "trend"]


def test_repeated_sentences_are_verified_once(registry: EvidenceRegistry) -> None:
    sentence = "Revenue reached $133.1 billion in FY2025."
    report = verify_text({"a": sentence, "b": sentence}, registry, "TEST")
    assert len(report.claims) == 1


def test_negative_values_reported_as_losses(registry: EvidenceRegistry) -> None:
    claim, _ = _one(registry, "Capital expenditure was $8 billion in FY2025.")  # stored as -8B outflow
    assert claim.status == "VERIFIED"

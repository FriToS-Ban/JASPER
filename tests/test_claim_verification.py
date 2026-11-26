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


# Regression tests for misreads found when verifying real LLM narratives (6-case live benchmark).
def _statuses(registry: EvidenceRegistry, text: str, verdict: str = "PARTIALLY_SUPPORTED"):
    report = verify_text({"narrative": text}, registry, "TEST", verdict=verdict)
    return [(c.category, c.metric, c.status) for c in report.claims]


def test_growth_noun_is_not_a_direction_claim(registry: EvidenceRegistry) -> None:
    assert _statuses(registry, "Revenue growth data is unavailable for quarterly validation.") == []


def test_verdict_labels_with_underscores_and_component_statements(registry: EvidenceRegistry) -> None:
    claims = _statuses(registry, "The overall verdict is PARTIALLY_SUPPORTED, with the margin component contradicted.")
    assert claims == [("verdict", None, "VERIFIED")]
    assert _statuses(registry, "The hypothesis' central claim of slowing revenue growth is not supported by the data.") == []
    assert _statuses(registry, "Therefore, the hypothesis is not supported.") == [("verdict", None, "CONTRADICTED")]


def test_stated_year_range_is_compared_endpoint_to_endpoint(registry: EvidenceRegistry) -> None:
    assert _statuses(registry, "Operating margin rose from FY2022 to FY2025.") == [("trend", "operating_margin", "VERIFIED")]
    assert _statuses(registry, "Operating cash flow improved from FY2023 to FY2025.") == [("trend", "operating_cash_flow", "CONTRADICTED")]


def test_numbers_attach_to_the_metric_named_before_them_and_keep_their_sign(registry: EvidenceRegistry) -> None:
    text = ("Free cash flow fell from $9.5 billion in FY2023 to $7 billion in FY2024, a 26.3% drop despite "
            "an operating margin of 14% in FY2024.")
    claims = _statuses(registry, text)
    assert ("growth", "free_cash_flow_growth_yoy", "VERIFIED") in claims
    assert ("margin", "operating_margin", "VERIFIED") in claims
    assert all(status == "VERIFIED" for _, _, status in claims)
    assert _statuses(registry, "Operating margin expanded 6.0 percentage points from FY2022 to FY2025.") == [
        ("margin", "operating_margin_change", "VERIFIED")]


def test_bookkeeping_numbers_are_not_financial_claims(registry: EvidenceRegistry) -> None:
    assert _statuses(registry, "Primary evidence score for operating margin (G1) is 0.00.") == []


def test_direction_words_go_to_the_nearest_metric(registry: EvidenceRegistry) -> None:
    claims = _statuses(registry, "Rising revenue and falling free cash flow point to a divergence.")
    assert claims == [("trend", "total_revenue", "VERIFIED"), ("trend", "free_cash_flow", "VERIFIED")]

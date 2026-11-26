"""Hypothesis parsing, plan revision, assessment rules, verdict aggregation and branching decisions."""
from __future__ import annotations

import pytest

from hypothesis_agent.research import analytics
from hypothesis_agent.research.assessment import ExpectationAssessment, assess_plan, determine_verdict
from hypothesis_agent.research.critique import CritiqueIssue, RequiredAction, decide, merge_llm_critique
from hypothesis_agent.research.evidence import EvidenceRegistry
from hypothesis_agent.research.extraction import build_snapshot, extract_evidence
from hypothesis_agent.research.planner import build_plan, parse_hypothesis, revise_plan


def _exp(text: str):
    return [(e.metric, e.direction, e.weight) for e in parse_hypothesis(text)[0]]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Operating margins are improving but cash conversion is deteriorating.",
         [("operating_margin", "increase", "primary"), ("cash_conversion", "decrease", "primary")]),
        ("Revenue growth is strong, but rising debt weakens the thesis.",
         [("total_revenue_growth_yoy", "above", "primary"), ("total_debt", "increase", "primary")]),
        ("Margins are expanding as revenue scales.",
         [("operating_margin", "increase", "primary"), ("total_revenue", "increase", "secondary")]),
        ("Margins are not improving.", [("operating_margin", "decrease", "primary")]),
        ("The company appears overvalued relative to its historical earnings growth.", [("peg_computed", "overvalued", "primary")]),
        ("Revenue grows 10%", [("total_revenue_growth_yoy", "above", "primary")]),
    ],
)
def test_parse_hypothesis_directions(text, expected) -> None:
    assert _exp(text) == expected


def test_parse_windows_base_effect_and_unmeasurable() -> None:
    expectations, _ = parse_hypothesis("Profitability improved between FY2022 and FY2025.")
    assert {(e.window_start_year, e.window_end_year) for e in expectations} == {(2022, 2025)}
    expectations, _ = parse_hypothesis("Margins are recovering after the 2025 trough.")
    assert expectations[0].window_start_year == 2025
    expectations, _ = parse_hypothesis("2024 revenue growth is explained by a weak 2023 comparison base.")
    assert expectations[0].direction == "base_effect" and expectations[0].target_year == 2024
    expectations, meta = parse_hypothesis("The streaming segment drives operating income growth.")
    assert expectations[0].metric is None and not expectations[0].measurable and "streaming" in meta["unmeasurable"]


def test_llm_plan_proposals_are_validated_and_cannot_replace_parsed_expectations() -> None:
    proposal = {"expectations": [
        {"concept": "margins", "metric": "operating_margin", "direction": "decrease"},  # conflicts: ignored
        {"concept": "liquidity", "metric": "current_ratio", "direction": "increase"},
        {"concept": "made up", "metric": "alpha_score", "direction": "increase"},
        {"concept": "bad", "metric": "net_margin", "direction": "sideways"},
    ], "additional_metrics": ["interest_coverage", "not_a_metric"]}
    plan = build_plan("Operating margins are improving.", "TEST", proposal)
    by_metric = {e.metric: e for e in plan.expectations}
    assert by_metric["operating_margin"].direction == "increase" and by_metric["operating_margin"].source == "rule_parser"
    assert by_metric["current_ratio"].source == "llm_planner" and by_metric["current_ratio"].weight == "secondary"
    assert "alpha_score" not in by_metric and "net_margin" not in by_metric
    assert any("rejected" in note for note in plan.notes)
    assert plan.has_metric("interest_coverage") and not plan.has_metric("not_a_metric")
    assert "balance_sheet_annual" in plan.datasets  # current_ratio needs the balance sheet


def test_plan_revision_is_recorded_with_trigger() -> None:
    plan = build_plan("Operating margins are improving.", "TEST")
    before = list(plan.datasets)
    revision = revise_plan(plan, iteration=1, reason="Explain cash divergence", triggered_by=["I1.1"],
                           metrics=[("capex_intensity", "annual"), ("operating_margin", "quarterly")])
    assert plan.version == 2 and revision.triggered_by == ["I1.1"]
    assert "income_statement_quarterly" in plan.datasets and "income_statement_quarterly" not in before
    assert revise_plan(plan, iteration=2, reason="again", triggered_by=[], metrics=[("capex_intensity", "annual")]) is None


def _registry(values, field="Free Cash Flow", slug="YFINANCE_CASH_FLOW") -> EvidenceRegistry:
    registry = EvidenceRegistry()
    payload = [{"date": f"{2022 + i}-12-31", field: v} for i, v in enumerate(values)]
    snapshot = build_snapshot(ticker="TEST", tool_slug=slug, parameters={"symbol": "TEST"}, payload=payload, retrieval_function="f")
    registry.register_snapshot(snapshot)
    extract_evidence(registry, snapshot, payload)
    return registry


@pytest.mark.parametrize(
    "values, hypothesis, status",
    [
        ([1, 2, 3, 4], "Free cash flow has been improving.", "SUPPORTS"),
        ([4, 3, 2, 1], "Free cash flow has been improving.", "CONTRADICTS"),
        ([1, 5, 9, 6], "Free cash flow has been improving.", "MIXED_POSITIVE"),
        ([9, 5, 1, 3], "Free cash flow has been improving.", "MIXED_NEGATIVE"),
        ([9, 5, 1, 3], "Free cash flow improved between 2022 and 2025.", "CONTRADICTS"),  # explicit window: endpoints decide
        ([10, 10.1, 9.95, 10.05], "Free cash flow has been improving.", "NEUTRAL"),
    ],
)
def test_assessment_statuses(values, hypothesis, status) -> None:
    registry = _registry(values)
    plan = build_plan(hypothesis, "TEST")
    assessment = assess_plan(plan, registry)[0]
    assert assessment.status == status
    if status.startswith("MIXED"):
        assert assessment.caveats


def test_valuation_assessment_handles_undefined_peg() -> None:
    registry = _registry([-1.0, -2.0, -1.5, -0.5], field="Diluted EPS", slug="YFINANCE_FINANCIALS")
    analytics.compute_valuation(registry, "TEST")
    plan = build_plan("TEST is overvalued relative to its earnings growth.", "TEST")
    assessment = assess_plan(plan, registry)[0]
    assert assessment.status == "INSUFFICIENT"
    assert determine_verdict([assessment]).verdict == "INSUFFICIENT_EVIDENCE"


def _a(status: str, group: str = "G1", weight: str = "primary", metric: str = "m") -> ExpectationAssessment:
    return ExpectationAssessment(expectation_id="H", group=group, concept="c", metric=metric, direction="increase",
                                 weight=weight, status=status, summary="s")


def test_verdict_aggregation_rules() -> None:
    assert determine_verdict([_a("SUPPORTS"), _a("SUPPORTS", "G2")]).verdict == "SUPPORTED"
    assert determine_verdict([_a("CONTRADICTS"), _a("CONTRADICTS", "G2")]).verdict == "NOT_SUPPORTED"
    assert determine_verdict([_a("SUPPORTS"), _a("CONTRADICTS", "G2")]).verdict == "PARTIALLY_SUPPORTED"
    assert determine_verdict([_a("SUPPORTS"), _a("CONTRADICTS", "G2", "secondary")]).verdict == "PARTIALLY_SUPPORTED"
    assert determine_verdict([_a("INSUFFICIENT", metric=None), _a("SUPPORTS", "G2")]).verdict == "INSUFFICIENT_EVIDENCE"
    assert determine_verdict([]).verdict == "INSUFFICIENT_EVIDENCE"
    # Composite concepts are averaged within their group first.
    assert determine_verdict([_a("SUPPORTS"), _a("SUPPORTS"), _a("CONTRADICTS", "G2")]).primary_score == 0.5


def _decide(**overrides):
    params = dict(issues=[], actions=[], assessments=[_a("SUPPORTS")], iteration=1, max_iterations=3,
                  reanalysis_count=0, max_reanalysis=1, new_evidence_last_iteration=0, attempted_actions=[])
    params.update(overrides)
    return decide(**params)


def test_decision_policy_branches() -> None:
    assert _decide()[0] == "CONTINUE"
    collect = RequiredAction(action="collect_dataset", rationale="need cash flow", datasets=["cash_flow_annual"])
    assert _decide(actions=[collect])[0] == "COLLECT_MORE_DATA"
    widen = RequiredAction(action="add_metrics", rationale="drill down", metrics=[("capex_intensity", "annual")])
    decision, _, to_run = _decide(actions=[widen])
    assert decision == "REVISE_PLAN" and to_run == [widen]
    inconsistent = CritiqueIssue(issue_id="I1", type="INCONSISTENT_CALCULATION", severity="high", description="x")
    assert _decide(issues=[inconsistent])[0] == "REANALYZE"
    assert _decide(issues=[inconsistent], reanalysis_count=1)[0] == "HUMAN_REVIEW"
    conflict = CritiqueIssue(issue_id="I2", type="SOURCE_CONFLICT", severity="high", description="sources disagree")
    assert _decide(issues=[conflict])[0] == "HUMAN_REVIEW"
    assert _decide(assessments=[_a("INSUFFICIENT", metric=None)])[0] == "INSUFFICIENT_EVIDENCE"


def test_decision_policy_bounds() -> None:
    widen = RequiredAction(action="add_metrics", rationale="drill down", metrics=[("capex_intensity", "annual")])
    decision, rationale, _ = _decide(actions=[widen], iteration=3, max_iterations=3)
    assert decision == "CONTINUE" and "Iteration limit" in rationale
    decision, _, _ = _decide(actions=[widen], iteration=3, max_iterations=3, assessments=[_a("INSUFFICIENT")])
    assert decision == "INSUFFICIENT_EVIDENCE"
    # Diminishing returns: the previous iteration produced nothing new, so do not loop again.
    decision, rationale, _ = _decide(actions=[widen], iteration=2, new_evidence_last_iteration=0)
    assert decision == "CONTINUE" and "diminishing" in rationale
    # Actions already attempted are not repeated.
    from hypothesis_agent.research.critique import action_key
    assert _decide(actions=[widen], attempted_actions=[action_key(widen)])[0] == "CONTINUE"


def test_llm_critique_is_validated() -> None:
    plan = build_plan("Operating margins are improving.", "TEST")
    issues, actions = [], []
    notes = merge_llm_critique({"issues": [
        {"severity": "high", "description": "Vague worry", "metrics": ["vibes"]},
        {"severity": "high", "description": "Margin gains rely on one-offs", "metrics": ["operating_margin"]},
    ], "additional_metrics": ["capex_intensity", "unicorn_index"]}, issues, actions, plan, 1)
    assert [i.severity for i in issues] == ["low", "high"]  # cannot escalate without naming real metrics
    assert actions and actions[0].metrics == [("capex_intensity", "annual")]
    assert "unicorn_index" in notes
    assert merge_llm_critique({"issues": "not a list"}, [], [], plan, 1).startswith("LLM critique rejected")

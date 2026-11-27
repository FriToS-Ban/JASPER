"""Structured critique of an analysis iteration and the branching decision that follows.

Deterministic checks run first and are authoritative. An optional LLM critic can *add* issues
and request extra metrics, but its requests are validated against the metric catalogue and it
cannot downgrade a deterministic finding (e.g. it cannot clear an inconsistent calculation).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from hypothesis_agent.research.assessment import ExpectationAssessment, VerdictResult
from hypothesis_agent.research.evidence import EvidenceRegistry, PeriodType
from hypothesis_agent.research.metrics import (
    dataset_for_metric,
    get_metric,
    is_known_metric,
    metric_category,
    raw_dependencies,
)
from hypothesis_agent.research.planner import ResearchPlan

Decision = Literal["CONTINUE", "COLLECT_MORE_DATA", "REVISE_PLAN", "REANALYZE", "HUMAN_REVIEW", "INSUFFICIENT_EVIDENCE"]
IssueType = Literal[
    "MISSING_METRIC", "DATA_UNAVAILABLE", "CONTRADICTORY_EVIDENCE", "INCONSISTENT_CALCULATION",
    "INSUFFICIENT_PERIODS", "TIME_PERIOD_SENSITIVITY", "UNSUPPORTED_ASSUMPTION", "SOURCE_CONFLICT",
    "IRRELEVANT_METRIC", "METRIC_NOT_MEANINGFUL", "LLM_CRITIC",
]


class CritiqueIssue(BaseModel):
    issue_id: str
    type: IssueType
    severity: Literal["low", "medium", "high"] = "medium"
    description: str
    metrics: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    source: Literal["deterministic", "llm"] = "deterministic"


class RequiredAction(BaseModel):
    action: Literal["collect_dataset", "add_metrics", "recompute"]
    rationale: str
    metrics: List[Tuple[str, str]] = Field(default_factory=list, description="(metric, frequency) pairs")
    datasets: List[str] = Field(default_factory=list)
    triggered_by: List[str] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    iteration: int
    status: Decision
    issues: List[CritiqueIssue] = Field(default_factory=list)
    required_actions: List[RequiredAction] = Field(default_factory=list)
    rationale: str
    llm_notes: Optional[str] = None


# Drill-downs: when a pattern appears in the evidence, the plan must widen to explain it.
CASH_FLOW_DRILLDOWN = ["operating_cash_flow", "capital_expenditure", "capex_intensity", "cash_conversion",
                       "change_in_working_capital", "receivables_days", "free_cash_flow_computed"]
LEVERAGE_FALLBACK = ["debt_to_ebitda", "net_debt", "interest_coverage"]
EARNINGS_QUALITY = ["pretax_income", "operating_income", "operating_margin"]


def _trend(assessments: Sequence[ExpectationAssessment], metric: str) -> Optional[ExpectationAssessment]:
    return next((a for a in assessments if a.metric == metric and "overall" in a.details), None)


def _series_direction(registry: EvidenceRegistry, ticker: str, metric: str) -> Optional[Tuple[str, str]]:
    from hypothesis_agent.research.analytics import classify_steps

    series = registry.series(ticker, metric, PeriodType.ANNUAL)
    if len(series) < 2:
        return None
    unit = series[0].unit
    steps, overall = classify_steps([r.value for r in series], unit in {"ratio", "x", "days"})
    return overall, steps[-1]


def deterministic_critique(
    *,
    plan: ResearchPlan,
    registry: EvidenceRegistry,
    assessments: List[ExpectationAssessment],
    verdict: VerdictResult,
    consistency_problems: List[Dict[str, Any]],
    collected_datasets: List[str],
    iteration: int,
) -> Tuple[List[CritiqueIssue], List[RequiredAction]]:
    ticker = plan.ticker
    issues: List[CritiqueIssue] = []
    actions: List[RequiredAction] = []

    def issue(kind: IssueType, description: str, *, severity: str = "medium", metrics: Sequence[str] = (),
              evidence_ids: Sequence[str] = ()) -> str:
        issue_id = f"I{iteration}.{len(issues) + 1}"
        issues.append(CritiqueIssue(issue_id=issue_id, type=kind, severity=severity, description=description,
                                    metrics=list(metrics), evidence_ids=list(evidence_ids)))
        return issue_id

    # 1. Calculations must reproduce from their inputs.
    for problem in consistency_problems:
        iid = issue("INCONSISTENT_CALCULATION", f"{problem['metric']} ({problem['evidence_id']}) does not reproduce from its inputs: {problem}.",
                    severity="high", metrics=[problem["metric"]], evidence_ids=[problem["evidence_id"]])
        actions.append(RequiredAction(action="recompute", rationale="Discard and recompute inconsistent derived values.",
                                      triggered_by=[iid]))

    # 2. Planned metrics must exist; distinguish "not yet collected" from "unavailable at source".
    available = set(registry.metrics_available(ticker))
    for planned in plan.metrics:
        if planned.frequency != "annual" or planned.metric in available:
            continue
        raws = raw_dependencies(planned.metric)
        uncollected = [d for d in (dataset_for_metric(r) for r in raws) if d and d not in collected_datasets]
        if uncollected:
            iid = issue("MISSING_METRIC", f"{planned.metric} not evaluated; dataset(s) {sorted(set(uncollected))} not yet collected.",
                        metrics=[planned.metric])
            actions.append(RequiredAction(action="collect_dataset", rationale=f"Collect data needed for {planned.metric}.",
                                          datasets=sorted(set(uncollected)), triggered_by=[iid]))
        elif planned.origin != "baseline":
            reasons = [m.reason for m in registry.missing if m.ticker == ticker and m.metric in {planned.metric, *raws}]
            issue("DATA_UNAVAILABLE", f"{planned.metric} could not be computed: {reasons[-1] if reasons else 'inputs unavailable'}",
                  severity="high" if planned.origin == "expectation" else "low", metrics=[planned.metric])

    # 3. Contradictory signals force a wider investigation.
    profit_up = [m for m in ("operating_margin", "net_income", "total_revenue")
                 if (d := _series_direction(registry, ticker, m)) and d[0] == "increasing"]
    cash_down = [m for m in ("free_cash_flow", "cash_conversion")
                 if (d := _series_direction(registry, ticker, m)) and (d[0] == "decreasing" or d[1] == "decreasing")]
    if profit_up and cash_down:
        iid = issue("CONTRADICTORY_EVIDENCE",
                    f"{', '.join(profit_up)} trend up while {', '.join(cash_down)} deteriorated (overall or in the latest period); "
                    "accounting profit and cash generation diverge.", severity="high", metrics=profit_up + cash_down)
        missing = [m for m in CASH_FLOW_DRILLDOWN if not plan.has_metric(m)]
        if missing:
            actions.append(RequiredAction(action="add_metrics", triggered_by=[iid],
                                          rationale="Explain the profit/cash divergence: capex, working capital, receivables and cash conversion.",
                                          metrics=[(m, "annual") for m in missing]))

    # 4. Late reversals: check the most recent quarters before concluding.
    for a in assessments:
        if a.status in {"MIXED_POSITIVE", "MIXED_NEGATIVE"} and a.metric and a.weight == "primary":
            iid = issue("TIME_PERIOD_SENSITIVITY", f"{a.expectation_id}: {a.caveats[0] if a.caveats else a.summary}",
                        metrics=[a.metric], evidence_ids=a.evidence_ids)
            base = a.metric[:-len('_growth_yoy')] if a.metric.endswith("_growth_yoy") else a.metric
            if not plan.has_metric(base, "quarterly") and dataset_for_metric(raw_dependencies(base)[0] if raw_dependencies(base) else base, "quarterly"):
                actions.append(RequiredAction(action="add_metrics", triggered_by=[iid],
                                              rationale=f"Check whether the latest-period reversal in {base} persists in recent quarters.",
                                              metrics=[(base, "quarterly")]))

    # 5. Earnings quality: positive net income with an operating loss.
    ni = registry.series(ticker, "net_income", PeriodType.ANNUAL)
    oi = registry.series(ticker, "operating_income", PeriodType.ANNUAL)
    if ni and oi and ni[-1].period_end == oi[-1].period_end and ni[-1].value > 0 > oi[-1].value:
        iid = issue("UNSUPPORTED_ASSUMPTION",
                    f"Net income is positive ({ni[-1].period}) while operating income is negative: earnings likely include "
                    "non-operating or one-time gains, so net income overstates underlying profitability.",
                    severity="high", metrics=["net_income", "operating_income"], evidence_ids=[ni[-1].evidence_id, oi[-1].evidence_id])
        missing = [m for m in EARNINGS_QUALITY if not plan.has_metric(m)]
        if missing:
            actions.append(RequiredAction(action="add_metrics", triggered_by=[iid], rationale="Separate operating from non-operating earnings.",
                                          metrics=[(m, "annual") for m in missing]))

    # 6. Negative equity makes equity-based leverage meaningless.
    equity = registry.series(ticker, "stockholders_equity", PeriodType.ANNUAL)
    if equity and equity[-1].value < 0:
        iid = issue("METRIC_NOT_MEANINGFUL", "Stockholders' equity is negative; debt-to-equity cannot measure leverage.",
                    metrics=["debt_to_equity", "stockholders_equity"], evidence_ids=[equity[-1].evidence_id])
        missing = [m for m in LEVERAGE_FALLBACK if not plan.has_metric(m)]
        if missing:
            actions.append(RequiredAction(action="add_metrics", triggered_by=[iid], rationale="Use debt/EBITDA and net debt instead.",
                                          metrics=[(m, "annual") for m in missing]))

    # 7. Too few periods for a trend claim.
    for a in assessments:
        if a.status == "INSUFFICIENT" and a.metric and "trend cannot be established" in a.summary:
            iid = issue("INSUFFICIENT_PERIODS", a.summary, metrics=[a.metric])
            if not plan.has_metric(a.metric, "quarterly"):
                actions.append(RequiredAction(action="add_metrics", triggered_by=[iid],
                                              rationale="Use quarterly data to obtain more observations.",
                                              metrics=[(a.metric, "quarterly")]))

    # 8. Source conflicts: Yahoo TTM ratios vs ratios computed from statements.
    for derived, ttm in (("operating_margin", "ttm_operating_margin"), ("net_margin", "ttm_profit_margin")):
        fy = registry.latest(ticker, derived, PeriodType.ANNUAL)
        reported = registry.latest(ticker, ttm)
        if fy and reported and abs(fy.value - reported.value) > 0.10:
            issue("SOURCE_CONFLICT",
                  f"{get_metric(derived).label} {fy.period} {fy.value:.1%} differs from Yahoo {reported.label} {reported.value:.1%} by "
                  f"more than 10 points; the periods differ (FY vs TTM) but the gap is large.",
                  severity="medium", metrics=[derived, ttm], evidence_ids=[fy.evidence_id, reported.evidence_id])
    for conflict in registry.conflicts:
        issue("SOURCE_CONFLICT", conflict.description, severity="high", metrics=[conflict.metric], evidence_ids=conflict.evidence_ids)

    # 9. Unmeasurable concepts.
    for a in assessments:
        if a.metric is None:
            issue("UNSUPPORTED_ASSUMPTION", f"{a.expectation_id}: {a.summary}", severity="high")
    return issues, actions


def decide(
    *,
    issues: List[CritiqueIssue],
    actions: List[RequiredAction],
    assessments: List[ExpectationAssessment],
    iteration: int,
    max_iterations: int,
    reanalysis_count: int,
    max_reanalysis: int,
    new_evidence_last_iteration: int,
    attempted_actions: List[str],
) -> Tuple[Decision, str, List[RequiredAction]]:
    """Choose the next branch. Returns (decision, rationale, actions to execute)."""

    fresh = [a for a in actions if _action_key(a) not in attempted_actions and a.action != "recompute"]
    primary = [a for a in assessments if a.weight == "primary"]
    unmeasurable_core = [a for a in primary if a.metric is None]
    insufficient_core = [a for a in primary if a.status == "INSUFFICIENT"]
    inconsistent = [i for i in issues if i.type == "INCONSISTENT_CALCULATION"]
    conflicts = [i for i in issues if i.type == "SOURCE_CONFLICT" and i.severity == "high"]
    llm_high = [i for i in issues if i.source == "llm" and i.severity == "high"]

    if inconsistent:
        if reanalysis_count < max_reanalysis:
            return "REANALYZE", f"{len(inconsistent)} calculation(s) did not reproduce; recomputing from raw evidence.", []
        return "HUMAN_REVIEW", "Calculations remained inconsistent after reanalysis; a reviewer must inspect them.", []
    if iteration >= max_iterations:
        if insufficient_core:
            return "INSUFFICIENT_EVIDENCE", (f"Iteration limit ({max_iterations}) reached with core expectations unresolved: "
                                             + ", ".join(a.expectation_id for a in insufficient_core)), []
        suffix = f" {len(fresh)} proposed follow-up(s) were not executed." if fresh else ""
        return "CONTINUE", f"Iteration limit ({max_iterations}) reached; reporting with documented gaps.{suffix}", []
    if fresh and (iteration == 1 or new_evidence_last_iteration > 0):
        adds_scope = any(a.action == "add_metrics" for a in fresh)
        decision: Decision = "REVISE_PLAN" if adds_scope else "COLLECT_MORE_DATA"
        return decision, "; ".join(a.rationale for a in fresh), fresh
    if fresh:
        return "CONTINUE", "Further actions were proposed but the previous iteration added no new evidence (diminishing returns).", []
    if unmeasurable_core or (insufficient_core and len(insufficient_core) == len(primary)):
        return "INSUFFICIENT_EVIDENCE", "The core claim depends on evidence that cannot be collected from available sources.", []
    if insufficient_core:
        return "INSUFFICIENT_EVIDENCE", "Some core expectations could not be evaluated with the available data.", []
    if conflicts or llm_high:
        reasons = [i.description for i in conflicts + llm_high]
        return "HUMAN_REVIEW", "High-impact ambiguity requires a reviewer: " + " | ".join(reasons[:3]), []
    return "CONTINUE", "Evidence covers every expectation; no outstanding actions.", []


def _action_key(action: RequiredAction) -> str:
    metrics = ",".join(f"{m}:{f}" for m, f in sorted(action.metrics))
    return f"{action.action}|{metrics}|{','.join(sorted(action.datasets))}"


def action_key(action: RequiredAction) -> str:
    return _action_key(action)


class _LLMIssue(BaseModel):
    type: str = "LLM_CRITIC"
    severity: Literal["low", "medium", "high"] = "medium"
    description: str
    metrics: List[str] = Field(default_factory=list)


class _LLMCritique(BaseModel):
    issues: List[_LLMIssue] = Field(default_factory=list)
    additional_metrics: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


def merge_llm_critique(
    payload: Optional[Dict[str, Any]],
    issues: List[CritiqueIssue],
    actions: List[RequiredAction],
    plan: ResearchPlan,
    iteration: int,
) -> Optional[str]:
    """Validate and merge LLM critic output. Returns notes (including rejections) for the record."""

    if not payload:
        return None
    try:
        critique = _LLMCritique.model_validate(payload)
    except ValidationError as exc:
        return f"LLM critique rejected (schema): {exc.errors(include_url=False)[0]['msg']}"
    notes = [critique.notes] if critique.notes else []
    for item in critique.issues[:5]:
        known = [m for m in item.metrics if is_known_metric(m)]
        severity = item.severity if known else "low"  # cannot escalate without naming real metrics
        issues.append(CritiqueIssue(issue_id=f"I{iteration}.{len(issues) + 1}", type="LLM_CRITIC", severity=severity,
                                    description=item.description[:500], metrics=known, source="llm"))
    accepted = [m for m in critique.additional_metrics if is_known_metric(m) and not plan.has_metric(m)]
    rejected = [m for m in critique.additional_metrics if not is_known_metric(m)]
    relevant = [m for m in accepted if metric_category(m) in set(plan.categories()) | {"cash_flow", "profitability", "growth", "leverage"}]
    if relevant:
        actions.append(RequiredAction(action="add_metrics", rationale="LLM critic requested additional metrics.",
                                      metrics=[(m, "annual") for m in relevant[:6]], triggered_by=["llm"]))
    if rejected:
        notes.append(f"Rejected unknown metrics from LLM critic: {rejected}")
    return " ".join(notes) or None

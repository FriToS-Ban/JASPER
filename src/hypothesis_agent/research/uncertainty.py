"""Structured uncertainty assessment.

JASPER deliberately does not emit a 0-100 confidence number. Instead it reports what was asked
for versus what was obtained, how the evidence splits, how trustworthy the source is, and what
the analysis cannot rule out. Every item is derived from recorded facts about the run.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hypothesis_agent.research.assessment import ExpectationAssessment, evidence_balance
from hypothesis_agent.research.evidence import EvidenceRegistry, PeriodType
from hypothesis_agent.research.metrics import get_metric
from hypothesis_agent.research.planner import ResearchPlan

SOURCE_LIMITATIONS = [
    "Yahoo Finance data are unaudited aggregations of company filings; field definitions are not published and can change.",
    "yfinance is an unofficial client; Yahoo may revise or drop historical values, so live re-runs can differ from stored snapshots.",
    "Annual statements typically cover only the last four fiscal years; older periods are unavailable.",
    "Company-info ratios (TTM margins, P/E, PEG) are computed by Yahoo with undisclosed methodology and are point-in-time.",
    "Consolidated statements provide no segment, geographic or product-level breakdown.",
]


class DataCompleteness(BaseModel):
    metrics_requested: List[str]
    metrics_collected: List[str]
    metrics_unavailable: List[Dict[str, str]]
    coverage_ratio: float = Field(description="collected / requested; a count ratio, not a probability")
    missing_periods: List[str] = Field(default_factory=list)
    failed_datasets: List[Dict[str, str]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class SourceQuality(BaseModel):
    sources: List[str]
    limitations: List[str]
    retrieval: List[Dict[str, Any]]
    latest_period_end: Optional[str] = None
    staleness_days: Optional[int] = None
    inconsistencies: List[str] = Field(default_factory=list)


class UncertaintyAssessment(BaseModel):
    data_completeness: DataCompleteness
    evidence_balance: Dict[str, List[Dict[str, Any]]]
    source_quality: SourceQuality
    analytical_limitations: List[str]


def build_uncertainty(
    *,
    plan: ResearchPlan,
    registry: EvidenceRegistry,
    assessments: List[ExpectationAssessment],
    critique_issues: List[Dict[str, Any]],
    iteration_limit_hit: bool,
    reference_date: Optional[date] = None,
) -> UncertaintyAssessment:
    ticker = plan.ticker
    available = set(registry.metrics_available(ticker))
    requested = [m.metric for m in plan.metrics if m.frequency == "annual"]
    collected = [m for m in requested if m in available]
    unavailable = []
    for metric in requested:
        if metric in available:
            continue
        reasons = [m.reason for m in registry.missing if m.ticker == ticker and m.metric == metric]
        unavailable.append({"metric": metric, "label": _label(metric), "reason": reasons[-1] if reasons else "Not computed in this run."})
    missing_periods = sorted({m.period for m in registry.missing if m.ticker == ticker and m.metric == "*" and m.period})
    failed = [{"dataset": s.dataset, "error": s.error or "unknown"} for s in registry.snapshots.values()
              if s.ticker == ticker and s.status == "error"]
    completeness_limits = []
    annual_revenue = registry.series(ticker, "total_revenue", PeriodType.ANNUAL)
    if annual_revenue and len(annual_revenue) < 4:
        completeness_limits.append(f"Only {len(annual_revenue)} annual periods of revenue were available.")
    if failed:
        completeness_limits.append(f"{len(failed)} dataset retrieval(s) failed; see failed datasets.")

    period_ends = [r.period_end for r in registry.records.values()
                   if r.ticker == ticker and r.period_type in {PeriodType.ANNUAL, PeriodType.QUARTERLY} and r.period_end]
    latest = max(period_ends) if period_ends else None
    today = reference_date or date.today()
    staleness = (today - datetime.strptime(latest, "%Y-%m-%d").date()).days if latest else None
    retrieval = [
        {"snapshot_id": s.snapshot_id, "dataset": s.dataset, "function": s.retrieval_function,
         "retrieved_at": s.retrieved_at, "mode": s.source_mode, "status": s.status}
        for s in registry.snapshots.values() if s.ticker == ticker
    ]
    inconsistencies = [c.description for c in registry.conflicts if c.ticker == ticker]
    inconsistencies += [i["description"] for i in critique_issues if i.get("type") == "SOURCE_CONFLICT"]
    sources = sorted({s.source for s in registry.snapshots.values() if s.ticker == ticker})
    source_limits = list(SOURCE_LIMITATIONS)
    if any(s.source_mode == "frozen_fixture" for s in registry.snapshots.values()):
        source_limits.append("Data were replayed from frozen fixtures; they reflect Yahoo's values at capture time, not today.")
    if any(s.source_mode == "custom_toolset" for s in registry.snapshots.values()):
        source_limits.append("Some data came from a custom toolset whose provenance JASPER cannot verify.")
    if staleness is not None and staleness > 200:
        source_limits.append(f"Latest financial period ended {latest}, {staleness} days before the report date.")

    limitations = [
        "Associations between metrics are descriptive; no causal controls (peers, macro, FX, M&A) were applied.",
        "Trend classifications use fixed thresholds (2% relative for amounts, 0.5 percentage points for ratios).",
    ]
    if plan.forward_looking:
        limitations.append("The hypothesis is forward-looking; historical data cannot validate a forecast.")
    for issue in critique_issues:
        # Time-period sensitivity is reported through the assessment caveats below.
        if issue.get("type") in {"UNSUPPORTED_ASSUMPTION", "METRIC_NOT_MEANINGFUL", "CONTRADICTORY_EVIDENCE"}:
            limitations.append(issue["description"])
    for assessment in assessments:
        for caveat in assessment.caveats:
            text = f"{assessment.expectation_id} ({assessment.metric or assessment.concept}): {caveat}"
            if not any(caveat in existing for existing in limitations):
                limitations.append(text)
    if iteration_limit_hit:
        limitations.append("The research loop reached its iteration limit; proposed follow-up analyses were not executed.")
    return UncertaintyAssessment(
        data_completeness=DataCompleteness(
            metrics_requested=requested,
            metrics_collected=collected,
            metrics_unavailable=unavailable,
            coverage_ratio=round(len(collected) / len(requested), 4) if requested else 0.0,
            missing_periods=missing_periods,
            failed_datasets=failed,
            limitations=completeness_limits,
        ),
        evidence_balance=evidence_balance(assessments),
        source_quality=SourceQuality(
            sources=sources, limitations=source_limits, retrieval=retrieval, latest_period_end=latest,
            staleness_days=staleness, inconsistencies=list(dict.fromkeys(inconsistencies)),
        ),
        analytical_limitations=list(dict.fromkeys(limitations)),
    )


def _label(metric: str) -> str:
    try:
        return get_metric(metric).label
    except KeyError:
        return metric

"""Deterministic evaluation of hypothesis expectations and the resulting verdict.

The verdict is computed from evidence by fixed rules, never by the LLM. The narrative must
explain this verdict; claim verification flags narratives that state a different one.
"""
from __future__ import annotations

from statistics import fmean
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from hypothesis_agent.research.analytics import classify_steps, short_period
from hypothesis_agent.research.evidence import EvidenceRecord, EvidenceRegistry, PeriodType
from hypothesis_agent.research.metrics import get_metric
from hypothesis_agent.research.planner import Expectation, ResearchPlan

Status = Literal["SUPPORTS", "CONTRADICTS", "MIXED_POSITIVE", "MIXED_NEGATIVE", "NEUTRAL", "INSUFFICIENT"]
Verdict = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"]

STATUS_SCORE: Dict[str, float] = {
    "SUPPORTS": 1.0,
    "MIXED_POSITIVE": 0.65,
    "NEUTRAL": 0.5,
    "MIXED_NEGATIVE": 0.35,
    "CONTRADICTS": 0.0,
}
PEG_RICH = 2.0
PEG_CHEAP = 1.0


class ExpectationAssessment(BaseModel):
    expectation_id: str
    group: str
    concept: str
    metric: Optional[str]
    direction: str
    weight: str
    status: Status
    evidence_ids: List[str] = Field(default_factory=list)
    window: Optional[str] = None
    summary: str
    caveats: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class VerdictResult(BaseModel):
    verdict: Verdict
    rationale: str
    primary_score: Optional[float] = None
    group_scores: Dict[str, float] = Field(default_factory=dict)
    counts: Dict[str, int] = Field(default_factory=dict)


def _window_bounds(expectation: Expectation, series: List[EvidenceRecord]) -> Tuple[Optional[int], Optional[int]]:
    start, end = expectation.window_start_year, expectation.window_end_year
    if expectation.window_last_n_years and series:
        latest = series[-1].period_year or 0
        start, end = latest - expectation.window_last_n_years, latest
    return start, end


def _windowed(series: List[EvidenceRecord], start: Optional[int], end: Optional[int]) -> List[EvidenceRecord]:
    return [r for r in series if (start is None or (r.period_year or 0) >= start) and (end is None or (r.period_year or 0) <= end)]


def _fmt(value: float, unit: str) -> str:
    if unit in {"ratio", "ratio_points"}:
        return f"{value * 100:.1f}%" if unit == "ratio" else f"{value * 100:+.1f}pp"
    if unit == "x":
        return f"{value:.2f}x"
    if abs(value) >= 1e9:
        return f"{value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.1f}M"
    return f"{value:,.2f}"


def assess_expectation(expectation: Expectation, registry: EvidenceRegistry, ticker: str) -> ExpectationAssessment:
    base = dict(expectation_id=expectation.expectation_id, group=expectation.group, concept=expectation.concept,
                metric=expectation.metric, direction=expectation.direction, weight=expectation.weight)
    if not expectation.measurable or expectation.metric is None:
        return ExpectationAssessment(**base, status="INSUFFICIENT", summary=expectation.rationale,
                                     caveats=["No structured data source covers this concept."])
    if expectation.direction in {"overvalued", "undervalued"}:
        return _assess_valuation(expectation, registry, ticker, base)
    if expectation.direction == "base_effect":
        return _assess_base_effect(expectation, registry, ticker, base)

    series = registry.series(ticker, expectation.metric, PeriodType.ANNUAL)
    if not series and expectation.metric in {"price_return", "annualized_volatility", "max_drawdown"}:
        series = registry.find(ticker=ticker, metric=expectation.metric)
    start, end = _window_bounds(expectation, series)
    window_series = _windowed(series, start, end)
    label = get_metric(expectation.metric).label
    if not window_series:
        reasons = [m.reason for m in registry.missing if m.ticker == ticker and m.metric in {expectation.metric, "*"}]
        return ExpectationAssessment(**base, status="INSUFFICIENT",
                                     summary=f"No {label} evidence available for the window.",
                                     caveats=reasons[:3] or ["Metric not collected."])
    unit = window_series[0].unit
    window_text = f"{short_period(window_series[0])} to {short_period(window_series[-1])}"

    if expectation.direction in {"above", "below", "positive", "negative"}:
        latest = window_series[-1]
        threshold = expectation.threshold if expectation.threshold is not None else 0.0
        if expectation.direction in {"above", "positive"}:
            ok = latest.value > threshold
        else:
            ok = latest.value < threshold
        status: Status = "SUPPORTS" if ok else "CONTRADICTS"
        relation = ">" if expectation.direction in {"above", "positive"} else "<"
        caveats = []
        earlier_fail = [r for r in window_series[:-1] if (r.value > threshold) != (relation == ">")]
        if earlier_fail and ok:
            caveats.append(f"Condition did not hold in {', '.join(short_period(r) for r in earlier_fail)}.")
        return ExpectationAssessment(
            **base, status=status, evidence_ids=[latest.evidence_id], window=short_period(latest),
            summary=(f"{label} was {_fmt(latest.value, unit)} in {short_period(latest)} "
                     f"(test: {relation} {_fmt(threshold, unit if unit != 'ratio_points' else 'ratio')})."),
            caveats=caveats,
            details={"latest": latest.value, "threshold": threshold},
        )

    if len(window_series) < 2:
        return ExpectationAssessment(**base, status="INSUFFICIENT", evidence_ids=[r.evidence_id for r in window_series],
                                     window=window_text,
                                     summary=f"Only one period of {label} in the window; a trend cannot be established.")
    is_ratio = unit in {"ratio", "x", "days", "ratio_points"}
    values = [r.value for r in window_series]
    steps, overall = classify_steps(values, is_ratio)
    wanted = "increasing" if expectation.direction == "increase" else "decreasing"
    opposite = "decreasing" if wanted == "increasing" else "increasing"
    latest_step = steps[-1]
    caveats: List[str] = []
    if expectation.has_explicit_window:
        # The hypothesis names its window, so endpoints decide; a late reversal is a caveat.
        status = "SUPPORTS" if overall == wanted else ("CONTRADICTS" if overall == opposite else "NEUTRAL")
        if latest_step not in {overall, "flat"} and overall != "flat":
            caveats.append(f"The most recent period moved {latest_step}, against the window's overall direction.")
    elif overall == wanted:
        status = "MIXED_POSITIVE" if latest_step == opposite else "SUPPORTS"
        if status == "MIXED_POSITIVE":
            caveats.append(f"Direction holds over {window_text}, but the latest period was {latest_step}: conclusion depends on the period chosen.")
    elif overall == opposite:
        status = "MIXED_NEGATIVE" if latest_step == wanted else "CONTRADICTS"
        if status == "MIXED_NEGATIVE":
            caveats.append(f"The latest period was {latest_step}, but {window_text} as a whole was {overall}.")
    else:
        status = "NEUTRAL"
    if latest_step == "flat" and overall != "flat" and len(steps) > 1:
        caveats.append(f"The latest period was flat after a {overall} path, which may indicate stabilisation.")
    earlier = [r for r in series if start is not None and (r.period_year or 0) < start]
    if earlier:
        reference = max(earlier, key=lambda r: r.value) if wanted == "increasing" else min(earlier, key=lambda r: r.value)
        relation = "below" if window_series[-1].value < reference.value else "above"
        caveats.append(
            f"Before the stated window, {label} was {_fmt(reference.value, unit)} in {short_period(reference)}; "
            f"the latest value remains {relation} that level."
        )
    trend = next((r for r in registry.find(ticker=ticker, metric=f"{expectation.metric}_trend")
                  if r.period_start == window_series[0].period_end and r.period_end == window_series[-1].period_end), None)
    evidence_ids = [r.evidence_id for r in window_series] + ([trend.evidence_id] if trend else [])
    if status in {"MIXED_POSITIVE", "MIXED_NEGATIVE"}:
        quarterly = [r for r in registry.find(ticker=ticker, metric=f"{expectation.metric}_trend")
                     if r.derivation and r.derivation.parameters.get("frequency") == "quarterly"]
        if quarterly:
            q = quarterly[-1]
            caveats.append(f"Quarterly drill-down: {q.period} the metric was {q.qualifier} [{q.evidence_id}].")
            evidence_ids.append(q.evidence_id)
    path = " -> ".join(_fmt(v, unit) for v in values)
    return ExpectationAssessment(
        **base, status=status, evidence_ids=evidence_ids, window=window_text,
        summary=f"{label} moved {overall} over {window_text} ({path}); expected {wanted}.",
        caveats=caveats,
        details={"steps": steps, "overall": overall, "values": values},
    )


def _assess_valuation(expectation: Expectation, registry: EvidenceRegistry, ticker: str, base: Dict[str, Any]) -> ExpectationAssessment:
    readings: List[Tuple[str, EvidenceRecord]] = []
    for metric in ("peg_computed", "peg_ratio"):
        record = registry.latest(ticker, metric)
        if record is not None:
            readings.append((metric, record))
    if not readings:
        reasons = [m.reason for m in registry.missing if m.ticker == ticker and m.metric in {"peg_computed", "pe_computed", "diluted_eps_cagr", "peg_ratio"}]
        return ExpectationAssessment(
            **base, status="INSUFFICIENT",
            summary="Valuation relative to growth cannot be measured: no meaningful PEG ratio could be formed.",
            caveats=list(dict.fromkeys(reasons))[:4],
        )

    def classify(value: float) -> str:
        if value > PEG_RICH:
            return "overvalued"
        if value < PEG_CHEAP:
            return "undervalued"
        return "fair"

    classes = [classify(r.value) for _, r in readings]
    wanted = expectation.direction
    opposite = "undervalued" if wanted == "overvalued" else "overvalued"
    if all(c == wanted for c in classes):
        status: Status = "SUPPORTS"
    elif all(c == opposite for c in classes):
        status = "CONTRADICTS"
    elif wanted in classes:
        status = "MIXED_POSITIVE"
    elif opposite in classes:
        status = "MIXED_NEGATIVE"
    else:
        status = "NEUTRAL"
    caveats = [f"PEG thresholds are conventions (>{PEG_RICH:g} rich, <{PEG_CHEAP:g} cheap), not facts."]
    near = [r for _, r in readings if abs(r.value - PEG_RICH) < 0.25 or abs(r.value - PEG_CHEAP) < 0.15]
    if near:
        caveats.append("At least one PEG reading sits close to a threshold; the conclusion is threshold-sensitive.")
    if len(readings) == 1:
        caveats.append(f"Only {readings[0][0]} was available.")
    text = "; ".join(f"{r.label} {r.value:.2f}x ({classify(r.value)})" for _, r in readings)
    return ExpectationAssessment(**base, status=status, evidence_ids=[r.evidence_id for _, r in readings],
                                 summary=f"{text}; hypothesis expects {wanted}.", caveats=caveats,
                                 details={"readings": {m: r.value for m, r in readings}})


def _assess_base_effect(expectation: Expectation, registry: EvidenceRegistry, ticker: str, base: Dict[str, Any]) -> ExpectationAssessment:
    series = registry.series(ticker, "total_revenue", PeriodType.ANNUAL)
    by_year = {r.period_year: r for r in series}
    target = expectation.target_year or (series[-1].period_year if series else None)
    if target is None or target not in by_year or (target - 1) not in by_year or (target - 2) not in by_year:
        return ExpectationAssessment(**base, status="INSUFFICIENT",
                                     summary=f"Need revenue for FY{(target or 0) - 2}..FY{target} to test a base effect.")
    cur, prev, prev2 = by_year[target], by_year[target - 1], by_year[target - 2]
    growth = cur.value / prev.value - 1
    prior_growth = prev.value / prev2.value - 1
    peak = max(r.value for r in series if (r.period_year or 0) < target - 1)
    ids = [prev2.evidence_id, prev.evidence_id, cur.evidence_id]
    if growth > 0 and prior_growth < 0 and cur.value < peak:
        status: Status = "SUPPORTS"
        summary = (f"FY{target} revenue grew {growth:.1%} after falling {prior_growth:.1%} in FY{target - 1}, "
                   f"and remains {1 - cur.value / peak:.1%} below the earlier peak: growth is measured off a depressed base.")
    elif growth > 0 and prior_growth < 0:
        status = "MIXED_NEGATIVE"
        summary = (f"FY{target} growth of {growth:.1%} followed a {prior_growth:.1%} decline, but revenue now exceeds the "
                   "earlier peak, which points to a genuine recovery rather than only a base effect.")
    elif growth > 0:
        status = "CONTRADICTS"
        summary = f"FY{target} growth of {growth:.1%} followed growth of {prior_growth:.1%} in FY{target - 1}; there was no weak base."
    else:
        status = "NEUTRAL"
        summary = f"Revenue did not grow in FY{target} ({growth:.1%}); a base-effect explanation is moot."
    caveats = []
    following = by_year.get(target + 1)
    if following is not None:
        ids.append(following.evidence_id)
        caveats.append(f"FY{target + 1} revenue changed {following.value / cur.value - 1:+.1%} versus FY{target}.")
    return ExpectationAssessment(**base, status=status, evidence_ids=ids, window=f"FY{target - 2} to FY{target}",
                                 summary=summary, caveats=caveats,
                                 details={"growth": growth, "prior_growth": prior_growth, "peak": peak})


def assess_plan(plan: ResearchPlan, registry: EvidenceRegistry) -> List[ExpectationAssessment]:
    return [assess_expectation(e, registry, plan.ticker) for e in plan.expectations]


def determine_verdict(assessments: List[ExpectationAssessment]) -> VerdictResult:
    counts: Dict[str, int] = {}
    for a in assessments:
        counts[a.status] = counts.get(a.status, 0) + 1
    primary = [a for a in assessments if a.weight == "primary"]
    if not assessments or not primary:
        return VerdictResult(verdict="INSUFFICIENT_EVIDENCE", counts=counts,
                             rationale="The hypothesis contains no testable financial expectation.")
    insufficient_primary = [a for a in primary if a.status == "INSUFFICIENT"]
    if insufficient_primary:
        return VerdictResult(
            verdict="INSUFFICIENT_EVIDENCE", counts=counts,
            rationale=("Core expectation(s) could not be evaluated: "
                       + "; ".join(f"{a.expectation_id} ({a.concept}): {a.summary}" for a in insufficient_primary)),
        )
    groups: Dict[str, List[float]] = {}
    for a in primary:
        groups.setdefault(a.group, []).append(STATUS_SCORE[a.status])
    group_scores = {g: fmean(v) for g, v in groups.items()}
    score = fmean(group_scores.values())
    if score >= 0.8:
        verdict: Verdict = "SUPPORTED"
    elif score <= 0.2:
        verdict = "NOT_SUPPORTED"
    else:
        verdict = "PARTIALLY_SUPPORTED"
    secondary_contra = [a for a in assessments if a.weight == "secondary" and a.status == "CONTRADICTS"]
    rationale = (f"Primary evidence score {score:.2f} across {len(group_scores)} concept group(s) "
                 f"(SUPPORTS=1, MIXED_POSITIVE=0.65, NEUTRAL=0.5, MIXED_NEGATIVE=0.35, CONTRADICTS=0; "
                 f">=0.8 supported, <=0.2 not supported).")
    if verdict == "SUPPORTED" and secondary_contra:
        verdict = "PARTIALLY_SUPPORTED"
        rationale += " Downgraded because secondary evidence contradicts: " + ", ".join(a.expectation_id for a in secondary_contra) + "."
    return VerdictResult(verdict=verdict, rationale=rationale, primary_score=round(score, 4),
                         group_scores={k: round(v, 4) for k, v in group_scores.items()}, counts=counts)


def evidence_balance(assessments: List[ExpectationAssessment]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"supporting": [], "contradicting": [], "mixed_or_neutral": [], "insufficient": []}
    for a in assessments:
        entry = {"expectation_id": a.expectation_id, "metric": a.metric, "summary": a.summary, "evidence_ids": a.evidence_ids}
        if a.status == "SUPPORTS":
            buckets["supporting"].append(entry)
        elif a.status == "CONTRADICTS":
            buckets["contradicting"].append(entry)
        elif a.status == "INSUFFICIENT":
            buckets["insufficient"].append(entry)
        else:
            buckets["mixed_or_neutral"].append(entry)
    return buckets

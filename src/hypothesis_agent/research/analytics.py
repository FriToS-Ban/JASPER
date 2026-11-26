"""Deterministic financial analytics over the evidence registry.

All financial arithmetic lives here, outside the LLM. Each function reads evidence records,
computes a value, and registers a derived record whose :class:`Derivation` names the operation,
human-readable formula and input evidence IDs. :func:`recompute` re-executes any derivation from
its inputs so verification and reproducibility checks never have to trust a stored number.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hypothesis_agent.research.evidence import (
    Derivation,
    EvidenceRecord,
    EvidenceRegistry,
    PeriodType,
)
from hypothesis_agent.research.extraction import price_rows
from hypothesis_agent.research.metrics import METRICS, RATIO_RECIPES, get_metric

PayloadLoader = Callable[[str], Any]

# Relative change below which a USD metric is considered flat; absolute change for ratios.
FLAT_RELATIVE_THRESHOLD = 0.02
FLAT_RATIO_POINTS_THRESHOLD = 0.005
TRADING_DAYS = 252


class AnalyticsError(ValueError):
    """Raised when an analytical operation cannot be performed on the available evidence."""


def default_payload_loader(registry: EvidenceRegistry) -> PayloadLoader:
    def _load(snapshot_id: str) -> Any:
        snapshot = registry.snapshots.get(snapshot_id)
        if snapshot is None or not snapshot.artifact_path:
            raise AnalyticsError(f"Snapshot {snapshot_id} has no stored payload to read.")
        path = Path(snapshot.artifact_path)
        if not path.exists():
            raise AnalyticsError(f"Stored payload for {snapshot_id} is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


# ----------------------------------------------------------------------------------------------
# Ratios
# ----------------------------------------------------------------------------------------------
def compute_ratio(
    registry: EvidenceRegistry,
    ticker: str,
    metric: str,
    period_type: PeriodType = PeriodType.ANNUAL,
    tool_call_id: Optional[str] = None,
) -> List[EvidenceRecord]:
    """Compute a catalogue ratio (e.g. operating_margin) for every period with both inputs."""

    if metric == "free_cash_flow_computed":
        return compute_free_cash_flow(registry, ticker, period_type, tool_call_id)
    if metric not in RATIO_RECIPES:
        raise AnalyticsError(f"No ratio recipe for metric '{metric}'.")
    numerator_key, denominator_key, scale, sign = RATIO_RECIPES[metric]
    definition = METRICS[metric]
    numerators = {r.period_end: r for r in registry.series(ticker, numerator_key, period_type)}
    denominators = {r.period_end: r for r in registry.series(ticker, denominator_key, period_type)}
    if not numerators or not denominators:
        missing = [k for k, v in ((numerator_key, numerators), (denominator_key, denominators)) if not v]
        registry.record_missing(
            ticker=ticker, metric=metric,
            reason=f"Cannot compute {definition.label}: no {', '.join(missing)} evidence collected.",
        )
        return []
    created: List[EvidenceRecord] = []
    for period_end in sorted(set(numerators) & set(denominators)):
        num, den = numerators[period_end], denominators[period_end]
        problem = _ratio_problem(metric, num.value, den.value)
        if problem:
            registry.record_missing(ticker=ticker, metric=metric, period=num.period, reason=problem)
            continue
        value = sign * num.value / den.value * scale
        sign_text = "-" if sign < 0 else ""
        scale_text = f" x {scale:g}" if scale != 1 else ""
        created.append(
            registry.register_derived(
                ticker=ticker,
                metric=metric,
                label=definition.label,
                value=value,
                unit=definition.unit,
                period=num.period,
                period_type=period_type,
                period_end=period_end,
                derivation=Derivation(
                    operation="ratio",
                    formula=f"{sign_text}{num.label} / {den.label}{scale_text}",
                    inputs=[num.evidence_id, den.evidence_id],
                    parameters={"scale": scale, "sign": sign},
                    tool_call_id=tool_call_id,
                ),
            )
        )
    return created


def _ratio_problem(metric: str, numerator: float, denominator: float) -> Optional[str]:
    if denominator == 0:
        return "Denominator is zero; ratio undefined."
    if metric in {"debt_to_equity", "return_on_equity_computed"} and denominator < 0:
        return "Stockholders' equity is negative; equity-based ratio is not meaningful."
    if metric in {"cash_conversion", "fcf_conversion"} and denominator <= 0:
        return "Net income is not positive; conversion ratio is not meaningful."
    if metric == "debt_to_ebitda" and denominator <= 0:
        return "EBITDA is not positive; leverage multiple is not meaningful."
    if metric == "interest_coverage" and denominator <= 0:
        return "Interest expense not positive; coverage not meaningful."
    return None


def compute_free_cash_flow(
    registry: EvidenceRegistry,
    ticker: str,
    period_type: PeriodType = PeriodType.ANNUAL,
    tool_call_id: Optional[str] = None,
) -> List[EvidenceRecord]:
    ocf = {r.period_end: r for r in registry.series(ticker, "operating_cash_flow", period_type)}
    capex = {r.period_end: r for r in registry.series(ticker, "capital_expenditure", period_type)}
    created = []
    for period_end in sorted(set(ocf) & set(capex)):
        a, b = ocf[period_end], capex[period_end]
        created.append(
            registry.register_derived(
                ticker=ticker,
                metric="free_cash_flow_computed",
                label=METRICS["free_cash_flow_computed"].label,
                value=a.value + b.value,
                unit=a.unit,
                currency=a.currency,
                period=a.period,
                period_type=period_type,
                period_end=period_end,
                derivation=Derivation(
                    operation="sum",
                    formula="Operating Cash Flow + Capital Expenditure (capex reported as a negative outflow)",
                    inputs=[a.evidence_id, b.evidence_id],
                    tool_call_id=tool_call_id,
                ),
            )
        )
    return created


# ----------------------------------------------------------------------------------------------
# Growth, change, trend
# ----------------------------------------------------------------------------------------------
def compute_growth(
    registry: EvidenceRegistry,
    ticker: str,
    metric: str,
    period_type: PeriodType = PeriodType.ANNUAL,
    tool_call_id: Optional[str] = None,
) -> List[EvidenceRecord]:
    """Period-over-period growth. Annual -> YoY; quarterly -> YoY against the same quarter."""

    series = registry.series(ticker, metric, period_type)
    lag = 4 if period_type == PeriodType.QUARTERLY else 1
    suffix = "_growth_yoy"
    if len(series) <= lag:
        registry.record_missing(
            ticker=ticker, metric=f"{metric}{suffix}",
            reason=f"Need more than {lag} {period_type.value} periods of {metric} to compute growth; have {len(series)}.",
        )
        return []
    definition = get_metric(f"{metric}{suffix}")
    created = []
    for index in range(lag, len(series)):
        previous, current = series[index - lag], series[index]
        if previous.value == 0:
            registry.record_missing(ticker=ticker, metric=f"{metric}{suffix}", period=current.period,
                                    reason="Prior-period value is zero; growth undefined.")
            continue
        value = (current.value - previous.value) / abs(previous.value)
        notes = []
        if previous.value < 0:
            notes.append("Prior-period value is negative; growth computed against its absolute value.")
        created.append(
            registry.register_derived(
                ticker=ticker,
                metric=f"{metric}{suffix}",
                label=definition.label,
                value=value,
                unit="ratio",
                period=current.period,
                period_type=period_type,
                period_start=previous.period_end,
                period_end=current.period_end,
                derivation=Derivation(
                    operation="growth",
                    formula=f"({current.label} {current.period_end} - {previous.period_end}) / |{previous.period_end}|",
                    inputs=[previous.evidence_id, current.evidence_id],
                    tool_call_id=tool_call_id,
                ),
                notes=notes,
            )
        )
    return created


def short_period(record: EvidenceRecord) -> str:
    """Compact period label: FY2025 for annual records, the end date otherwise."""

    if record.period_type == PeriodType.ANNUAL and record.period_year:
        return f"FY{record.period_year}"
    return record.period_end or record.period


def _window(series: Sequence[EvidenceRecord], start_year: Optional[int], end_year: Optional[int]) -> List[EvidenceRecord]:
    return [
        r for r in series
        if (start_year is None or (r.period_year or 0) >= start_year)
        and (end_year is None or (r.period_year or 0) <= end_year)
    ]


def compute_cagr(
    registry: EvidenceRegistry,
    ticker: str,
    metric: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    tool_call_id: Optional[str] = None,
) -> Optional[EvidenceRecord]:
    series = _window(registry.series(ticker, metric, PeriodType.ANNUAL), start_year, end_year)
    if len(series) < 2:
        registry.record_missing(ticker=ticker, metric=f"{metric}_cagr",
                                reason=f"Need at least two annual {metric} values in window for a CAGR.")
        return None
    first, last = series[0], series[-1]
    if first.value <= 0 or last.value <= 0:
        registry.record_missing(ticker=ticker, metric=f"{metric}_cagr",
                                reason="CAGR undefined when start or end value is not positive.")
        return None
    years = (last.period_year or 0) - (first.period_year or 0)
    if years <= 0:
        return None
    value = (last.value / first.value) ** (1.0 / years) - 1.0
    return registry.register_derived(
        ticker=ticker,
        metric=f"{metric}_cagr",
        label=get_metric(f"{metric}_cagr").label,
        value=value,
        unit="ratio",
        period=f"FY{first.period_year}-FY{last.period_year}",
        period_type=PeriodType.RANGE,
        period_start=first.period_end,
        period_end=last.period_end,
        derivation=Derivation(
            operation="cagr",
            formula=f"({last.period_end} / {first.period_end}) ^ (1/{years}) - 1",
            inputs=[first.evidence_id, last.evidence_id],
            parameters={"years": years},
            tool_call_id=tool_call_id,
        ),
    )


def compare_periods(
    registry: EvidenceRegistry,
    ticker: str,
    metric: str,
    start_year: int,
    end_year: int,
    period_type: PeriodType = PeriodType.ANNUAL,
    tool_call_id: Optional[str] = None,
) -> Optional[EvidenceRecord]:
    series = registry.series(ticker, metric, period_type)
    start = next((r for r in series if r.period_year == start_year), None)
    end = next((r for r in series if r.period_year == end_year), None)
    if start is None or end is None:
        registry.record_missing(ticker=ticker, metric=f"{metric}_change",
                                reason=f"{metric} not available for both FY{start_year} and FY{end_year}.")
        return None
    base = METRICS.get(metric)
    unit = "ratio_points" if base and base.unit == "ratio" else (start.unit)
    return registry.register_derived(
        ticker=ticker,
        metric=f"{metric}_change",
        label=get_metric(f"{metric}_change").label,
        value=end.value - start.value,
        unit=unit,
        currency=start.currency,
        period=f"FY{start_year} to FY{end_year}",
        period_type=PeriodType.RANGE,
        period_start=start.period_end,
        period_end=end.period_end,
        derivation=Derivation(
            operation="change",
            formula=f"{metric}[{end.period_end}] - {metric}[{start.period_end}]",
            inputs=[start.evidence_id, end.evidence_id],
            tool_call_id=tool_call_id,
        ),
    )


def classify_steps(values: Sequence[float], is_ratio: bool) -> Tuple[List[str], str]:
    """Classify each step and the overall path as increasing / decreasing / flat / mixed."""

    steps: List[str] = []
    for previous, current in zip(values, values[1:]):
        steps.append(_direction(previous, current, is_ratio))
    overall = _direction(values[0], values[-1], is_ratio) if len(values) >= 2 else "flat"
    return steps, overall


def _direction(previous: float, current: float, is_ratio: bool) -> str:
    if is_ratio:
        delta = current - previous
        if abs(delta) < FLAT_RATIO_POINTS_THRESHOLD:
            return "flat"
        return "increasing" if delta > 0 else "decreasing"
    base = abs(previous) if previous != 0 else max(abs(current), 1.0)
    relative = (current - previous) / base
    if abs(relative) < FLAT_RELATIVE_THRESHOLD:
        return "flat"
    return "increasing" if relative > 0 else "decreasing"


def assess_trend(
    registry: EvidenceRegistry,
    ticker: str,
    metric: str,
    period_type: PeriodType = PeriodType.ANNUAL,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    tool_call_id: Optional[str] = None,
) -> Optional[EvidenceRecord]:
    """Summarise a metric's path over a window as a derived ``<metric>_trend`` record.

    The value is the change from first to last period (percentage points for ratios, relative
    change otherwise); the qualifier records the overall direction, and parameters keep each
    step's direction so the period-dependence of a conclusion is visible.
    """

    series = _window(registry.series(ticker, metric, period_type), start_year, end_year)
    if len(series) < 2:
        registry.record_missing(
            ticker=ticker, metric=f"{metric}_trend",
            reason=f"Need at least two {period_type.value} periods of {metric} in the window; have {len(series)}.",
        )
        return None
    base = METRICS.get(metric) or get_metric(metric)
    is_ratio = base.unit in {"ratio", "x", "days"} or metric.endswith("_growth_yoy")
    values = [r.value for r in series]
    steps, overall = classify_steps(values, is_ratio)
    if is_ratio:
        value = values[-1] - values[0]
        formula = f"{metric}[last] - {metric}[first] (percentage points for ratios)"
    else:
        base_value = abs(values[0]) if values[0] != 0 else 1.0
        value = (values[-1] - values[0]) / base_value
        formula = f"({metric}[last] - {metric}[first]) / |{metric}[first]|"
    qualifier = overall
    if steps and steps[-1] not in {overall, "flat"} and overall != "flat":
        qualifier = f"{overall}, latest period reversed"
    return registry.register_derived(
        ticker=ticker,
        metric=f"{metric}_trend",
        label=get_metric(f"{metric}_trend").label,
        value=value,
        unit="ratio_points" if is_ratio else "ratio",
        period=f"{short_period(series[0])} to {short_period(series[-1])}",
        period_type=PeriodType.RANGE,
        period_start=series[0].period_end,
        period_end=series[-1].period_end,
        qualifier=qualifier,
        derivation=Derivation(
            operation="trend",
            formula=formula,
            inputs=[r.evidence_id for r in series],
            parameters={"is_ratio": is_ratio, "steps": steps, "overall": overall,
                        "start_year": start_year, "end_year": end_year, "frequency": period_type.value},
            tool_call_id=tool_call_id,
        ),
    )


# ----------------------------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------------------------
def compute_price_metrics(
    registry: EvidenceRegistry,
    ticker: str,
    snapshot_id: str,
    loader: PayloadLoader,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> List[EvidenceRecord]:
    rows = _price_window(loader(snapshot_id), start_date, end_date)
    if len(rows) < 2:
        registry.record_missing(ticker=ticker, metric="price_return",
                                reason="Fewer than two closing prices in the requested window.")
        return []
    first_date, last_date = rows[0][0], rows[-1][0]
    params = {"start_date": first_date, "end_date": last_date, "field": "close", "observations": len(rows)}
    snapshot = registry.snapshots[snapshot_id]
    currency_unit = "ratio"
    period = f"{first_date} to {last_date}"
    created = []
    for metric, value, formula in (
        ("price_return", _price_return(rows), "Close(end) / Close(start) - 1"),
        ("annualized_volatility", _volatility(rows), "stdev(ln(Close_t / Close_t-1)) x sqrt(252)"),
        ("max_drawdown", _max_drawdown(rows), "min_t(Close_t / max_{s<=t} Close_s) - 1"),
    ):
        if value is None:
            continue
        created.append(
            registry.register_derived(
                ticker=ticker,
                metric=metric,
                label=METRICS[metric].label,
                value=value,
                unit=currency_unit,
                period=period,
                period_type=PeriodType.RANGE,
                period_start=first_date,
                period_end=last_date,
                derivation=Derivation(
                    operation=metric,
                    formula=formula,
                    input_snapshots=[snapshot.snapshot_id],
                    parameters=params,
                    tool_call_id=tool_call_id,
                ),
                notes=["Computed from split/dividend-adjusted daily closes."],
            )
        )
    return created


def _price_window(payload: Any, start_date: Optional[str], end_date: Optional[str]) -> List[Tuple[str, float]]:
    return [
        (d, c) for d, c in price_rows(payload)
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
    ]


def _price_return(rows: Sequence[Tuple[str, float]]) -> Optional[float]:
    if rows[0][1] == 0:
        return None
    return rows[-1][1] / rows[0][1] - 1.0


def _volatility(rows: Sequence[Tuple[str, float]]) -> Optional[float]:
    returns = [math.log(b / a) for (_, a), (_, b) in zip(rows, rows[1:]) if a > 0 and b > 0]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS)


def _max_drawdown(rows: Sequence[Tuple[str, float]]) -> Optional[float]:
    peak = rows[0][1]
    worst = 0.0
    for _, close in rows:
        peak = max(peak, close)
        if peak > 0:
            worst = min(worst, close / peak - 1.0)
    return worst


# ----------------------------------------------------------------------------------------------
# Valuation & statistics
# ----------------------------------------------------------------------------------------------
def compute_valuation(
    registry: EvidenceRegistry, ticker: str, tool_call_id: Optional[str] = None
) -> List[EvidenceRecord]:
    """P/E from price and latest FY diluted EPS, and PEG against historical EPS CAGR."""

    created: List[EvidenceRecord] = []
    price = registry.latest(ticker, "current_price")
    eps_series = registry.series(ticker, "diluted_eps", PeriodType.ANNUAL)
    if price is None or not eps_series:
        registry.record_missing(ticker=ticker, metric="pe_computed",
                                reason="Share price or annual diluted EPS not collected.")
    elif eps_series[-1].value <= 0:
        registry.record_missing(ticker=ticker, metric="pe_computed", period=eps_series[-1].period,
                                reason="Latest annual diluted EPS is not positive; P/E is not meaningful.")
    else:
        eps = eps_series[-1]
        created.append(
            registry.register_derived(
                ticker=ticker, metric="pe_computed", label=METRICS["pe_computed"].label,
                value=price.value / eps.value, unit="x",
                period=f"Price {price.period_end} / {short_period(eps)} EPS",
                period_type=PeriodType.POINT_IN_TIME, period_end=price.period_end,
                derivation=Derivation(operation="ratio", formula="Share Price / latest FY Diluted EPS",
                                      inputs=[price.evidence_id, eps.evidence_id],
                                      parameters={"scale": 1.0, "sign": 1.0}, tool_call_id=tool_call_id),
            )
        )
    pe = registry.latest(ticker, "trailing_pe") or next((r for r in created if r.metric == "pe_computed"), None)
    cagr = compute_cagr(registry, ticker, "diluted_eps", tool_call_id=tool_call_id)
    if pe is None:
        registry.record_missing(ticker=ticker, metric="peg_computed",
                                reason="No P/E available (trailing P/E not reported and not computable).")
    elif cagr is None:
        registry.record_missing(ticker=ticker, metric="peg_computed",
                                reason="EPS CAGR undefined (needs positive start and end EPS).")
    elif cagr.value <= 0:
        registry.record_missing(ticker=ticker, metric="peg_computed", period=cagr.period,
                                reason="Historical EPS growth is not positive; PEG is not meaningful.")
    else:
        created.append(
            registry.register_derived(
                ticker=ticker, metric="peg_computed", label=METRICS["peg_computed"].label,
                value=pe.value / (cagr.value * 100.0), unit="x",
                period=f"P/E {pe.period_end} vs EPS CAGR {cagr.period}",
                period_type=PeriodType.POINT_IN_TIME, period_end=pe.period_end,
                derivation=Derivation(operation="peg", formula="P/E / (EPS CAGR x 100)",
                                      inputs=[pe.evidence_id, cagr.evidence_id], tool_call_id=tool_call_id),
            )
        )
        created.append(cagr)
    return created


def compute_correlation(
    registry: EvidenceRegistry,
    ticker: str,
    metric_a: str,
    metric_b: str,
    period_type: PeriodType = PeriodType.ANNUAL,
    tool_call_id: Optional[str] = None,
) -> Optional[EvidenceRecord]:
    a = {r.period_end: r for r in registry.series(ticker, metric_a, period_type)}
    b = {r.period_end: r for r in registry.series(ticker, metric_b, period_type)}
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        registry.record_missing(ticker=ticker, metric=f"corr_{metric_a}_{metric_b}",
                                reason=f"Need at least 3 overlapping periods for a correlation; have {len(shared)}.")
        return None
    xs = [a[p].value for p in shared]
    ys = [b[p].value for p in shared]
    r = _pearson(xs, ys)
    if r is None:
        return None
    inputs: List[str] = []
    for p in shared:
        inputs.extend([a[p].evidence_id, b[p].evidence_id])
    return registry.register_derived(
        ticker=ticker, metric=f"corr_{metric_a}_{metric_b}",
        label=f"Correlation of {metric_a} and {metric_b}", value=r, unit="correlation",
        period=f"{len(shared)} {period_type.value} periods", period_type=PeriodType.RANGE,
        period_start=shared[0], period_end=shared[-1],
        derivation=Derivation(operation="correlation", formula="Pearson r over aligned periods",
                              inputs=inputs, parameters={"n": len(shared)}, tool_call_id=tool_call_id),
        notes=[f"Only {len(shared)} observations; correlation is descriptive, not statistically significant."],
    )


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    sy = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (sx * sy)


# ----------------------------------------------------------------------------------------------
# Recompute
# ----------------------------------------------------------------------------------------------
def recompute(registry: EvidenceRegistry, record: EvidenceRecord, loader: Optional[PayloadLoader] = None) -> float:
    """Re-execute a derived record's derivation from its inputs and return the fresh value."""

    if record.kind == "raw" or record.derivation is None:
        return record.value
    derivation = record.derivation
    inputs = [registry.get(eid).value for eid in derivation.inputs]
    op = derivation.operation
    params = derivation.parameters
    if op == "ratio":
        return float(params.get("sign", 1.0)) * inputs[0] / inputs[1] * float(params.get("scale", 1.0))
    if op == "sum":
        return float(sum(inputs))
    if op == "growth":
        return (inputs[1] - inputs[0]) / abs(inputs[0])
    if op == "cagr":
        return (inputs[1] / inputs[0]) ** (1.0 / float(params["years"])) - 1.0
    if op == "change":
        return inputs[1] - inputs[0]
    if op == "trend":
        if params.get("is_ratio"):
            return inputs[-1] - inputs[0]
        base = abs(inputs[0]) if inputs[0] != 0 else 1.0
        return (inputs[-1] - inputs[0]) / base
    if op == "peg":
        return inputs[0] / (inputs[1] * 100.0)
    if op == "correlation":
        xs, ys = inputs[0::2], inputs[1::2]
        value = _pearson(xs, ys)
        if value is None:
            raise AnalyticsError("Correlation undefined on recompute.")
        return value
    if op in {"price_return", "annualized_volatility", "max_drawdown"}:
        loader = loader or default_payload_loader(registry)
        rows = _price_window(loader(derivation.input_snapshots[0]), params.get("start_date"), params.get("end_date"))
        fn = {"price_return": _price_return, "annualized_volatility": _volatility, "max_drawdown": _max_drawdown}[op]
        value = fn(rows)
        if value is None:
            raise AnalyticsError(f"{op} undefined on recompute.")
        return value
    raise AnalyticsError(f"Unknown derivation operation '{op}'.")


def check_consistency(
    registry: EvidenceRegistry, loader: Optional[PayloadLoader] = None, rel_tol: float = 1e-9
) -> List[Dict[str, Any]]:
    """Recompute every derived record; return the ones whose stored value does not reproduce."""

    problems = []
    for record in list(registry.records.values()):
        if record.kind != "derived":
            continue
        try:
            fresh = recompute(registry, record, loader)
        except Exception as exc:  # surfaced to the critique as an inconsistency
            problems.append({"evidence_id": record.evidence_id, "metric": record.metric, "error": str(exc)})
            continue
        if not math.isclose(fresh, record.value, rel_tol=rel_tol, abs_tol=1e-12):
            problems.append({
                "evidence_id": record.evidence_id,
                "metric": record.metric,
                "stored": record.value,
                "recomputed": fresh,
            })
    return problems

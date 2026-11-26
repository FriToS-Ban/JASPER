"""Charts rendered from evidence records, each carrying a reproducible source caption."""
from __future__ import annotations

import io
from typing import List, Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from hypothesis_agent.research.analytics import short_period  # noqa: E402
from hypothesis_agent.research.evidence import EvidenceRegistry, PeriodType  # noqa: E402
from hypothesis_agent.research.metrics import get_metric  # noqa: E402
from hypothesis_agent.storage.artifact_store import ArtifactStore  # noqa: E402


class ChartSpec(BaseModel):
    chart_id: str
    title: str
    metric: str
    ticker: str
    unit: str
    frequency: str
    period_label: str
    evidence_ids: List[str] = Field(default_factory=list)
    snapshot_ids: List[str] = Field(default_factory=list)
    retrievals: List[str] = Field(default_factory=list)
    methodology: Optional[str] = None
    artifact_path: Optional[str] = None
    artifact_uri: Optional[str] = None
    caption: str = ""


def _unit_label(unit: str) -> str:
    return {
        "ratio": "% ",
        "ratio_points": "percentage points",
        "x": "multiple (x)",
        "USD/share": "USD per share",
        "days": "days",
    }.get(unit, unit)


def render_metric_chart(
    registry: EvidenceRegistry,
    artifact_store: ArtifactStore,
    workflow_id: str,
    *,
    ticker: str,
    metric: str,
    frequency: str = "annual",
    chart_id: str,
) -> Optional[ChartSpec]:
    """Render a metric's series. Returns None (and renders nothing) if fewer than 2 points exist."""

    period_type = PeriodType.QUARTERLY if frequency == "quarterly" else PeriodType.ANNUAL
    series = registry.series(ticker, metric, period_type)
    if len(series) < 2:
        return None
    definition = get_metric(metric)
    labels = [short_period(r) for r in series]
    values = [r.value for r in series]
    unit = series[0].unit
    is_ratio = unit in {"ratio", "ratio_points"}
    scale, scale_label = 1.0, unit
    if unit == "USD" or (series[0].currency is not None and unit == series[0].currency):
        scale, scale_label = 1e9, f"{unit} billions"

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    plotted = [v / scale for v in values]
    if is_ratio:
        ax.plot(labels, [v * 100 for v in values], color="#0f3460", marker="o", linewidth=1.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        y_label = "Percent"
    else:
        colors = ["#0f3460" if v >= 0 else "#e94560" for v in values]
        ax.bar(labels, plotted, color=colors)
        y_label = _unit_label(scale_label)
    ax.axhline(0, color="#999999", linewidth=0.6)
    title = f"{ticker} {definition.label} ({labels[0]}-{labels[-1]}, {frequency})"
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(y_label, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    safe_id = chart_id.replace(" ", "").lower()
    path = artifact_store.write_bytes(workflow_id, f"chart_{safe_id}_{ticker}_{metric}.png", buffer.getvalue()).resolve()
    buffer.close()

    snapshot_ids: List[str] = []
    for record in series:
        snapshot_ids.extend(registry.raw_snapshot_ids(record.evidence_id))
    snapshot_ids = list(dict.fromkeys(snapshot_ids))
    retrievals = [registry.snapshots[sid].describe() for sid in snapshot_ids if sid in registry.snapshots]
    methodology = None
    if series[0].kind == "derived" and series[0].derivation is not None:
        methodology = f"{definition.formula or series[0].derivation.formula} (computed by JASPER analytics)"
    spec = ChartSpec(
        chart_id=chart_id,
        title=title,
        metric=metric,
        ticker=ticker,
        unit=unit,
        frequency=frequency,
        period_label=f"{labels[0]} to {labels[-1]}",
        evidence_ids=[r.evidence_id for r in series],
        snapshot_ids=snapshot_ids,
        retrievals=retrievals,
        methodology=methodology,
        artifact_path=str(path),
        artifact_uri=path.as_uri(),
    )
    spec.caption = chart_caption(spec)
    return spec


def chart_caption(spec: ChartSpec) -> str:
    parts = [
        f"{spec.chart_id}. Unit: {_unit_label(spec.unit).strip()}. Periods: {spec.period_label}.",
        f"Evidence: {', '.join(spec.evidence_ids)}.",
        f"Source snapshots: {', '.join(spec.snapshot_ids)}.",
    ]
    if spec.retrievals:
        parts.append("Retrieval: " + "; ".join(spec.retrievals) + ".")
    if spec.methodology:
        parts.append(f"Calculation: {spec.methodology}.")
    return " ".join(parts)

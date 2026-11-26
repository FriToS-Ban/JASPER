"""Explicit, validated research tools the agent (deterministic or LLM-driven) can invoke.

The toolbox is the only way research code touches data: no tool accepts file paths, shell
commands or arbitrary code. Arguments are validated with pydantic (unknown fields rejected),
tickers are restricted to the plan's universe, every call is logged with its inputs and outputs,
and a hard call budget bounds execution.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hypothesis_agent.orchestration.yfinance_tools import YFINANCE_VERSION, YFinanceToolSet, describe_retrieval
from hypothesis_agent.research import analytics
from hypothesis_agent.research.charts import ChartSpec, render_metric_chart
from hypothesis_agent.research.evidence import EvidenceRegistry, PeriodType, utc_now_iso
from hypothesis_agent.research.extraction import build_snapshot, detect_currency, extract_evidence
from hypothesis_agent.research.metrics import DATASETS, METRICS, RATIO_RECIPES, is_known_metric
from hypothesis_agent.storage.artifact_store import ArtifactStore

Frequency = Literal["annual", "quarterly"]
DatasetName = Literal[
    "income_statement_annual",
    "income_statement_quarterly",
    "balance_sheet_annual",
    "balance_sheet_quarterly",
    "cash_flow_annual",
    "cash_flow_quarterly",
    "company_info",
    "price_history",
    "news",
    "dividends",
]
COMPUTABLE_METRICS = sorted(list(RATIO_RECIPES) + ["free_cash_flow_computed"])


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FetchDatasetArgs(_Args):
    dataset: DatasetName
    ticker: str = Field(min_length=1, max_length=12)
    period: Literal["1y", "2y", "3y", "5y", "max"] = "1y"


class ComputeMetricArgs(_Args):
    metric: str
    ticker: str
    frequency: Frequency = "annual"


class GrowthArgs(_Args):
    metric: str
    ticker: str
    frequency: Frequency = "annual"


class CagrArgs(_Args):
    metric: str
    ticker: str
    start_year: Optional[int] = Field(default=None, ge=1990, le=2100)
    end_year: Optional[int] = Field(default=None, ge=1990, le=2100)


class ComparePeriodsArgs(_Args):
    metric: str
    ticker: str
    start_year: int = Field(ge=1990, le=2100)
    end_year: int = Field(ge=1990, le=2100)


class TrendArgs(_Args):
    metric: str
    ticker: str
    frequency: Frequency = "annual"
    start_year: Optional[int] = Field(default=None, ge=1990, le=2100)
    end_year: Optional[int] = Field(default=None, ge=1990, le=2100)


class PricePerformanceArgs(_Args):
    ticker: str
    start_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class TickerArgs(_Args):
    ticker: str


class CorrelateArgs(_Args):
    metric_a: str
    metric_b: str
    ticker: str
    frequency: Frequency = "annual"


class InspectArgs(_Args):
    ticker: Optional[str] = None
    metric: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=60)


class PlotArgs(_Args):
    metric: str
    ticker: str
    frequency: Frequency = "annual"


class ToolCallRecord(BaseModel):
    call_id: str
    tool: str
    arguments: Dict[str, Any]
    origin: str
    rationale: Optional[str] = None
    iteration: int
    status: Literal["ok", "error", "rejected", "skipped"]
    summary: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    snapshot_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    started_at: str
    duration_ms: float = 0.0


class ToolResult(BaseModel):
    call_id: str
    tool: str
    status: Literal["ok", "error", "rejected", "skipped"]
    output: Dict[str, Any] = Field(default_factory=dict)
    evidence_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: Type[_Args]
    handler: Callable[..., Dict[str, Any]]
    read_only: bool = False


class ToolBudgetExceeded(RuntimeError):
    pass


class ResearchToolbox:
    """Registry-backed analytical tools with validation, logging and a call budget."""

    def __init__(
        self,
        *,
        registry: EvidenceRegistry,
        toolset: Any,
        artifact_store: ArtifactStore,
        workflow_id: str,
        allowed_tickers: List[str],
        call_log: List[Dict[str, Any]],
        currencies: Dict[str, Dict[str, Optional[str]]],
        charts: List[Dict[str, Any]],
        iteration: int = 1,
        max_calls: int = 60,
    ) -> None:
        self.registry = registry
        self.toolset = toolset
        self.artifact_store = artifact_store
        self.workflow_id = workflow_id
        self.allowed_tickers = [t.upper() for t in allowed_tickers]
        self.call_log = call_log
        self.currencies = currencies
        self.charts = charts
        self.iteration = iteration
        self.max_calls = max_calls
        self._payload_cache: Dict[str, Any] = {}
        self.specs: Dict[str, ToolSpec] = {}
        self._register_tools()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def describe_tools(self, compact: bool = False) -> List[Dict[str, Any]]:
        if not compact:
            return [
                {"name": spec.name, "description": spec.description, "parameters": spec.args_model.model_json_schema()}
                for spec in self.specs.values()
            ]
        described = []
        for spec in self.specs.values():
            params = {}
            for name, field in spec.args_model.model_fields.items():
                annotation = getattr(field.annotation, "__args__", None)
                choices = [a for a in annotation if isinstance(a, str)] if annotation else []
                params[name] = ("|".join(choices) if choices else getattr(field.annotation, "__name__", str(field.annotation))) + \
                    ("" if field.is_required() else f" (default {field.default})")
            described.append({"name": spec.name, "description": spec.description, "parameters": params})
        return described

    @property
    def calls_made(self) -> int:
        return len(self.call_log)

    def invoke(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        origin: str,
        rationale: Optional[str] = None,
    ) -> ToolResult:
        arguments = dict(arguments or {})
        call_id = f"TC-{self.iteration:02d}-{len(self.call_log) + 1:03d}"
        started = time.perf_counter()
        started_at = utc_now_iso()

        def _finish(status: str, output: Dict[str, Any], *, error: Optional[str] = None) -> ToolResult:
            evidence_ids = list(output.pop("_evidence_ids", []))
            snapshot_ids = list(output.pop("_snapshot_ids", []))
            record = ToolCallRecord(
                call_id=call_id, tool=name, arguments=arguments, origin=origin, rationale=rationale,
                iteration=self.iteration, status=status, summary=str(output.get("summary", ""))[:400],
                evidence_ids=evidence_ids, snapshot_ids=snapshot_ids, error=error, started_at=started_at,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            self.call_log.append(record.model_dump(mode="json"))
            return ToolResult(call_id=call_id, tool=name, status=status, output=output,
                              evidence_ids=evidence_ids, error=error)

        spec = self.specs.get(name)
        if spec is None:
            return _finish("rejected", {}, error=f"Unknown tool '{name}'. Available: {sorted(self.specs)}")
        if self.calls_made >= self.max_calls:
            return _finish("rejected", {}, error=f"Tool budget of {self.max_calls} calls exhausted.")
        try:
            parsed = spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            return _finish("rejected", {}, error=f"Invalid arguments: {exc.errors(include_url=False)}")
        ticker = getattr(parsed, "ticker", None)
        if ticker is not None:
            ticker = ticker.upper()
            if ticker not in self.allowed_tickers:
                return _finish("rejected", {}, error=f"Ticker {ticker} is outside the research universe {self.allowed_tickers}.")
            parsed = parsed.model_copy(update={"ticker": ticker})
        try:
            output = spec.handler(parsed, call_id)
        except analytics.AnalyticsError as exc:
            return _finish("error", {"summary": str(exc)}, error=str(exc))
        status = output.pop("_status", "ok")
        return _finish(status, output, error=output.get("error"))

    def payload(self, snapshot_id: str) -> Any:
        if snapshot_id not in self._payload_cache:
            self._payload_cache[snapshot_id] = analytics.default_payload_loader(self.registry)(snapshot_id)
        return self._payload_cache[snapshot_id]

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def _register_tools(self) -> None:
        def add(name: str, description: str, args_model: Type[_Args], handler: Callable[..., Dict[str, Any]], read_only: bool = False) -> None:
            self.specs[name] = ToolSpec(name=name, description=description, args_model=args_model,
                                        handler=handler, read_only=read_only)

        add("fetch_dataset", "Retrieve a Yahoo Finance dataset (statements, company info, prices) and register its metrics as evidence. Re-requesting an identical dataset returns the existing snapshot.", FetchDatasetArgs, self._fetch_dataset)
        add("compute_metric", f"Compute a derived ratio for every period with inputs. Metrics: {COMPUTABLE_METRICS}.", ComputeMetricArgs, self._compute_metric)
        add("compute_growth", "Compute period-over-period growth (YoY) for any collected or derived metric.", GrowthArgs, self._compute_growth)
        add("compute_cagr", "Compute a compound annual growth rate for an annual metric between two fiscal years.", CagrArgs, self._compute_cagr)
        add("compare_periods", "Difference of a metric between two fiscal years.", ComparePeriodsArgs, self._compare_periods)
        add("assess_trend", "Classify a metric's direction across a window (records step-by-step directions).", TrendArgs, self._assess_trend)
        add("price_performance", "Compute price return, annualised volatility and max drawdown from collected price history.", PricePerformanceArgs, self._price_performance)
        add("valuation_check", "Compute P/E from price and diluted EPS, and PEG against historical EPS CAGR.", TickerArgs, self._valuation_check)
        add("correlate", "Pearson correlation of two metrics across aligned periods (descriptive only).", CorrelateArgs, self._correlate)
        add("inspect_evidence", "List collected evidence records (read-only).", InspectArgs, self._inspect, read_only=True)
        add("list_missing", "List metrics that could not be collected and why (read-only).", TickerArgs, self._list_missing, read_only=True)
        add("plot_metric", "Render a chart of a metric series with source attribution.", PlotArgs, self._plot_metric)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _source_metadata(self, slug: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self.toolset, "source_metadata"):
            meta = dict(self.toolset.source_metadata(slug, arguments))
            version = meta.get("yfinance_version") or "unknown"
            captured = meta.get("captured_at") or "unknown time"
            meta["source_label"] = f"Yahoo Finance via yfinance {version} (frozen fixture captured {captured})"
            return meta
        if isinstance(self.toolset, YFinanceToolSet):
            return {"mode": "live", "yfinance_version": YFINANCE_VERSION,
                    "source_label": f"Yahoo Finance via yfinance {YFINANCE_VERSION}"}
        name = type(self.toolset).__name__
        return {"mode": "custom_toolset", "toolset": name,
                "source_label": f"Custom toolset {name} (provenance not verified as Yahoo Finance)"}

    def _fetch_dataset(self, args: FetchDatasetArgs, call_id: str) -> Dict[str, Any]:
        slug, extra = DATASETS[args.dataset]
        parameters: Dict[str, Any] = {"symbol": args.ticker, **extra}
        if args.dataset == "price_history":
            parameters["period"] = args.period
        existing = self.registry.find_snapshot(slug, parameters)
        if existing is not None:
            return {
                "_status": "skipped",
                "summary": f"{args.dataset} for {args.ticker} already collected as {existing.snapshot_id}; not re-fetched.",
                "snapshot_id": existing.snapshot_id,
            }
        retrieval = describe_retrieval(slug, parameters)
        meta = self._source_metadata(slug, parameters)
        try:
            payload = self.toolset.get_tool(slug).invoke(dict(parameters))
        except Exception as exc:  # recorded as an error snapshot + missing evidence, never hidden
            error = f"{type(exc).__name__}: {exc}"
            snapshot = build_snapshot(ticker=args.ticker, tool_slug=slug, parameters=parameters, payload=None,
                                      retrieval_function=retrieval, source_meta=meta, error=error)
            self.registry.register_snapshot(snapshot)
            for definition in METRICS.values():
                if definition.source and args.dataset.startswith(definition.source):
                    self.registry.record_missing(ticker=args.ticker, metric=definition.key, dataset=args.dataset,
                                                 reason=f"Data retrieval failed: {error}")
            self.registry.record_missing(ticker=args.ticker, metric="*", dataset=args.dataset,
                                         reason=f"Data retrieval failed: {error}")
            return {"_status": "error", "summary": f"Retrieval of {args.dataset} failed: {error}",
                    "error": error, "_snapshot_ids": [snapshot.snapshot_id]}

        provisional = build_snapshot(ticker=args.ticker, tool_slug=slug, parameters=parameters, payload=payload,
                                     retrieval_function=retrieval, source_meta=dict(meta))
        artifact_name = f"snapshot_{args.ticker}_{args.dataset}_{provisional.snapshot_id}"
        path = self.artifact_store.write_json(self.workflow_id, artifact_name, payload).resolve()
        snapshot = provisional.model_copy(update={"artifact_path": str(path), "artifact_uri": path.as_uri()})
        self.registry.register_snapshot(snapshot)
        self._payload_cache[snapshot.snapshot_id] = payload

        if slug == "YFINANCE_COMPANY_INFO":
            statement_currency, trading_currency = detect_currency(payload)
            self.currencies[args.ticker] = {"statement": statement_currency, "trading": trading_currency}
        currency_info = self.currencies.get(args.ticker) or {}
        currency = currency_info.get("trading") if slug == "YFINANCE_HISTORICAL_PRICES" else currency_info.get("statement")
        result = extract_evidence(self.registry, snapshot, payload, currency=currency)
        summary = (
            f"Collected {args.dataset} for {args.ticker} ({snapshot.record_count} rows"
            f"{', ' + snapshot.coverage_start + ' to ' + snapshot.coverage_end if snapshot.coverage_start else ''}); "
            f"registered {len(result.registered)} evidence records."
        )
        if snapshot.status == "empty":
            summary = f"{args.dataset} for {args.ticker} returned an empty payload."
        return {
            "summary": summary,
            "snapshot_id": snapshot.snapshot_id,
            "periods": result.periods,
            "warnings": result.warnings,
            "_evidence_ids": result.registered,
            "_snapshot_ids": [snapshot.snapshot_id],
            "_status": "ok" if snapshot.status == "ok" else "error",
        }

    @staticmethod
    def _period_type(frequency: str) -> PeriodType:
        return PeriodType.QUARTERLY if frequency == "quarterly" else PeriodType.ANNUAL

    def _records_output(self, records: List[Any], what: str) -> Dict[str, Any]:
        records = [r for r in records if r is not None]
        if not records:
            missing = [m.reason for m in self.registry.missing[-3:]]
            return {"_status": "error", "summary": f"No {what} could be computed.", "error": "; ".join(missing) or "insufficient inputs"}
        preview = [
            {"evidence_id": r.evidence_id, "period": r.period, "value": round(r.value, 6), "qualifier": r.qualifier}
            for r in records[-6:]
        ]
        return {"summary": f"Computed {len(records)} {what} record(s).", "records": preview,
                "_evidence_ids": [r.evidence_id for r in records]}

    def _compute_metric(self, args: ComputeMetricArgs, call_id: str) -> Dict[str, Any]:
        if args.metric not in COMPUTABLE_METRICS:
            raise analytics.AnalyticsError(f"compute_metric supports {COMPUTABLE_METRICS}; got '{args.metric}'.")
        records = analytics.compute_ratio(self.registry, args.ticker, args.metric, self._period_type(args.frequency), call_id)
        return self._records_output(records, args.metric)

    def _check_metric(self, metric: str) -> None:
        if not is_known_metric(metric) and not self.registry.find(metric=metric):
            raise analytics.AnalyticsError(f"Unknown metric '{metric}'.")

    def _compute_growth(self, args: GrowthArgs, call_id: str) -> Dict[str, Any]:
        self._check_metric(args.metric)
        records = analytics.compute_growth(self.registry, args.ticker, args.metric, self._period_type(args.frequency), call_id)
        return self._records_output(records, f"{args.metric} growth")

    def _compute_cagr(self, args: CagrArgs, call_id: str) -> Dict[str, Any]:
        self._check_metric(args.metric)
        record = analytics.compute_cagr(self.registry, args.ticker, args.metric, args.start_year, args.end_year, call_id)
        return self._records_output([record], f"{args.metric} CAGR")

    def _compare_periods(self, args: ComparePeriodsArgs, call_id: str) -> Dict[str, Any]:
        self._check_metric(args.metric)
        record = analytics.compare_periods(self.registry, args.ticker, args.metric, args.start_year, args.end_year, tool_call_id=call_id)
        return self._records_output([record], f"{args.metric} change")

    def _assess_trend(self, args: TrendArgs, call_id: str) -> Dict[str, Any]:
        self._check_metric(args.metric)
        record = analytics.assess_trend(self.registry, args.ticker, args.metric, self._period_type(args.frequency),
                                        args.start_year, args.end_year, call_id)
        return self._records_output([record], f"{args.metric} trend")

    def _price_performance(self, args: PricePerformanceArgs, call_id: str) -> Dict[str, Any]:
        snapshots = [s for s in self.registry.snapshots.values()
                     if s.ticker == args.ticker and s.tool_slug == "YFINANCE_HISTORICAL_PRICES" and s.status == "ok"]
        if not snapshots:
            raise analytics.AnalyticsError("No price history collected; call fetch_dataset(dataset='price_history') first.")
        snapshot = max(snapshots, key=lambda s: (s.record_count, s.retrieved_at))
        records = analytics.compute_price_metrics(self.registry, args.ticker, snapshot.snapshot_id, self.payload,
                                                  args.start_date, args.end_date, call_id)
        return self._records_output(records, "price metric")

    def _valuation_check(self, args: TickerArgs, call_id: str) -> Dict[str, Any]:
        records = analytics.compute_valuation(self.registry, args.ticker, call_id)
        return self._records_output(records, "valuation")

    def _correlate(self, args: CorrelateArgs, call_id: str) -> Dict[str, Any]:
        self._check_metric(args.metric_a)
        self._check_metric(args.metric_b)
        record = analytics.compute_correlation(self.registry, args.ticker, args.metric_a, args.metric_b,
                                               self._period_type(args.frequency), call_id)
        return self._records_output([record], "correlation")

    def _inspect(self, args: InspectArgs, call_id: str) -> Dict[str, Any]:
        records = self.registry.find(metric=args.metric, ticker=args.ticker.upper() if args.ticker else None)
        rows = [
            {"evidence_id": r.evidence_id, "metric": r.metric, "period": r.period, "value": round(r.value, 6),
             "unit": r.unit, "kind": r.kind, "qualifier": r.qualifier}
            for r in records[-args.limit:]
        ]
        return {"summary": f"{len(records)} matching record(s).", "records": rows}

    def _list_missing(self, args: TickerArgs, call_id: str) -> Dict[str, Any]:
        items = [m.model_dump(mode="json") for m in self.registry.missing if m.ticker == args.ticker][-30:]
        return {"summary": f"{len(items)} missing-evidence entries.", "missing": items}

    def _plot_metric(self, args: PlotArgs, call_id: str) -> Dict[str, Any]:
        self._check_metric(args.metric)
        for existing in self.charts:
            if existing["metric"] == args.metric and existing["ticker"] == args.ticker and existing["frequency"] == args.frequency:
                return {"_status": "skipped", "summary": f"Chart {existing['chart_id']} already exists.", "chart_id": existing["chart_id"]}
        chart_id = f"Figure {len(self.charts) + 1}"
        spec: Optional[ChartSpec] = render_metric_chart(
            self.registry, self.artifact_store, self.workflow_id,
            ticker=args.ticker, metric=args.metric, frequency=args.frequency, chart_id=chart_id,
        )
        if spec is None:
            return {"_status": "error", "summary": f"Not enough {args.metric} data points to chart.", "error": "fewer than two periods"}
        self.charts.append(spec.model_dump(mode="json"))
        return {"summary": f"Rendered {chart_id}: {spec.title}.", "chart_id": chart_id, "_evidence_ids": spec.evidence_ids}


def summarise_tool_log(call_log: List[Dict[str, Any]], limit: int = 12) -> str:
    """Compact text rendering of recent tool calls for LLM prompts."""

    lines = []
    for call in call_log[-limit:]:
        lines.append(f"{call['call_id']} {call['tool']}({json.dumps(call['arguments'])}) -> {call['status']}: {call['summary']}")
    return "\n".join(lines)

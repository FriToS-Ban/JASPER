"""Turn raw yfinance payloads into snapshots and traceable evidence records."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hypothesis_agent.research.evidence import (
    DataSnapshot,
    EvidenceRegistry,
    PeriodType,
    sha256_of,
    utc_now_iso,
)
from hypothesis_agent.research.metrics import DATASETS, METRICS

STATEMENT_SOURCES = {
    "YFINANCE_FINANCIALS": "income_statement",
    "YFINANCE_BALANCE_SHEET": "balance_sheet",
    "YFINANCE_CASH_FLOW": "cash_flow",
}


@dataclass
class ExtractionResult:
    snapshot_id: str
    registered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    periods: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def dataset_name(tool_slug: str, arguments: Dict[str, Any]) -> str:
    for name, (slug, extra) in DATASETS.items():
        if slug != tool_slug:
            continue
        if "quarterly" in extra and bool(arguments.get("quarterly", False)) != extra["quarterly"]:
            continue
        return name
    return tool_slug.lower()


def build_snapshot(
    *,
    ticker: str,
    tool_slug: str,
    parameters: Dict[str, Any],
    payload: Any,
    retrieval_function: str,
    artifact_path: Optional[str] = None,
    artifact_uri: Optional[str] = None,
    source_meta: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> DataSnapshot:
    """Describe a retrieval. ``retrieved_at`` is when the data left Yahoo, not when it was replayed."""

    source_meta = dict(source_meta or {})
    mode = source_meta.pop("mode", "live")
    replayed_at = utc_now_iso()
    retrieved_at = source_meta.get("captured_at") if mode == "frozen_fixture" and source_meta.get("captured_at") else replayed_at
    if mode == "frozen_fixture":
        source_meta["replayed_at"] = replayed_at
    content_hash = sha256_of(payload) if error is None else None
    rows = _rows_with_dates(payload) if error is None else []
    dates = sorted(d for d, _ in rows if d)
    status = "error" if error else ("empty" if payload in (None, [], {}) else "ok")
    return DataSnapshot(
        snapshot_id=EvidenceRegistry.make_snapshot_id(tool_slug, parameters, content_hash or error),
        ticker=ticker,
        dataset=dataset_name(tool_slug, parameters),
        tool_slug=tool_slug,
        retrieval_function=retrieval_function,
        parameters=dict(parameters),
        retrieved_at=retrieved_at,
        source_mode=mode if mode in {"live", "frozen_fixture", "custom_toolset"} else "custom_toolset",
        source=source_meta.get("source_label", "Yahoo Finance via yfinance"),
        source_detail=source_meta,
        content_sha256=content_hash,
        artifact_path=artifact_path,
        artifact_uri=artifact_uri,
        record_count=_record_count(payload) if error is None else 0,
        coverage_start=dates[0] if dates else None,
        coverage_end=dates[-1] if dates else None,
        status=status,
        error=error,
    )


def extract_evidence(
    registry: EvidenceRegistry,
    snapshot: DataSnapshot,
    payload: Any,
    *,
    currency: Optional[str] = None,
) -> ExtractionResult:
    """Register every canonical metric found in ``payload`` against ``snapshot``."""

    result = ExtractionResult(snapshot_id=snapshot.snapshot_id)
    if snapshot.status != "ok":
        return result
    if snapshot.tool_slug in STATEMENT_SOURCES:
        _extract_statement(registry, snapshot, payload, currency, result)
    elif snapshot.tool_slug == "YFINANCE_COMPANY_INFO":
        _extract_company_info(registry, snapshot, payload, result)
    elif snapshot.tool_slug == "YFINANCE_HISTORICAL_PRICES":
        _extract_prices(registry, snapshot, payload, currency, result)
    return result


def detect_currency(payload: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (statement currency, trading currency) from a company-info payload."""

    info = _info_dict(payload)
    if not info:
        return None, None
    return info.get("financialCurrency"), info.get("currency")


# ----------------------------------------------------------------------------------------------
# Statements
# ----------------------------------------------------------------------------------------------
def _extract_statement(
    registry: EvidenceRegistry,
    snapshot: DataSnapshot,
    payload: Any,
    currency: Optional[str],
    result: ExtractionResult,
) -> None:
    source = STATEMENT_SOURCES[snapshot.tool_slug]
    quarterly = bool(snapshot.parameters.get("quarterly", False))
    period_type = PeriodType.QUARTERLY if quarterly else PeriodType.ANNUAL
    definitions = [d for d in METRICS.values() if d.source == source]
    rows = _rows_with_dates(payload)
    if not rows:
        result.warnings.append("Statement payload had no dated rows that could be parsed.")
        for definition in definitions:
            registry.record_missing(
                ticker=snapshot.ticker, metric=definition.key, dataset=snapshot.dataset,
                reason="Statement payload contained no parseable periods.",
            )
            result.missing.append(definition.key)
        return
    for row_date, row in rows:
        label = _period_label(row_date, period_type)
        numeric_values = [v for k, v in row.items() if k != "date" and _finite(v) is not None]
        if not numeric_values:
            registry.record_missing(
                ticker=snapshot.ticker, metric="*", dataset=snapshot.dataset, period=label,
                reason="Yahoo Finance returned an empty column for this period.",
            )
            continue
        result.periods.append(label)
        for definition in definitions:
            field_name, value = _first_field(row, definition.fields)
            if field_name is None:
                registry.record_missing(
                    ticker=snapshot.ticker, metric=definition.key, dataset=snapshot.dataset, period=label,
                    reason=f"None of the fields {list(definition.fields)} were reported for this period.",
                )
                result.missing.append(f"{definition.key}@{row_date}")
                continue
            unit = _currency_unit(definition.unit, currency)
            notes = []
            if currency is None and "USD" in definition.unit:
                notes.append("Reporting currency not confirmed (company info unavailable); USD assumed.")
            record = registry.register_raw(
                snapshot_id=snapshot.snapshot_id,
                metric=definition.key,
                label=definition.label,
                value=value,
                unit=unit,
                currency=currency or ("USD" if "USD" in definition.unit else None),
                period=label,
                period_type=period_type,
                period_end=row_date,
                locator={"row_key": "date", "row": row_date, "field": field_name},
                notes=notes,
            )
            result.registered.append(record.evidence_id)


# ----------------------------------------------------------------------------------------------
# Company info
# ----------------------------------------------------------------------------------------------
def _extract_company_info(
    registry: EvidenceRegistry, snapshot: DataSnapshot, payload: Any, result: ExtractionResult
) -> None:
    info = _info_dict(payload)
    if not info:
        result.warnings.append("Company info payload was empty or not a mapping.")
        return
    as_of = snapshot.retrieved_at[:10]
    mrq = _epoch_to_date(info.get("mostRecentQuarter"))
    statement_currency = info.get("financialCurrency")
    trading_currency = info.get("currency")
    index_hint = 0 if isinstance(payload, list) else None
    for definition in (d for d in METRICS.values() if d.source == "company_info"):
        field_name, value = _first_field(info, definition.fields)
        if field_name is None:
            registry.record_missing(
                ticker=snapshot.ticker, metric=definition.key, dataset=snapshot.dataset,
                period=f"as of {as_of}", reason=f"Yahoo info field(s) {list(definition.fields)} not reported.",
            )
            result.missing.append(definition.key)
            continue
        if definition.period_type == "ttm":
            period_type = PeriodType.TTM
            period_end = mrq or as_of
            label = f"TTM to {mrq}" if mrq else f"TTM as of {as_of}"
        else:
            period_type = PeriodType.POINT_IN_TIME
            period_end = as_of
            label = f"As of {as_of}"
        currency = trading_currency if definition.unit == "USD/share" else statement_currency
        if definition.key in {"market_cap", "enterprise_value"}:
            currency = trading_currency
        locator: Dict[str, Any] = {"field": field_name}
        if index_hint is not None:
            locator = {"index": index_hint, "field": field_name}
        record = registry.register_raw(
            snapshot_id=snapshot.snapshot_id,
            metric=definition.key,
            label=definition.label,
            value=value,
            unit=_currency_unit(definition.unit, currency),
            currency=currency if "USD" in definition.unit else None,
            period=label,
            period_type=period_type,
            period_end=period_end,
            locator=locator,
            notes=["Value as published by Yahoo Finance; methodology not disclosed by the source."],
        )
        result.registered.append(record.evidence_id)


# ----------------------------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------------------------
def price_rows(payload: Any) -> List[Tuple[str, float]]:
    """Return (date, close) pairs from either the live list format or legacy dict formats."""

    rows: List[Tuple[str, float]] = []
    for row_date, row in _rows_with_dates(payload):
        close = _finite(row.get("close") if isinstance(row, dict) else None)
        if close is not None:
            rows.append((row_date, close))
    return rows


def _extract_prices(
    registry: EvidenceRegistry,
    snapshot: DataSnapshot,
    payload: Any,
    currency: Optional[str],
    result: ExtractionResult,
) -> None:
    rows = price_rows(payload)
    if not rows:
        result.warnings.append("Price payload contained no closing prices.")
        registry.record_missing(
            ticker=snapshot.ticker, metric="close_price", dataset=snapshot.dataset,
            reason="Price history returned no closing prices.",
        )
        return
    for row_date, close in (rows[0], rows[-1]):
        record = registry.register_raw(
            snapshot_id=snapshot.snapshot_id,
            metric="close_price",
            label="Closing price",
            value=close,
            unit=_currency_unit("USD/share", currency),
            currency=currency or "USD",
            period=f"Close on {row_date}",
            period_type=PeriodType.POINT_IN_TIME,
            period_end=row_date,
            locator={"row_key": "date", "row": row_date, "field": "close"},
            notes=["yfinance history() closes are split- and dividend-adjusted by default (auto_adjust=True)."],
        )
        result.registered.append(record.evidence_id)
    result.periods.append(f"{rows[0][0]} to {rows[-1][0]}")


def read_locator(payload: Any, locator: Dict[str, Any]) -> Optional[float]:
    """Re-read a raw value from a stored payload (used by reproducibility checks)."""

    if "row" in locator:
        for row_date, row in _rows_with_dates(payload):
            if row_date == locator["row"]:
                return _finite(row.get(locator["field"]))
        return None
    info = _info_dict(payload)
    if info is None:
        return None
    return _finite(info.get(locator.get("field")))


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _rows_with_dates(payload: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalise list-of-records, {date: {...}} and {"key": {date: {...}}} shapes."""

    rows: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("date"):
                rows.append((str(item["date"])[:10], item))
    elif isinstance(payload, dict):
        candidates: Iterable[Tuple[str, Any]] = payload.items()
        nested = [v for v in payload.values() if isinstance(v, dict) and v and all(_looks_like_date(k) for k in v)]
        if not any(_looks_like_date(k) for k in payload) and len(nested) == 1:
            candidates = nested[0].items()
        for key, value in candidates:
            if _looks_like_date(key) and isinstance(value, dict):
                rows.append((str(key)[:10], value))
    rows.sort(key=lambda item: item[0])
    return rows


def _info_dict(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return None


def _record_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        rows = _rows_with_dates(payload)
        return len(rows) if rows else len(payload)
    return 0 if payload is None else 1


def _looks_like_date(value: Any) -> bool:
    text = str(value)
    if len(text) < 10:
        return False
    try:
        datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_field(row: Dict[str, Any], fields: Iterable[str]) -> Tuple[Optional[str], Optional[float]]:
    for name in fields:
        value = _finite(row.get(name))
        if value is not None:
            return name, value
    return None, None


def _period_label(row_date: str, period_type: PeriodType) -> str:
    if period_type == PeriodType.ANNUAL:
        return f"FY{row_date[:4]} (ended {row_date})"
    return f"Quarter ended {row_date}"


def _currency_unit(unit: str, currency: Optional[str]) -> str:
    if currency and currency != "USD" and "USD" in unit:
        return unit.replace("USD", currency)
    return unit


def _epoch_to_date(value: Any) -> Optional[str]:
    number = _finite(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None

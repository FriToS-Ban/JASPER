"""Structured evidence registry: every number JASPER uses is a traceable record.

Two kinds of objects live here:

* :class:`DataSnapshot` describes one raw dataset retrieval (which yfinance call, which
  parameters, when, the SHA-256 of the exact payload, and where the payload is stored).
* :class:`EvidenceRecord` describes one metric value. Raw records point at a snapshot plus a
  locator (row/field) inside it; derived records carry a :class:`Derivation` naming the
  operation, formula and the evidence IDs of every input, so lineage can be walked back to raw
  data and recomputed.

The registry never stores a missing value as if it were collected: unavailable metrics are
recorded separately via :meth:`EvidenceRegistry.record_missing`.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class PeriodType(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    TTM = "ttm"
    POINT_IN_TIME = "point_in_time"
    RANGE = "range"


class DataSnapshot(BaseModel):
    """One raw dataset retrieval, stored verbatim so analyses can be reproduced later."""

    snapshot_id: str
    ticker: str
    dataset: str
    tool_slug: str
    retrieval_function: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: str
    source: str = "Yahoo Finance via yfinance"
    source_mode: Literal["live", "frozen_fixture", "custom_toolset"] = "live"
    source_detail: Dict[str, Any] = Field(default_factory=dict)
    content_sha256: Optional[str] = None
    artifact_path: Optional[str] = None
    artifact_uri: Optional[str] = None
    record_count: int = 0
    coverage_start: Optional[str] = None
    coverage_end: Optional[str] = None
    status: Literal["ok", "empty", "error"] = "ok"
    error: Optional[str] = None

    def describe(self) -> str:
        coverage = ""
        if self.coverage_start or self.coverage_end:
            coverage = f", covering {self.coverage_start or '?'} to {self.coverage_end or '?'}"
        mode = " (replayed from frozen fixture)" if self.source_mode == "frozen_fixture" else ""
        return f"{self.retrieval_function} retrieved {self.retrieved_at}{coverage}{mode}"


class Derivation(BaseModel):
    """How a derived metric was computed from other evidence records."""

    operation: str
    formula: str
    inputs: List[str] = Field(default_factory=list, description="Evidence IDs of the inputs, in formula order.")
    input_snapshots: List[str] = Field(default_factory=list, description="Snapshots read directly (series operations).")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class EvidenceRecord(BaseModel):
    """A single traceable metric value."""

    evidence_id: str
    kind: Literal["raw", "derived"]
    ticker: str
    metric: str
    label: str
    value: float
    unit: str
    currency: Optional[str] = None
    period: str
    period_type: PeriodType
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    source: str
    retrieved_at: str
    snapshot_id: Optional[str] = None
    source_tool: str
    source_function: str
    locator: Dict[str, Any] = Field(default_factory=dict)
    field_path: Optional[str] = None
    derivation: Optional[Derivation] = None
    qualifier: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

    @property
    def period_year(self) -> Optional[int]:
        if self.period_end:
            try:
                return int(self.period_end[:4])
            except ValueError:
                return None
        return None


class MissingEvidence(BaseModel):
    ticker: str
    metric: str
    dataset: Optional[str] = None
    period: Optional[str] = None
    reason: str
    recorded_at: str = Field(default_factory=utc_now_iso)


class EvidenceConflict(BaseModel):
    ticker: str
    metric: str
    period: str
    evidence_ids: List[str]
    values: List[float]
    description: str


class EvidenceRegistry:
    """In-memory registry with JSON round-tripping so it can travel in the stage context."""

    VALUE_CONFLICT_TOLERANCE = 1e-6

    def __init__(self) -> None:
        self.snapshots: Dict[str, DataSnapshot] = {}
        self.records: Dict[str, EvidenceRecord] = {}
        self.missing: List[MissingEvidence] = []
        self.conflicts: List[EvidenceConflict] = []

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    @staticmethod
    def make_snapshot_id(tool_slug: str, parameters: Dict[str, Any], content_sha256: Optional[str]) -> str:
        digest = sha256_of({"slug": tool_slug, "parameters": parameters, "content": content_sha256})
        return f"SNAP-{digest[:10]}"

    def register_snapshot(self, snapshot: DataSnapshot) -> DataSnapshot:
        existing = self.snapshots.get(snapshot.snapshot_id)
        if existing is not None:
            return existing
        self.snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def find_snapshot(self, tool_slug: str, parameters: Dict[str, Any]) -> Optional[DataSnapshot]:
        """Return a previously collected snapshot for the same call (used to avoid re-fetching)."""

        wanted = canonical_json(parameters)
        for snapshot in self.snapshots.values():
            if snapshot.tool_slug == tool_slug and canonical_json(snapshot.parameters) == wanted:
                return snapshot
        return None

    # ------------------------------------------------------------------
    # Evidence records
    # ------------------------------------------------------------------
    @staticmethod
    def _make_evidence_id(parts: Dict[str, Any]) -> str:
        return f"EV-{sha256_of(parts)[:8]}"

    def register_raw(
        self,
        *,
        snapshot_id: str,
        metric: str,
        label: str,
        value: Any,
        unit: str,
        period: str,
        period_type: PeriodType,
        locator: Dict[str, Any],
        currency: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> EvidenceRecord:
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            raise KeyError(f"Cannot register raw evidence: unknown snapshot {snapshot_id}")
        number = _require_finite(value, metric)
        evidence_id = self._make_evidence_id(
            {"kind": "raw", "ticker": snapshot.ticker, "metric": metric, "period_type": period_type.value,
             "period_end": period_end, "snapshot": snapshot_id, "locator": locator}
        )
        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind="raw",
            ticker=snapshot.ticker,
            metric=metric,
            label=label,
            value=number,
            unit=unit,
            currency=currency,
            period=period,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            source=snapshot.source,
            retrieved_at=snapshot.retrieved_at,
            snapshot_id=snapshot_id,
            source_tool=snapshot.tool_slug,
            source_function=snapshot.retrieval_function,
            locator=dict(locator),
            field_path=_format_locator(locator),
            notes=list(notes or []),
        )
        return self._store(record)

    def register_derived(
        self,
        *,
        ticker: str,
        metric: str,
        label: str,
        value: Any,
        unit: str,
        period: str,
        period_type: PeriodType,
        derivation: Derivation,
        currency: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        qualifier: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> EvidenceRecord:
        missing_inputs = [eid for eid in derivation.inputs if eid not in self.records]
        if missing_inputs:
            raise KeyError(f"Derived metric {metric} references unknown evidence: {missing_inputs}")
        missing_snapshots = [sid for sid in derivation.input_snapshots if sid not in self.snapshots]
        if missing_snapshots:
            raise KeyError(f"Derived metric {metric} references unknown snapshots: {missing_snapshots}")
        number = _require_finite(value, metric)
        evidence_id = self._make_evidence_id(
            {"kind": "derived", "ticker": ticker, "metric": metric, "period_type": period_type.value,
             "period_start": period_start, "period_end": period_end, "operation": derivation.operation,
             "inputs": derivation.inputs, "snapshots": derivation.input_snapshots,
             "parameters": derivation.parameters}
        )
        input_records = [self.records[eid] for eid in derivation.inputs]
        retrieved_candidates = [r.retrieved_at for r in input_records]
        retrieved_candidates += [self.snapshots[sid].retrieved_at for sid in derivation.input_snapshots]
        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind="derived",
            ticker=ticker,
            metric=metric,
            label=label,
            value=number,
            unit=unit,
            currency=currency,
            period=period,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            source="JASPER deterministic analytics",
            retrieved_at=max(retrieved_candidates) if retrieved_candidates else utc_now_iso(),
            snapshot_id=None,
            source_tool=f"analytics.{derivation.operation}",
            source_function=derivation.formula,
            derivation=derivation,
            qualifier=qualifier,
            notes=list(notes or []),
        )
        return self._store(record)

    def _store(self, record: EvidenceRecord) -> EvidenceRecord:
        existing = self.records.get(record.evidence_id)
        if existing is not None:
            if not _close(existing.value, record.value):
                raise ValueError(
                    f"Evidence {record.evidence_id} re-registered with a different value "
                    f"({existing.value} vs {record.value})"
                )
            return existing
        for other in self.records.values():
            if (
                other.ticker == record.ticker
                and other.metric == record.metric
                and other.period_type == record.period_type
                and other.period_end == record.period_end
                and other.period_start == record.period_start
                and not _close(other.value, record.value)
            ):
                self.conflicts.append(
                    EvidenceConflict(
                        ticker=record.ticker,
                        metric=record.metric,
                        period=record.period,
                        evidence_ids=[other.evidence_id, record.evidence_id],
                        values=[other.value, record.value],
                        description=(
                            f"{record.label} for {record.period} differs between {other.evidence_id} "
                            f"({other.value:g}) and {record.evidence_id} ({record.value:g})."
                        ),
                    )
                )
                break
        self.records[record.evidence_id] = record
        return record

    def remove(self, evidence_id: str) -> None:
        """Remove a record (used when reanalysis discards an inconsistent derived value)."""

        dependants = [r.evidence_id for r in self.records.values() if r.derivation and evidence_id in r.derivation.inputs]
        for dependant in dependants:
            self.remove(dependant)
        self.records.pop(evidence_id, None)

    def record_missing(
        self,
        *,
        ticker: str,
        metric: str,
        reason: str,
        dataset: Optional[str] = None,
        period: Optional[str] = None,
    ) -> MissingEvidence:
        for item in self.missing:
            if item.ticker == ticker and item.metric == metric and item.period == period and item.dataset == dataset:
                return item
        item = MissingEvidence(ticker=ticker, metric=metric, dataset=dataset, period=period, reason=reason)
        self.missing.append(item)
        return item

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, evidence_id: str) -> EvidenceRecord:
        try:
            return self.records[evidence_id]
        except KeyError as exc:
            raise KeyError(f"Evidence {evidence_id} not found in registry") from exc

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self.records

    def find(
        self,
        *,
        metric: Optional[str] = None,
        ticker: Optional[str] = None,
        period_type: Optional[PeriodType] = None,
        kind: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[EvidenceRecord]:
        results = [
            record
            for record in self.records.values()
            if (metric is None or record.metric == metric)
            and (ticker is None or record.ticker == ticker)
            and (period_type is None or record.period_type == period_type)
            and (kind is None or record.kind == kind)
            and (year is None or record.period_year == year)
        ]
        return sorted(results, key=lambda r: (r.period_end or "", r.period_start or ""))

    def series(self, ticker: str, metric: str, period_type: PeriodType = PeriodType.ANNUAL) -> List[EvidenceRecord]:
        """Return one record per period (latest retrieval wins), ordered oldest to newest."""

        by_period: Dict[str, EvidenceRecord] = {}
        for record in self.find(metric=metric, ticker=ticker, period_type=period_type):
            key = record.period_end or record.period
            current = by_period.get(key)
            if current is None or record.retrieved_at >= current.retrieved_at:
                by_period[key] = record
        return [by_period[key] for key in sorted(by_period)]

    def latest(self, ticker: str, metric: str, period_type: Optional[PeriodType] = None) -> Optional[EvidenceRecord]:
        records = self.find(metric=metric, ticker=ticker, period_type=period_type)
        return records[-1] if records else None

    def metrics_available(self, ticker: Optional[str] = None) -> List[str]:
        return sorted({r.metric for r in self.records.values() if ticker is None or r.ticker == ticker})

    def lineage(self, evidence_id: str) -> Dict[str, Any]:
        """Return the full derivation tree of a record down to raw snapshots."""

        record = self.get(evidence_id)
        node: Dict[str, Any] = {
            "evidence_id": record.evidence_id,
            "metric": record.metric,
            "value": record.value,
            "unit": record.unit,
            "period": record.period,
            "kind": record.kind,
        }
        if record.kind == "raw":
            snapshot = self.snapshots.get(record.snapshot_id or "")
            node["snapshot_id"] = record.snapshot_id
            node["field_path"] = record.field_path
            node["retrieval"] = snapshot.describe() if snapshot else None
        elif record.derivation is not None:
            node["operation"] = record.derivation.operation
            node["formula"] = record.derivation.formula
            node["parameters"] = record.derivation.parameters
            node["input_snapshots"] = list(record.derivation.input_snapshots)
            node["inputs"] = [self.lineage(eid) for eid in record.derivation.inputs]
        return node

    def raw_snapshot_ids(self, evidence_id: str) -> List[str]:
        """Return every snapshot that a record ultimately depends on."""

        record = self.get(evidence_id)
        if record.kind == "raw":
            return [record.snapshot_id] if record.snapshot_id else []
        found: List[str] = []
        if record.derivation is not None:
            found.extend(record.derivation.input_snapshots)
            for eid in record.derivation.inputs:
                found.extend(self.raw_snapshot_ids(eid))
        return list(dict.fromkeys(found))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshots": [s.model_dump(mode="json") for s in self.snapshots.values()],
            "records": [r.model_dump(mode="json") for r in self.records.values()],
            "missing": [m.model_dump(mode="json") for m in self.missing],
            "conflicts": [c.model_dump(mode="json") for c in self.conflicts],
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "EvidenceRegistry":
        registry = cls()
        if not payload:
            return registry
        for item in payload.get("snapshots", []):
            snapshot = DataSnapshot.model_validate(item)
            registry.snapshots[snapshot.snapshot_id] = snapshot
        for item in payload.get("records", []):
            record = EvidenceRecord.model_validate(item)
            registry.records[record.evidence_id] = record
        registry.missing = [MissingEvidence.model_validate(m) for m in payload.get("missing", [])]
        registry.conflicts = [EvidenceConflict.model_validate(c) for c in payload.get("conflicts", [])]
        return registry

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.records)

    def __iter__(self) -> Iterable[EvidenceRecord]:  # pragma: no cover - convenience
        return iter(list(self.records.values()))


def _require_finite(value: Any, metric: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Metric {metric} value must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {metric} value {value!r} is not numeric; record it as missing instead") from exc
    if not math.isfinite(number):
        raise ValueError(f"Metric {metric} value {value!r} is not finite; record it as missing instead")
    return number


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=EvidenceRegistry.VALUE_CONFLICT_TOLERANCE, abs_tol=1e-12)


def _format_locator(locator: Dict[str, Any]) -> str:
    if not locator:
        return ""
    parts = []
    if "row" in locator:
        parts.append(f"[{locator['row_key']}={locator['row']}]" if "row_key" in locator else f"[{locator['row']}]")
    if "index" in locator:
        parts.append(f"[{locator['index']}]")
    if "field" in locator:
        parts.append(f"[{json.dumps(locator['field'])}]")
    return "".join(parts)

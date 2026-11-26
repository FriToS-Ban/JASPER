"""Evidence registry, extraction, analytics lineage and the validated toolbox."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hypothesis_agent.research import analytics
from hypothesis_agent.research.evidence import Derivation, EvidenceRegistry, PeriodType
from hypothesis_agent.research.extraction import build_snapshot, extract_evidence
from hypothesis_agent.research.tools import ResearchToolbox
from hypothesis_agent.storage.artifact_store import ArtifactStore
from tests.research_helpers import B, SyntheticToolSet, company_payloads


def _snapshot(registry: EvidenceRegistry, payload, slug="YFINANCE_FINANCIALS", params=None):
    params = params or {"symbol": "TEST", "quarterly": False}
    snapshot = build_snapshot(ticker="TEST", tool_slug=slug, parameters=params, payload=payload,
                              retrieval_function="yfinance.Ticker('TEST').financials")
    registry.register_snapshot(snapshot)
    return snapshot


@pytest.fixture
def toolbox(tmp_path: Path) -> ResearchToolbox:
    return ResearchToolbox(
        registry=EvidenceRegistry(), toolset=SyntheticToolSet(), artifact_store=ArtifactStore.from_path(str(tmp_path)),
        workflow_id="wf", allowed_tickers=["TEST"], call_log=[], currencies={}, charts=[], max_calls=40,
    )


def test_raw_evidence_registration_preserves_source_and_locator() -> None:
    registry = EvidenceRegistry()
    payload = company_payloads()["YFINANCE_FINANCIALS|annual"]
    snapshot = _snapshot(registry, payload)
    result = extract_evidence(registry, snapshot, payload, currency="USD")
    revenue = registry.series("TEST", "total_revenue")
    assert [r.value for r in revenue] == [100 * B, 110 * B, 121 * B, 133.1 * B]
    record = revenue[-1]
    assert record.kind == "raw"
    assert record.snapshot_id == snapshot.snapshot_id
    assert record.locator == {"row_key": "date", "row": "2025-12-31", "field": "Total Revenue"}
    assert record.source_function == "yfinance.Ticker('TEST').financials"
    assert record.period == "FY2025 (ended 2025-12-31)" and record.period_type == PeriodType.ANNUAL
    assert record.currency == "USD" and record.unit == "USD"
    assert snapshot.content_sha256 and snapshot.coverage_start == "2022-12-31" and snapshot.coverage_end == "2025-12-31"
    assert record.evidence_id in result.registered


def test_snapshot_ids_are_content_addressed() -> None:
    payload = company_payloads()["YFINANCE_FINANCIALS|annual"]
    first = build_snapshot(ticker="TEST", tool_slug="YFINANCE_FINANCIALS", parameters={"symbol": "TEST"}, payload=payload, retrieval_function="f")
    second = build_snapshot(ticker="TEST", tool_slug="YFINANCE_FINANCIALS", parameters={"symbol": "TEST"}, payload=payload, retrieval_function="f")
    changed = json.loads(json.dumps(payload))
    changed[0]["Total Revenue"] = 1.0
    third = build_snapshot(ticker="TEST", tool_slug="YFINANCE_FINANCIALS", parameters={"symbol": "TEST"}, payload=changed, retrieval_function="f")
    assert first.snapshot_id == second.snapshot_id
    assert first.content_sha256 != third.content_sha256 and first.snapshot_id != third.snapshot_id


def test_raw_evidence_requires_known_snapshot_and_finite_value() -> None:
    registry = EvidenceRegistry()
    with pytest.raises(KeyError):
        registry.register_raw(snapshot_id="SNAP-missing", metric="total_revenue", label="Revenue", value=1.0, unit="USD",
                              period="FY2025", period_type=PeriodType.ANNUAL, locator={})
    snapshot = _snapshot(registry, [{"date": "2025-12-31", "Total Revenue": 5.0}])
    for bad in (None, float("nan"), float("inf"), "n/a"):
        with pytest.raises(ValueError):
            registry.register_raw(snapshot_id=snapshot.snapshot_id, metric="total_revenue", label="Revenue", value=bad,
                                  unit="USD", period="FY2025", period_type=PeriodType.ANNUAL, locator={"field": "x"})


def test_missing_fields_are_recorded_not_stored() -> None:
    registry = EvidenceRegistry()
    payload = [{"date": "2025-12-31", "Total Revenue": 10.0, "Net Income": None},
               {"date": "2021-12-31", "Total Revenue": None, "Net Income": None}]
    snapshot = _snapshot(registry, payload)
    extract_evidence(registry, snapshot, payload)
    assert registry.find(metric="net_income") == []
    reasons = {(m.metric, m.period) for m in registry.missing}
    assert ("net_income", "FY2025 (ended 2025-12-31)") in reasons
    assert ("*", "FY2021 (ended 2021-12-31)") in reasons  # empty column recorded once, not as zeros
    assert registry.find(metric="total_revenue")[0].notes  # currency assumption is disclosed


def test_duplicate_registration_is_deduplicated_and_conflicts_detected() -> None:
    registry = EvidenceRegistry()
    payload = [{"date": "2025-12-31", "Total Revenue": 10.0}]
    snapshot = _snapshot(registry, payload)
    extract_evidence(registry, snapshot, payload)
    extract_evidence(registry, snapshot, payload)
    assert len(registry.find(metric="total_revenue")) == 1
    other_payload = [{"date": "2025-12-31", "Total Revenue": 12.0}]
    other = _snapshot(registry, other_payload, params={"symbol": "TEST", "quarterly": False, "variant": 2})
    extract_evidence(registry, other, other_payload)
    assert len(registry.conflicts) == 1
    assert sorted(registry.conflicts[0].values) == [10.0, 12.0]


def test_derived_metric_lineage_and_recompute() -> None:
    registry = EvidenceRegistry()
    payload = company_payloads()["YFINANCE_FINANCIALS|annual"]
    extract_evidence(registry, _snapshot(registry, payload), payload, currency="USD")
    margins = analytics.compute_ratio(registry, "TEST", "operating_margin")
    assert [round(m.value, 4) for m in margins] == [0.10, 0.12, 0.14, 0.16]
    latest = margins[-1]
    assert latest.derivation.operation == "ratio"
    inputs = [registry.get(e) for e in latest.derivation.inputs]
    assert [i.metric for i in inputs] == ["operating_income", "total_revenue"]
    lineage = registry.lineage(latest.evidence_id)
    assert lineage["inputs"][0]["snapshot_id"] and lineage["inputs"][0]["field_path"] == '[date=2025-12-31]["Operating Income"]'
    assert math.isclose(analytics.recompute(registry, latest), latest.value)
    with pytest.raises(KeyError):
        registry.register_derived(ticker="TEST", metric="operating_margin", label="x", value=0.1, unit="ratio",
                                  period="FY2025", period_type=PeriodType.ANNUAL,
                                  derivation=Derivation(operation="ratio", formula="a/b", inputs=["EV-00000000"]))


def test_tampered_derived_value_is_detected_and_removal_cascades() -> None:
    registry = EvidenceRegistry()
    payload = company_payloads()["YFINANCE_FINANCIALS|annual"]
    extract_evidence(registry, _snapshot(registry, payload), payload, currency="USD")
    margin = analytics.compute_ratio(registry, "TEST", "operating_margin")[-1]
    trend = analytics.assess_trend(registry, "TEST", "operating_margin")
    assert analytics.check_consistency(registry) == []
    registry.records[margin.evidence_id] = margin.model_copy(update={"value": 0.5})
    problems = analytics.check_consistency(registry)
    assert any(p["evidence_id"] == margin.evidence_id for p in problems)
    registry.remove(margin.evidence_id)
    assert not registry.has(margin.evidence_id) and not registry.has(trend.evidence_id)


def test_trend_records_latest_period_reversal() -> None:
    registry = EvidenceRegistry()
    payload = [{"date": d, "Free Cash Flow": v} for d, v in
               [("2022-12-31", 1.0), ("2023-12-31", 5.0), ("2024-12-31", 9.0), ("2025-12-31", 6.0)]]
    extract_evidence(registry, _snapshot(registry, payload, slug="YFINANCE_CASH_FLOW"), payload)
    trend = analytics.assess_trend(registry, "TEST", "free_cash_flow")
    assert trend.qualifier == "increasing, latest period reversed"
    assert trend.derivation.parameters["steps"] == ["increasing", "increasing", "decreasing"]


def test_equity_ratios_refuse_negative_equity() -> None:
    registry = EvidenceRegistry()
    payload = company_payloads(negative_equity=True)["YFINANCE_BALANCE_SHEET|annual"]
    extract_evidence(registry, _snapshot(registry, payload, slug="YFINANCE_BALANCE_SHEET"), payload)
    assert analytics.compute_ratio(registry, "TEST", "debt_to_equity") == []
    assert any("negative" in m.reason for m in registry.missing if m.metric == "debt_to_equity")


def test_registry_round_trips_through_json() -> None:
    registry = EvidenceRegistry()
    payload = company_payloads()["YFINANCE_FINANCIALS|annual"]
    extract_evidence(registry, _snapshot(registry, payload), payload)
    analytics.compute_ratio(registry, "TEST", "net_margin")
    restored = EvidenceRegistry.from_dict(json.loads(json.dumps(registry.to_dict())))
    assert restored.records.keys() == registry.records.keys()
    assert restored.snapshots.keys() == registry.snapshots.keys()


# ---------------------------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------------------------
def test_toolbox_fetch_stores_payload_and_skips_duplicates(toolbox: ResearchToolbox) -> None:
    first = toolbox.invoke("fetch_dataset", {"dataset": "income_statement_annual", "ticker": "TEST"}, origin="test")
    assert first.status == "ok" and first.evidence_ids
    snapshot = toolbox.registry.snapshots[first.output["snapshot_id"]]
    assert Path(snapshot.artifact_path).exists()
    assert snapshot.source_mode == "custom_toolset"  # provenance of a non-Yahoo toolset is not overstated
    second = toolbox.invoke("fetch_dataset", {"dataset": "income_statement_annual", "ticker": "TEST"}, origin="test")
    assert second.status == "skipped"
    assert len(toolbox.toolset.invocations) == 1
    assert [c["status"] for c in toolbox.call_log] == ["ok", "skipped"]


def test_toolbox_rejects_invalid_calls(toolbox: ResearchToolbox) -> None:
    assert toolbox.invoke("run_shell", {"cmd": "ls"}, origin="llm").status == "rejected"
    assert toolbox.invoke("fetch_dataset", {"dataset": "income_statement_annual", "ticker": "TEST", "path": "/etc"}, origin="llm").status == "rejected"
    assert toolbox.invoke("fetch_dataset", {"dataset": "secret_db", "ticker": "TEST"}, origin="llm").status == "rejected"
    outside = toolbox.invoke("fetch_dataset", {"dataset": "company_info", "ticker": "AAPL"}, origin="llm")
    assert outside.status == "rejected" and "outside the research universe" in outside.error
    assert toolbox.invoke("compute_metric", {"metric": "made_up", "ticker": "TEST"}, origin="llm").status == "error"
    assert all(c["origin"] == "llm" for c in toolbox.call_log)


def test_toolbox_budget_is_enforced(toolbox: ResearchToolbox) -> None:
    toolbox.max_calls = 2
    toolbox.invoke("list_missing", {"ticker": "TEST"}, origin="test")
    toolbox.invoke("list_missing", {"ticker": "TEST"}, origin="test")
    third = toolbox.invoke("list_missing", {"ticker": "TEST"}, origin="test")
    assert third.status == "rejected" and "budget" in third.error


def test_failed_retrieval_is_recorded_as_error_snapshot(toolbox: ResearchToolbox) -> None:
    result = toolbox.invoke("fetch_dataset", {"dataset": "news", "ticker": "TEST"}, origin="test")
    assert result.status == "error"
    snapshot = next(s for s in toolbox.registry.snapshots.values() if s.dataset == "news")
    assert snapshot.status == "error" and "no YFINANCE_NEWS" in snapshot.error
    assert any(m.dataset == "news" for m in toolbox.registry.missing)


def test_toolbox_analytics_and_price_metrics(toolbox: ResearchToolbox) -> None:
    for dataset in ("company_info", "income_statement_annual", "cash_flow_annual", "price_history"):
        toolbox.invoke("fetch_dataset", {"dataset": dataset, "ticker": "TEST", "period": "1y"}, origin="test")
    growth = toolbox.invoke("compute_growth", {"metric": "total_revenue", "ticker": "TEST"}, origin="test")
    assert growth.status == "ok"
    assert [round(r["value"], 6) for r in growth.output["records"]] == [0.1, 0.1, 0.1]
    conversion = toolbox.invoke("compute_metric", {"metric": "cash_conversion", "ticker": "TEST"}, origin="test")
    assert [round(r["value"], 4) for r in conversion.output["records"]] == [1.5, 1.3, 1.0, 0.6429]
    price = toolbox.invoke("price_performance", {"ticker": "TEST"}, origin="test")
    assert price.status == "ok"
    price_return = toolbox.registry.find(metric="price_return")[0]
    assert price_return.derivation.input_snapshots and math.isclose(analytics.recompute(toolbox.registry, price_return), price_return.value)
    chart = toolbox.invoke("plot_metric", {"metric": "operating_margin", "ticker": "TEST"}, origin="test")
    assert chart.status == "error"  # margin not yet computed: charts never invent data
    toolbox.invoke("compute_metric", {"metric": "operating_margin", "ticker": "TEST"}, origin="test")
    chart = toolbox.invoke("plot_metric", {"metric": "operating_margin", "ticker": "TEST"}, origin="test")
    assert chart.status == "ok"
    spec = toolbox.charts[0]
    assert spec["snapshot_ids"] and "Source snapshots:" in spec["caption"] and "Retrieval:" in spec["caption"]
    assert "Operating Income / Total Revenue" in spec["caption"] and spec["evidence_ids"][0] in spec["caption"]

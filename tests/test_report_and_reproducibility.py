"""PDF content, fixture replay and run reproducibility."""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.orchestration.fixture_tools import FixtureDataUnavailable, FixtureToolSet
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.reporting.pdf_report import SECTION_TITLES, render_research_report
from hypothesis_agent.research.offline_llm import DeterministicResearchLLM
from hypothesis_agent.research.reproduce import verify_run_directory
from hypothesis_agent.storage.artifact_store import ArtifactStore
from tests.research_helpers import ScriptedLLM, SyntheticToolSet, company_payloads, make_request

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "v1" / "fixtures"


def _run(tmp_path: Path, text: str, *, llm=None, toolset=None, workflow_id: str = "wf-pdf"):
    settings = AppSettings(artifact_store_path=str(tmp_path / "artifacts"), llm_provider="offline")
    store = ArtifactStore.from_path(settings.artifact_store_path)
    orchestrator = LangGraphValidationOrchestrator(settings=settings, llm=llm or DeterministicResearchLLM(), artifact_store=store,
                                                   toolset=toolset or SyntheticToolSet())
    context = orchestrator.run_research(make_request(text), {"metadata": {"workflow_id": workflow_id}})
    return context, store


def _pdf_text(context, store, tmp_path: Path) -> str:
    """Re-render the report without stream compression so its text can be asserted on."""

    uri, sections = render_research_report(request=make_request("x"), workflow_id="wf-render", state=context["research"],
                                           artifact_store=ArtifactStore.from_path(str(tmp_path / "render")), compress=False)
    path = Path(urllib.request.url2pathname(urlparse(uri).path))
    return path.read_bytes().decode("latin-1"), sections


def test_pdf_contains_all_audit_sections_and_attribution(tmp_path: Path) -> None:
    context, store = _run(tmp_path, "TEST's operating margins are improving.")
    text, sections = _pdf_text(context, store, tmp_path)
    assert sections == SECTION_TITLES
    for title in SECTION_TITLES:
        assert title in text
    state = context["research"]
    snapshot_id = state["registry"]["snapshots"][0]["snapshot_id"]
    assert snapshot_id in text
    chart = state["charts"][0]
    assert chart["evidence_ids"][0] in text and "Source snapshots:" in text  # chart caption with lineage
    assert "Calculation: Operating Income / Total Revenue" in text
    assert "does not report a numeric confidence score" in text
    assert "Evidence balance" in text and "Source quality" in text
    assert "REVISE_PLAN" in text  # iteration history shows the plan change
    original = Path(urllib.request.url2pathname(urlparse(context["report"]["pdf_path"]).path))
    assert original.exists() and original.read_bytes()[:4] == b"%PDF"


def test_pdf_lists_unsupported_and_contradicted_claims(tmp_path: Path) -> None:
    llm = ScriptedLLM(narratives=[{"narrative": "Revenue was $150 billion in FY2025. Debt was $99 billion in FY2025. "
                                                 "Brand loyalty remains strong."}])
    context, store = _run(tmp_path, "TEST's operating margins are improving.", llm=llm,
                          toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    text, _ = _pdf_text(context, store, tmp_path)
    assert "CONTRADICTED" in text and "UNSUPPORTED" in text and "NOT_VERIFIABLE" in text
    assert "not proof the claim is false" in text
    assert "All checked claims passed verification" not in text


def test_pdf_states_when_all_claims_pass(tmp_path: Path) -> None:
    llm = ScriptedLLM(narratives=[{"narrative": "Revenue was $133.1 billion in FY2025."}])
    context, store = _run(tmp_path, "TEST's operating margins are improving.", llm=llm,
                          toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    text, _ = _pdf_text(context, store, tmp_path)
    assert "All checked claims passed verification." in text


def test_pdf_reports_missing_data(tmp_path: Path) -> None:
    payloads = company_payloads()
    payloads.pop("YFINANCE_CASH_FLOW|annual")
    context, store = _run(tmp_path, "TEST's free cash flow is improving.", toolset=SyntheticToolSet(payloads))
    text, _ = _pdf_text(context, store, tmp_path)
    assert context["research"]["verdict"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "Unavailable metric" in text and "Failed retrievals" in text
    assert "Synthetic data source has no YFINANCE_CASH_FLOW" in text


def test_fixture_toolset_replays_and_slices(tmp_path: Path) -> None:
    toolset = FixtureToolSet(FIXTURES)
    full = toolset.get_tool("YFINANCE_HISTORICAL_PRICES").invoke({"symbol": "KO", "period": "5y"})
    one_year = toolset.get_tool("YFINANCE_HISTORICAL_PRICES").invoke({"symbol": "KO", "period": "1y"})
    assert len(full) > 1000 and 240 <= len(one_year) <= 260 and one_year[-1] == full[-1]
    assert toolset.source_metadata("YFINANCE_FINANCIALS", {"symbol": "KO"})["mode"] == "frozen_fixture"
    with pytest.raises(FixtureDataUnavailable):
        toolset.get_tool("YFINANCE_NEWS").invoke({"symbol": "KO"})  # captured outage replays as an error
    with pytest.raises(FixtureDataUnavailable):
        toolset.get_tool("YFINANCE_FINANCIALS").invoke({"symbol": "ZZZZ"})


def test_run_directory_reproduces_and_detects_tampering(tmp_path: Path) -> None:
    context, store = _run(tmp_path, "TEST's operating margins are improving.", workflow_id="wf-repro")
    run_dir = tmp_path / "artifacts" / "wf-repro"
    report = verify_run_directory(run_dir)
    assert report["reproducible"] and report["raw_records_checked"] > 50 and report["derived_records_checked"] > 10

    tampered = tmp_path / "tampered"
    shutil.copytree(run_dir, tampered)
    payload_file = next(tampered.glob("snapshot_TEST_income_statement_annual_*.json"))
    rows = json.loads(payload_file.read_text(encoding="utf-8"))
    rows[0]["Total Revenue"] = rows[0]["Total Revenue"] * 2
    payload_file.write_text(json.dumps(rows), encoding="utf-8")
    kinds = {p["kind"] for p in verify_run_directory(tampered)["problems"]}
    assert {"hash_mismatch", "raw_mismatch"} <= kinds

    registry_file = tampered / "evidence_registry.json"
    shutil.copy(run_dir / "evidence_registry.json", registry_file)
    payload_file.write_text((run_dir / payload_file.name).read_text(encoding="utf-8"), encoding="utf-8")
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    derived = next(r for r in registry["records"] if r["kind"] == "derived" and r["metric"] == "operating_margin")
    derived["value"] = 0.42
    registry_file.write_text(json.dumps(registry), encoding="utf-8")
    problems = verify_run_directory(tampered)["problems"]
    assert {p["kind"] for p in problems} == {"derived_mismatch"}
    # The tampered record is caught, and so is the trend recomputed from it.
    assert derived["evidence_id"] in {p["evidence_id"] for p in problems}

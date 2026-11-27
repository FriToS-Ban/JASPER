"""End-to-end behaviour of the research agent's conditional, bounded loop."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.research.evidence import EvidenceRegistry
from hypothesis_agent.research.offline_llm import DeterministicResearchLLM
from hypothesis_agent.research.reproduce import verify_run_directory
from hypothesis_agent.storage.artifact_store import ArtifactStore
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient
from tests.research_helpers import ScriptedLLM, SyntheticToolSet, company_payloads, make_request


def _orchestrator(tmp_path: Path, *, llm=None, toolset=None, **settings: Any) -> LangGraphValidationOrchestrator:
    config = AppSettings(artifact_store_path=str(tmp_path / "artifacts"), llm_provider="offline", **settings)
    return LangGraphValidationOrchestrator(settings=config, llm=llm or DeterministicResearchLLM(),
                                           artifact_store=ArtifactStore.from_path(config.artifact_store_path),
                                           toolset=toolset or SyntheticToolSet())


def _run(orchestrator, text: str, workflow_id: str = "wf-test") -> Dict[str, Any]:
    return orchestrator.run_research(make_request(text), {"metadata": {"workflow_id": workflow_id}})


def test_first_pass_research_completes_without_loops(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = _run(orchestrator, "TEST's operating margins are improving.")
    state = context["research"]
    assert context["metadata"]["route"] == ["plan_generation", "data_collection", "hybrid_analysis", "critique",
                                            "detailed_analysis", "claim_verification", "report_generation"]
    assert [it["decision"] for it in state["iterations"]] == ["CONTINUE"]
    assert state["verdict"]["verdict"] == "SUPPORTED"
    assert state["claims"]["counts"].get("VERIFIED", 0) > 0
    assert set(state["claims"]["counts"]) <= {"VERIFIED", "NOT_VERIFIABLE"}
    assert verify_run_directory(tmp_path / "artifacts" / "wf-test")["reproducible"]


def test_contradictory_evidence_revises_plan_and_collects_more_data(tmp_path: Path) -> None:
    toolset = SyntheticToolSet()  # margins up, cash conversion and FCF down
    context = _run(_orchestrator(tmp_path, toolset=toolset), "TEST's operating margins are improving.")
    state = context["research"]
    assert [it["decision"] for it in state["iterations"]] == ["REVISE_PLAN", "CONTINUE"]
    first = state["iterations"][0]
    assert "CONTRADICTORY_EVIDENCE" in first["issues"]
    revision = first["revision"]
    assert revision["version"] == 2 and "capex_intensity" in revision["added_metrics"]
    assert "balance_sheet_annual" in revision["added_datasets"]  # receivables need a dataset not in the initial plan
    assert context["metadata"]["route"].count("data_collection") == 2
    fetched = [call["slug"] for call in toolset.invocations]
    assert fetched.count("YFINANCE_BALANCE_SHEET") == 1 and fetched.count("YFINANCE_FINANCIALS") == 1  # no re-fetching
    registry = EvidenceRegistry.from_dict(state["registry"])
    assert registry.find(metric="capex_intensity") and registry.find(metric="receivables_days")
    assert state["plan"]["revisions"][0]["triggered_by"]


def test_missing_dataset_triggers_collect_more_data(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    request = make_request("TEST's free cash flow is improving.")
    context = orchestrator.run_stage("plan_generation", request, {"metadata": {"workflow_id": "wf-collect"}}).context
    # Simulate an incomplete initial plan: the cash-flow dataset was left out.
    context["research"]["plan"]["datasets"].remove("cash_flow_annual")
    stage = "data_collection"
    while stage != "report_generation":
        context = orchestrator.run_stage(stage, request, context).context
        stage = orchestrator.next_stage(stage, request, context)
    decisions = [it["decision"] for it in context["research"]["iterations"]]
    assert decisions[0] == "COLLECT_MORE_DATA"
    critique = context["research"]["critiques"][0]
    assert any(i["type"] == "MISSING_METRIC" for i in critique["issues"])
    assert context["research"]["verdict"]["verdict"] == "SUPPORTED"


def test_inconsistent_calculation_triggers_reanalysis(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    request = make_request("TEST's operating margins are improving.")
    context: Dict[str, Any] = {"metadata": {"workflow_id": "wf-reanalyze"}}
    for stage in ("plan_generation", "data_collection", "hybrid_analysis"):
        context = orchestrator.run_stage(stage, request, context).context
    registry = EvidenceRegistry.from_dict(context["research"]["registry"])
    margin = registry.series("TEST", "operating_margin")[-1]
    registry.records[margin.evidence_id] = margin.model_copy(update={"value": 0.99})  # corrupt a derived value
    context["research"]["registry"] = registry.to_dict()
    context["research"]["last_consistency"] = [{"evidence_id": margin.evidence_id, "metric": "operating_margin",
                                                "stored": 0.99, "recomputed": margin.value}]
    context = orchestrator.run_stage("critique", request, context).context
    assert context["research"]["decision"]["status"] == "REANALYZE"
    assert orchestrator.next_stage("critique", request, context) == "hybrid_analysis"
    context = orchestrator.run_stage("hybrid_analysis", request, context).context
    restored = EvidenceRegistry.from_dict(context["research"]["registry"])
    assert abs(restored.series("TEST", "operating_margin")[-1].value - 0.16) < 1e-9
    assert context["research"]["last_consistency"] == []
    context = orchestrator.run_stage("critique", request, context).context
    assert context["research"]["decision"]["status"] == "CONTINUE"
    assert context["research"]["reanalysis_count"] == 1


def test_iteration_limit_produces_honest_partial_result(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, research_max_iterations=1)
    context = _run(orchestrator, "TEST's operating margins are improving.")
    state = context["research"]
    assert [it["decision"] for it in state["iterations"]] == ["CONTINUE"]
    assert "Iteration limit (1) reached" in state["iterations"][0]["rationale"]
    assert state["iteration_limit_hit"] is True
    assert any("iteration limit" in item for item in state["uncertainty"]["analytical_limitations"])


def test_unmeasurable_hypothesis_ends_in_insufficient_evidence(tmp_path: Path) -> None:
    context = _run(_orchestrator(tmp_path), "TEST's streaming segment has become its main profit driver.")
    state = context["research"]
    assert state["decision"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert state["verdict"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert context["report"]["pdf_path"].endswith("validation_report.pdf")


def test_llm_critic_can_escalate_to_human_review(tmp_path: Path) -> None:
    llm = ScriptedLLM(critique={"issues": [{"severity": "high", "description": "Revenue recognition policy changed in FY2024.",
                                            "metrics": ["total_revenue"]}]})
    orchestrator = _orchestrator(tmp_path, llm=llm, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = _run(orchestrator, "TEST's operating margins are improving.")
    state = context["research"]
    assert state["decision"]["status"] == "HUMAN_REVIEW"
    assert orchestrator.human_review_requested(context)
    assert orchestrator.next_stage("report_generation", make_request("x"), context) == "human_review"


def test_llm_critic_raising_unassessed_metric_does_not_escalate(tmp_path: Path) -> None:
    llm = ScriptedLLM(critique={"issues": [{"severity": "high", "description": "Capex intensity was never assessed.",
                                            "metrics": ["capex_intensity"]}]})
    orchestrator = _orchestrator(tmp_path, llm=llm, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = _run(orchestrator, "TEST's operating margins are improving.")
    state = context["research"]
    assert state["decision"]["status"] != "HUMAN_REVIEW"
    assert not orchestrator.human_review_requested(context)
    llm_issues = [i for c in state["critiques"] for i in c["issues"] if i["source"] == "llm"]
    assert llm_issues and all(i["severity"] == "medium" for i in llm_issues)
    assert "lowered from high to medium" in state["critiques"][-1]["llm_notes"]


@pytest.mark.asyncio
async def test_workflow_client_pauses_when_agent_requests_review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = ScriptedLLM(critique={"issues": [{"severity": "high", "description": "Accounting change.", "metrics": ["total_revenue"]}]})

    def factory() -> LangGraphValidationOrchestrator:
        return _orchestrator(tmp_path, llm=llm, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))

    monkeypatch.setattr("hypothesis_agent.workflows.hypothesis_workflow.LangGraphValidationOrchestrator", factory)
    client = HypothesisWorkflowClient(namespace="t", task_queue="q", workflow="w", address="a")
    submission = await client.submit(uuid4(), make_request("TEST's operating margins are improving."))
    for _ in range(200):
        execution = await client.describe(submission.workflow_id, submission.workflow_run_id)
        if execution.status in {"AWAITING_REVIEW", "COMPLETED", "FAILED"}:
            break
        await asyncio.sleep(0.05)
    assert execution.status == "AWAITING_REVIEW"
    review = next(m for m in execution.milestones if m.name == "human_review")
    assert "Agent escalation" in review.detail
    names = [m.name for m in execution.milestones]
    assert "critique" in names and "claim_verification" in names
    summary = await client.fetch_summary(submission.workflow_id, submission.workflow_run_id)
    assert summary.verdict == "SUPPORTED" and summary.human_review_reason
    final = await client.resume(submission.workflow_id, submission.workflow_run_id, "approved")
    assert final.milestones[-1].name == "delivery"
    await client.close()


def test_failed_claims_send_narrative_back_for_correction(tmp_path: Path) -> None:
    wrong = {"narrative": "Revenue was $150 billion in FY2025. The hypothesis is supported."}
    right = {"narrative": "Revenue was $133.1 billion in FY2025. The hypothesis is supported."}
    llm = ScriptedLLM(narratives=[wrong, right])
    orchestrator = _orchestrator(tmp_path, llm=llm, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = _run(orchestrator, "TEST's operating margins are improving.")
    state = context["research"]
    assert context["metadata"]["route"].count("detailed_analysis") == 2
    assert state["narratives"][0]["feedback"] and "CONTRADICTED" in state["narratives"][0]["feedback"][0]
    assert llm.briefings[1]["corrections"] == state["narratives"][0]["feedback"]
    assert state["claims"]["counts"] == {"VERIFIED": 2}


def test_narrative_retries_are_bounded(tmp_path: Path) -> None:
    llm = ScriptedLLM(narratives=[{"narrative": "Revenue was $150 billion in FY2025."}])
    orchestrator = _orchestrator(tmp_path, llm=llm, research_max_narrative_attempts=2,
                                 toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = _run(orchestrator, "TEST's operating margins are improving.")
    assert context["metadata"]["route"].count("detailed_analysis") == 2
    assert context["research"]["claims"]["counts"].get("CONTRADICTED") == 1  # reported, not hidden


def test_llm_tool_calls_are_validated_and_logged(tmp_path: Path) -> None:
    llm = ScriptedLLM(tool_actions=[
        {"tool": "compute_metric", "arguments": {"metric": "net_margin", "ticker": "TEST"}, "rationale": "check net margin"},
        {"tool": "delete_files", "arguments": {"path": "/"}, "rationale": "malicious"},
        {"tool": "fetch_dataset", "arguments": {"dataset": "company_info", "ticker": "MSFT"}},
        {"tool": "finish"},
    ])
    orchestrator = _orchestrator(tmp_path, llm=llm, toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = _run(orchestrator, "TEST's operating margins are improving.")
    llm_calls = [c for c in context["research"]["tool_calls"] if c["origin"] == "llm"]
    assert [c["status"] for c in llm_calls] == ["ok", "rejected", "rejected"]
    assert llm_calls[0]["rationale"] == "check net margin"
    assert EvidenceRegistry.from_dict(context["research"]["registry"]).find(metric="net_margin")

"""The research pipeline as a LangGraph StateGraph: topology, routing, stage budget, review interrupts."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List
from uuid import uuid4

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.models.hypothesis import HypothesisRequest, MilestoneStatus, WorkflowMilestone
from hypothesis_agent.orchestration.langgraph_pipeline import (
    BATCH_REVIEW_DECISION,
    MAX_STAGE_EXECUTIONS,
    LangGraphValidationOrchestrator,
    StageExecutionResult,
)
from hypothesis_agent.research.agent import ROUTES, STAGES, next_stage
from hypothesis_agent.research.offline_llm import DeterministicResearchLLM
from hypothesis_agent.storage.artifact_store import ArtifactStore
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient, WorkflowExecutionDetails
from tests.research_helpers import ScriptedLLM, SyntheticToolSet, company_payloads, make_request

ESCALATING_CRITIC = {"issues": [{"severity": "high", "description": "Accounting change.", "metrics": ["total_revenue"]}]}


def _orchestrator(root: Path, *, llm=None, toolset=None) -> LangGraphValidationOrchestrator:
    settings = AppSettings(artifact_store_path=str(root), llm_provider="offline")
    return LangGraphValidationOrchestrator(settings=settings, llm=llm or DeterministicResearchLLM(),
                                           artifact_store=ArtifactStore.from_path(str(root)),
                                           toolset=toolset or SyntheticToolSet(company_payloads(healthy_cash=True)))


def _looping(orchestrator: LangGraphValidationOrchestrator, calls: List[str]) -> LangGraphValidationOrchestrator:
    """Replace run_stage with a stub whose critique always asks for re-analysis: an endless loop."""

    def run_stage(stage: str, request: HypothesisRequest, context: Dict[str, Any]) -> StageExecutionResult:
        calls.append(stage)
        context = copy.deepcopy(context)
        context.setdefault("research", {})["decision"] = {"status": "REANALYZE"}
        return StageExecutionResult(context=context, milestone=WorkflowMilestone(name=stage, status=MilestoneStatus.COMPLETED))

    orchestrator.run_stage = run_stage  # type: ignore[assignment]
    return orchestrator


async def _wait_for(client: HypothesisWorkflowClient, workflow_id: str, run_id: str, statuses: set[str]) -> WorkflowExecutionDetails:
    for _ in range(400):
        execution = await client.describe(workflow_id, run_id)
        if execution.status in statuses:
            return execution
        await asyncio.sleep(0.025)
    raise AssertionError(f"Workflow never reached {statuses}; last status {execution.status}")


# ----------------------------------------------------------------------
# Topology and routing
# ----------------------------------------------------------------------
def test_graph_topology_matches_routing_table(tmp_path: Path) -> None:
    graph = _orchestrator(tmp_path).build_graph()
    drawn = graph.get_graph()
    assert set(drawn.nodes) == set(STAGES) | {"__start__", "__end__"}
    edges = {(edge.source, edge.target, edge.conditional) for edge in drawn.edges}
    assert edges == {
        ("__start__", "plan_generation", False),
        ("plan_generation", "data_collection", False),
        ("data_collection", "hybrid_analysis", False),
        ("hybrid_analysis", "critique", False),
        ("critique", "data_collection", True),
        ("critique", "hybrid_analysis", True),
        ("critique", "detailed_analysis", True),
        ("detailed_analysis", "claim_verification", False),
        ("claim_verification", "detailed_analysis", True),
        ("claim_verification", "report_generation", True),
        ("report_generation", "human_review", True),
        ("report_generation", "delivery", True),
        ("human_review", "delivery", False),
        ("delivery", "__end__", False),
    }
    # The conditional edges call the research routers themselves (one source of routing truth).
    branches = {source: set(names) for source, names in graph.builder.branches.items()}
    assert branches == {source: {edge.router.__name__} for source, edge in ROUTES.items() if edge.router is not None}


def test_stop_after_ends_the_graph_after_that_stage(tmp_path: Path) -> None:
    drawn = _orchestrator(tmp_path).build_graph(stop_after="report_generation").get_graph()
    out_of_report = {(edge.target, edge.conditional) for edge in drawn.edges if edge.source == "report_generation"}
    assert out_of_report == {("__end__", False)}


@pytest.mark.parametrize(
    ("stage", "research", "human", "expected"),
    [
        ("plan_generation", {}, False, "data_collection"),
        ("data_collection", {}, False, "hybrid_analysis"),
        ("hybrid_analysis", {}, False, "critique"),
        ("critique", {"decision": {"status": "COLLECT_MORE_DATA"}}, False, "data_collection"),
        ("critique", {"decision": {"status": "REVISE_PLAN"}}, False, "data_collection"),
        ("critique", {"decision": {"status": "REANALYZE"}}, False, "hybrid_analysis"),
        ("critique", {"decision": {"status": "CONTINUE"}}, False, "detailed_analysis"),
        ("critique", {"decision": {"status": "HUMAN_REVIEW"}}, False, "detailed_analysis"),
        ("critique", {}, False, "detailed_analysis"),
        ("detailed_analysis", {}, False, "claim_verification"),
        ("claim_verification", {"narrative_retry": True}, False, "detailed_analysis"),
        ("claim_verification", {"narrative_retry": False}, False, "report_generation"),
        ("report_generation", {}, False, "delivery"),
        ("report_generation", {}, True, "human_review"),
        ("report_generation", {"human_review": {"requested": True, "reason": "x"}}, False, "human_review"),
        ("human_review", {}, False, "delivery"),
        ("delivery", {}, False, None),
        ("not_a_stage", {}, False, None),
    ],
)
def test_next_stage_and_graph_edges_share_the_routers(tmp_path: Path, stage: str, research: Dict[str, Any], human: bool,
                                                      expected: str | None) -> None:
    context = {"research": research}
    assert next_stage(stage, context, human_review_required=human) == expected
    if stage in ROUTES:
        assert ROUTES[stage].resolve(context, human) == expected
        assert expected in ROUTES[stage].targets
    orchestrator = _orchestrator(tmp_path)
    request = make_request("x", human=human)
    assert orchestrator.next_stage(stage, request, context) == expected


def _router_loop(orchestrator: LangGraphValidationOrchestrator, request: HypothesisRequest, stop_after: str) -> Dict[str, Any]:
    """Drive the stages one by one with the router, the way the pipeline ran before the graph."""

    context: Dict[str, Any] = {"metadata": {"workflow_id": "wf-route"}}
    stage: str | None = "plan_generation"
    while stage is not None and stage != "human_review":
        context = orchestrator.run_stage(stage, request, context).context
        if stage == stop_after:
            break
        stage = orchestrator.next_stage(stage, request, context)
    return context


@pytest.mark.parametrize(
    ("scenario", "llm", "toolset", "stop_after"),
    [
        ("revise_plan_loop", DeterministicResearchLLM, SyntheticToolSet, "report_generation"),
        ("narrative_retry", lambda: ScriptedLLM(narratives=[
            {"narrative": "Revenue was $150 billion in FY2025. The hypothesis is supported."},
            {"narrative": "Revenue was $133.1 billion in FY2025. The hypothesis is supported."},
        ]), lambda: SyntheticToolSet(company_payloads(healthy_cash=True)), "report_generation"),
        ("escalation_in_batch", lambda: ScriptedLLM(critique=ESCALATING_CRITIC),
         lambda: SyntheticToolSet(company_payloads(healthy_cash=True)), "delivery"),
        ("delivery_without_review", DeterministicResearchLLM, lambda: SyntheticToolSet(company_payloads(healthy_cash=True)),
         "delivery"),
    ],
)
def test_run_research_route_matches_router(tmp_path: Path, scenario: str, llm: Callable[[], Any],
                                           toolset: Callable[[], Any], stop_after: str) -> None:
    request = make_request("TEST's operating margins are improving.")
    expected = _router_loop(_orchestrator(tmp_path / "router", llm=llm(), toolset=toolset()), request, stop_after)
    context = _orchestrator(tmp_path / "graph", llm=llm(), toolset=toolset()).run_research(
        request, {"metadata": {"workflow_id": "wf-route"}}, stop_after=stop_after)

    assert context["metadata"]["route"] == expected["metadata"]["route"]
    state, reference = context["research"], expected["research"]
    assert [it["decision"] for it in state["iterations"]] == [it["decision"] for it in reference["iterations"]]
    assert state["verdict"] == reference["verdict"]
    assert state["claims"]["counts"] == reference["claims"]["counts"]
    if scenario == "revise_plan_loop":
        assert context["metadata"]["route"].count("data_collection") == 2
    if scenario == "narrative_retry":
        assert context["metadata"]["route"].count("detailed_analysis") == 2
    if scenario == "escalation_in_batch":
        # Batch runs have no reviewer: the route stops at the review interrupt and records it as pending.
        assert context["metadata"]["route"][-1] == "report_generation"
        assert context["metadata"]["human_review"] == {"required": True, "decision": BATCH_REVIEW_DECISION}
    else:
        assert "human_review" not in context["metadata"]
    if scenario == "delivery_without_review":
        assert context["metadata"]["route"][-1] == "delivery"


# ----------------------------------------------------------------------
# Stage budget (recursion limit)
# ----------------------------------------------------------------------
def test_run_research_aborts_when_the_stage_budget_is_exhausted(tmp_path: Path) -> None:
    calls: List[str] = []
    orchestrator = _looping(_orchestrator(tmp_path), calls)
    with pytest.raises(RuntimeError, match=f"exceeded {MAX_STAGE_EXECUTIONS} stage executions"):
        orchestrator.run_research(make_request("x"), {"metadata": {"workflow_id": "wf-loop"}})
    assert len(calls) == MAX_STAGE_EXECUTIONS
    assert calls[:4] == ["plan_generation", "data_collection", "hybrid_analysis", "critique"]
    assert set(calls[4:]) == {"hybrid_analysis", "critique"}


@pytest.mark.asyncio
async def test_workflow_client_blocks_the_next_stage_when_the_budget_is_exhausted(monkeypatch: pytest.MonkeyPatch,
                                                                                  tmp_path: Path) -> None:
    calls: List[str] = []
    monkeypatch.setattr("hypothesis_agent.workflows.hypothesis_workflow.LangGraphValidationOrchestrator",
                        lambda: _looping(_orchestrator(tmp_path), calls))
    client = HypothesisWorkflowClient(namespace="t", task_queue="q", workflow="w", address="a")
    submission = await client.submit(uuid4(), make_request("x"))
    execution = await _wait_for(client, submission.workflow_id, submission.workflow_run_id, {"FAILED", "COMPLETED"})

    assert execution.status == "FAILED"
    assert len(calls) == MAX_STAGE_EXECUTIONS and calls[-1] == "critique"
    blocked = [m for m in execution.milestones if m.status == MilestoneStatus.BLOCKED]
    assert [(m.name, m.detail) for m in blocked] == [
        ("hybrid_analysis", f"Stage budget of {MAX_STAGE_EXECUTIONS} executions exhausted.")]
    assert client._checkpointer.get_tuple({"configurable": {"thread_id": client._thread_id(
        (submission.workflow_id, submission.workflow_run_id))}}) is None
    await client.close()


# ----------------------------------------------------------------------
# Human review: interrupt, checkpoint, resume
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_review_pauses_on_interrupt_and_resumes_from_the_checkpoint(monkeypatch: pytest.MonkeyPatch,
                                                                          tmp_path: Path) -> None:
    stages_by_instance: List[List[str]] = []

    def factory() -> LangGraphValidationOrchestrator:
        orchestrator = _orchestrator(tmp_path)
        ran: List[str] = []
        stages_by_instance.append(ran)
        original = orchestrator.run_stage

        def run_stage(stage: str, request: HypothesisRequest, context: Dict[str, Any]) -> StageExecutionResult:
            ran.append(stage)
            return original(stage, request, context)

        orchestrator.run_stage = run_stage  # type: ignore[assignment]
        return orchestrator

    monkeypatch.setattr("hypothesis_agent.workflows.hypothesis_workflow.LangGraphValidationOrchestrator", factory)
    client = HypothesisWorkflowClient(namespace="t", task_queue="q", workflow="w", address="a")
    submission = await client.submit(uuid4(), make_request("TEST's operating margins are improving.", human=True))
    key = (submission.workflow_id, submission.workflow_run_id)
    paused = await _wait_for(client, *key, {"AWAITING_REVIEW", "FAILED", "COMPLETED"})
    assert paused.status == "AWAITING_REVIEW"
    assert stages_by_instance[0][-1] == "report_generation" and "delivery" not in stages_by_instance[0]

    # The paused run lives in the client's checkpointer, interrupted in front of human_review.
    config = {"configurable": {"thread_id": client._thread_id(key)}}
    snapshot = _orchestrator(tmp_path).build_graph(checkpointer=client._checkpointer).get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.interrupts[0].value["required_by_request"] is True
    assert snapshot.values["context"]["report"]["pdf_path"].endswith("validation_report.pdf")
    json.dumps(snapshot.values)  # the graph state is plain JSON

    # Resuming must not depend on the client's in-memory copy of the context: the checkpoint resumes it.
    client._local_runs[key].context = {}
    final = await client.resume(*key, "approved")

    assert len(stages_by_instance) == 2 and stages_by_instance[1] == ["delivery"]  # nothing upstream re-ran
    review = next(m for m in final.milestones if m.name == "human_review")
    assert (review.status, review.detail) == (MilestoneStatus.COMPLETED, "Human decision: approved.")
    assert final.milestones[-1].name == "delivery" and final.milestones[-1].status == MilestoneStatus.COMPLETED
    assert final.current_stage == "delivery" and final.verdict == "SUPPORTED"
    context = client._local_runs[key].context
    assert context["metadata"]["human_review"] == {"required": True, "decision": "approved"}
    assert context["metadata"]["route"][-1] == "delivery"
    assert context["metadata"]["stages"]["delivery"]["status"] == "completed"
    assert (await client.describe(*key)).status == "COMPLETED"
    assert client._checkpointer.get_tuple(config) is None  # finished runs drop their checkpoints
    await client.close()


@pytest.mark.asyncio
async def test_cancel_during_resume_stays_cancelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import time
    import hypothesis_agent.workflows.hypothesis_workflow as wf_module

    def factory() -> LangGraphValidationOrchestrator:
        orchestrator = _orchestrator(tmp_path)
        original = orchestrator.run_stage

        def slow_delivery(stage, request, context):
            if stage == "delivery":
                time.sleep(0.3)
            return original(stage, request, context)

        orchestrator.run_stage = slow_delivery  # type: ignore[assignment]
        return orchestrator

    monkeypatch.setattr(wf_module, "LangGraphValidationOrchestrator", factory)
    client = HypothesisWorkflowClient(namespace="t", task_queue="q", workflow="w", address="a")
    request = make_request("TEST's operating margins are improving.").model_copy(update={"requires_human_review": True})
    submission = await client.submit(uuid4(), request)
    for _ in range(200):
        if (await client.describe(submission.workflow_id, submission.workflow_run_id)).status == "AWAITING_REVIEW":
            break
        await asyncio.sleep(0.05)
    resume = asyncio.create_task(client.resume(submission.workflow_id, submission.workflow_run_id, "approved"))
    await asyncio.sleep(0.1)
    await client.cancel(submission.workflow_id, submission.workflow_run_id)
    summary = await resume
    assert summary.current_stage == "cancelled"
    assert (await client.describe(submission.workflow_id, submission.workflow_run_id)).status == "CANCELLED"
    await client.close()

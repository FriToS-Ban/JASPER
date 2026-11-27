"""Orchestration for the JASPER research pipeline as a LangGraph ``StateGraph``.

:meth:`LangGraphValidationOrchestrator.build_graph` compiles the routing table
(:data:`hypothesis_agent.research.agent.ROUTES`) into a graph with one node per stage. Each node
wraps :meth:`LangGraphValidationOrchestrator.run_stage`; fixed edges link the linear steps and
conditional edges call the research routers, which lets the critique branch back to data
collection, re-analysis, or forward to narrative, lets claim verification send a failed
narrative back for correction, and sends the report to human review when it is required.

The graph state (:class:`ResearchGraphState`) is plain JSON: the request as
``model_dump(mode="json")``, the stage context, and the last stage's milestone/evidence/summary.
``human_review`` pauses with LangGraph's ``interrupt()``; with a checkpointer the run resumes
from the checkpoint via ``Command(resume=decision)``. Every graph invocation is bounded by a
``recursion_limit`` of :data:`MAX_STAGE_EXECUTIONS` node executions on top of the research
iteration limits. Stage names are unchanged from the original linear pipeline so existing
callers keep working.
"""
from __future__ import annotations

import copy
import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

import matplotlib
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, interrupt

# Use a headless backend for deterministic chart generation in tests and workers.
matplotlib.use("Agg", force=True)

from hypothesis_agent.config import AppSettings, get_settings  # noqa: E402
from hypothesis_agent.llm import BaseLLM, LLMError, NvidiaLLM  # noqa: E402
from hypothesis_agent.models.hypothesis import (  # noqa: E402
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.fixture_tools import FixtureToolSet  # noqa: E402
from hypothesis_agent.orchestration.python_sandbox import PythonSandbox  # noqa: E402
from hypothesis_agent.orchestration.tool_catalog import ToolDefinition, load_tool_catalog  # noqa: E402
from hypothesis_agent.orchestration.yfinance_tools import ToolSet, YFinanceToolSet  # noqa: E402
from hypothesis_agent.reporting.pdf_report import render_research_report  # noqa: E402
from hypothesis_agent.research.agent import (  # noqa: E402
    ROUTES,
    STAGES,
    ResearchAgent,
    ResearchLimits,
    StageEdge,
    next_stage as route_next_stage,
    research_state,
)
from hypothesis_agent.research.metrics import DATASETS  # noqa: E402
from hypothesis_agent.research.offline_llm import DeterministicResearchLLM  # noqa: E402
from hypothesis_agent.storage.artifact_store import ArtifactStore  # noqa: E402

StageContext = Dict[str, Any]

RESULT_PREFIX = "RESULT::"
MAX_REPL_ATTEMPTS = 5
STDOUT_CHARACTER_LIMIT = 1600
# Defence in depth on top of the research iteration limits: one graph invocation may execute at
# most this many stage nodes. Each stage is one LangGraph superstep, so this is the graph's
# recursion_limit (a limit of N allows exactly N node executions).
MAX_STAGE_EXECUTIONS = 40
BATCH_REVIEW_DECISION = "pending (not executed in batch mode)"
VERDICT_TEXT = {
    "SUPPORTED": "Supported",
    "PARTIALLY_SUPPORTED": "Partially supported",
    "NOT_SUPPORTED": "Not supported",
    "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
}


def get_data_format(artifact_map: Dict[str, str]) -> Dict[str, str]:
    """Produce compact JSON schema hints for each artifact path (legacy code-generation path)."""
    summaries: Dict[str, str] = {}
    for slug, path_str in artifact_map.items():
        try:
            payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        except Exception:
            continue
        summary_payload = _summarize_payload(payload)
        try:
            summary_text = json.dumps(summary_payload, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            summary_text = str(summary_payload)
        if len(summary_text) > 400:
            summary_text = summary_text[:397] + "..."
        summaries[slug] = summary_text
    return summaries


def _summarize_payload(payload: Any, depth: int = 0) -> Any:
    if isinstance(payload, dict):
        keys = list(payload.keys())
        summary: Dict[str, Any] = {"keys": keys[:16]}
        first_key = keys[0] if keys else None
        if first_key is not None:
            summary["sample_key"] = first_key
            summary["sample"] = _summarize_value(payload[first_key], depth + 1)
        return summary
    if isinstance(payload, list):
        summary_list: Dict[str, Any] = {"type": "array", "length": len(payload)}
        if payload and isinstance(payload[0], dict):
            summary_list["item_keys"] = list(payload[0].keys())[:16]
            summary_list["sample"] = _summarize_value(payload[0], depth + 1)
        elif payload:
            summary_list["sample"] = _summarize_value(payload[0], depth + 1)
        return summary_list
    return _summarize_value(payload, depth)


def _summarize_value(value: Any, depth: int) -> Any:
    if depth >= 2:
        if isinstance(value, dict):
            return {"keys": list(value.keys())[:6]}
        if isinstance(value, list):
            return {"type": "array", "length": len(value)}
        return _trim_scalar(value)
    if isinstance(value, dict):
        trimmed: Dict[str, Any] = {}
        for index, (key, nested) in enumerate(value.items()):
            if index >= 4:
                break
            trimmed[key] = _summarize_value(nested, depth + 1)
        return trimmed
    if isinstance(value, list):
        if not value:
            return []
        return [_summarize_value(value[0], depth + 1)]
    return _trim_scalar(value)


def _trim_scalar(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 60:
        return value[:57] + "..."
    return value


@dataclass(slots=True)
class StageExecutionResult:
    """Container describing the outcome of a single stage."""

    context: StageContext
    milestone: WorkflowMilestone
    evidence: List[EvidenceReference] = field(default_factory=list)
    summary: Optional[ValidationSummary] = None

    def outcome(self) -> Dict[str, Any]:
        """JSON form of everything except the context, as carried in the graph state."""

        return {
            "milestone": self.milestone.model_dump(mode="json"),
            "evidence": [reference.model_dump(mode="json") for reference in self.evidence],
            "summary": self.summary.model_dump(mode="json") if self.summary else None,
        }

    @classmethod
    def from_graph_update(cls, update: Dict[str, Any]) -> "StageExecutionResult":
        """Rebuild a stage result from the ``context`` and ``outcome`` a graph node wrote."""

        outcome = update["outcome"]
        summary = outcome.get("summary")
        return cls(
            context=update["context"],
            milestone=WorkflowMilestone.model_validate(outcome["milestone"]),
            evidence=[EvidenceReference.model_validate(reference) for reference in outcome.get("evidence", [])],
            summary=ValidationSummary.model_validate(summary) if summary else None,
        )


class ResearchGraphState(TypedDict, total=False):
    """State of the research graph. Every value is plain JSON so checkpoints can be serialised."""

    request: Dict[str, Any]
    """The ``HypothesisRequest`` as ``model_dump(mode="json")``."""
    context: StageContext
    """The stage context each node reads and replaces (research state, evidence, report, ...)."""
    outcome: Dict[str, Any]
    """Milestone, evidence and summary of the node that ran last (see ``StageExecutionResult``)."""


class LangGraphValidationOrchestrator:
    """Run the evidence-grounded research pipeline: single stages, or the compiled research graph."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        llm: BaseLLM | None = None,
        artifact_store: ArtifactStore | None = None,
        toolset: ToolSet | None = None,
        tool_catalog: Dict[str, ToolDefinition] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if llm is not None:
            self.llm = llm
        elif self.settings.llm_provider == "offline":
            self.llm = DeterministicResearchLLM()
        else:
            if not self.settings.nvidia_api_key:
                raise LLMError("NVIDIA API key must be configured for validation pipeline (or set LLM_PROVIDER=offline)")
            self.llm = NvidiaLLM(api_key=self.settings.nvidia_api_key, model=self.settings.nvidia_model)
        self.artifact_store = artifact_store or ArtifactStore.from_path(self.settings.artifact_store_path)
        self.tool_catalog = tool_catalog or load_tool_catalog()
        if toolset is not None:
            self.toolset = toolset
        elif self.settings.research_fixture_dir:
            self.toolset = FixtureToolSet(self.settings.research_fixture_dir)
        else:
            self.toolset = YFinanceToolSet()
        self.limits = ResearchLimits(
            max_iterations=self.settings.research_max_iterations,
            max_tool_calls=self.settings.research_max_tool_calls,
            max_llm_tool_steps=self.settings.research_max_llm_tool_steps,
            max_narrative_attempts=self.settings.research_max_narrative_attempts,
        )
        self.agent = ResearchAgent(llm=self.llm, toolset=self.toolset, artifact_store=self.artifact_store, limits=self.limits)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_stage(self, stage: str, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        stage_key = stage.lower()
        context = self._clone_context(context)
        self._append_workflow_log(context, f"Stage {stage_key} started.")
        handlers = {
            "plan_generation": self._run_plan_generation,
            "data_collection": self._run_data_collection,
            "hybrid_analysis": self._run_hybrid_analysis,
            "critique": self._run_critique,
            "detailed_analysis": self._run_detailed_analysis,
            "claim_verification": self._run_claim_verification,
            "report_generation": self._run_report_generation,
            "delivery": self._run_delivery,
        }
        handler = handlers.get(stage_key)
        if handler is None:
            raise ValueError(f"Unknown validation stage '{stage_key}'")
        try:
            result = handler(request, context)
        except Exception as exc:
            failure_message = str(exc).replace("\n", " ")
            self._append_workflow_log(context, f"Stage {stage_key} failed: {failure_message}")
            raise
        trace = result.context.setdefault("metadata", {}).setdefault("route", [])
        trace.append(stage_key)
        self._append_workflow_log(result.context, f"Stage {stage_key} completed successfully.")
        return result

    def next_stage(self, stage: str, request: HypothesisRequest, context: StageContext) -> Optional[str]:
        """Return the next stage chosen by the research router (None when finished)."""

        return route_next_stage(stage, context, human_review_required=request.requires_human_review)

    @staticmethod
    def human_review_requested(context: StageContext) -> bool:
        return bool(((context.get("research") or {}).get("human_review") or {}).get("requested"))

    # ------------------------------------------------------------------
    # LangGraph
    # ------------------------------------------------------------------
    def build_graph(
        self,
        *,
        checkpointer: Checkpointer = None,
        stop_after: Optional[str] = None,
        record_stage_status: bool = False,
    ) -> CompiledStateGraph:
        """Compile the research pipeline into a LangGraph ``StateGraph``.

        One node per stage in :data:`STAGES`; edges come from :data:`ROUTES` (fixed edges for the
        linear steps, conditional edges for the critique decision, the claim-verification retry and
        the review/delivery split). ``stop_after`` ends the graph after that stage instead of
        following its route. ``record_stage_status`` also records each stage's milestone status in
        ``context["metadata"]["stages"]`` (the workflow client's progress record). The
        ``human_review`` node pauses with ``interrupt()`` and needs a ``checkpointer`` to resume.
        """

        builder = StateGraph(ResearchGraphState)
        for stage in STAGES:
            if stage == "human_review":
                builder.add_node(stage, self._human_review_node)
            else:
                builder.add_node(stage, self._stage_node(stage, record_stage_status=record_stage_status))
        builder.add_edge(START, STAGES[0])
        for stage in STAGES:
            edge = ROUTES.get(stage)
            if stage == stop_after or edge is None:
                builder.add_edge(stage, END)
            elif edge.router is None:
                builder.add_edge(stage, edge.targets[0])
            else:
                builder.add_conditional_edges(stage, self._edge_router(edge), list(edge.targets))
        return builder.compile(checkpointer=checkpointer, name="jasper_research")

    @staticmethod
    def graph_input(request: HypothesisRequest, context: StageContext) -> ResearchGraphState:
        """Initial graph state for a run."""

        return {"request": request.model_dump(mode="json"), "context": context}

    @staticmethod
    def graph_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Run config: the stage budget as recursion limit, plus the checkpoint thread if any."""

        config: Dict[str, Any] = {"recursion_limit": MAX_STAGE_EXECUTIONS}
        if thread_id is not None:
            config["configurable"] = {"thread_id": thread_id}
        return config

    def _stage_node(self, stage: str, *, record_stage_status: bool) -> Callable[[ResearchGraphState], Dict[str, Any]]:
        def node(state: ResearchGraphState) -> Dict[str, Any]:
            request = HypothesisRequest.model_validate(state["request"])
            # Look run_stage up at call time so a replaced run_stage on the instance is honoured.
            result = self.run_stage(stage, request, state["context"])
            if record_stage_status:
                stages = result.context.setdefault("metadata", {}).setdefault("stages", {})
                stages[result.milestone.name] = {"status": result.milestone.status.value, "detail": result.milestone.detail}
            return {"context": result.context, "outcome": result.outcome()}

        node.__name__ = stage
        return node

    def _human_review_node(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Pause for a reviewer; on resume record the decision and continue to delivery."""

        context = state["context"]
        reason = ((context.get("research") or {}).get("human_review") or {}).get("reason")
        decision = interrupt({
            "stage": "human_review",
            "reason": reason,
            "required_by_request": bool(state["request"].get("requires_human_review")),
        })
        context = self._clone_context(context)
        context.setdefault("metadata", {})["human_review"] = {"required": True, "decision": decision}
        milestone = WorkflowMilestone(name="human_review", status=MilestoneStatus.COMPLETED, detail=f"Human decision: {decision}.")
        return {"context": context, "outcome": StageExecutionResult(context=context, milestone=milestone).outcome()}

    @staticmethod
    def _edge_router(edge: StageEdge) -> Callable[[ResearchGraphState], str]:
        """Conditional-edge path function: the same resolution next_stage uses, read from graph state."""

        def route(state: ResearchGraphState) -> str:
            return edge.resolve(state["context"], bool(state["request"].get("requires_human_review")))

        route.__name__ = edge.router.__name__ if edge.router else "route"  # names the branch in the graph
        return route

    def run_research(
        self,
        request: HypothesisRequest,
        context: Optional[StageContext] = None,
        *,
        stop_after: str = "report_generation",
    ) -> StageContext:
        """Run the research graph synchronously (CLI and benchmark entry point).

        Stops after ``stop_after``. Batch runs have no reviewer: when the route reaches
        ``human_review`` the graph's interrupt ends the run and the review is recorded as pending.
        """

        context = context or {"metadata": {"workflow_id": f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"}}
        graph = self.build_graph(stop_after=stop_after)
        try:
            final = graph.invoke(self.graph_input(request, context), self.graph_config())
        except GraphRecursionError as exc:
            raise RuntimeError(f"Pipeline exceeded {MAX_STAGE_EXECUTIONS} stage executions; aborting to prevent a loop.") from exc
        context = final["context"]
        if final.get("__interrupt__"):
            context.setdefault("metadata", {})["human_review"] = {"required": True, "decision": BATCH_REVIEW_DECISION}
        return context

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------
    def _run_plan_generation(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        plan = self.agent.plan(request, context, workflow_id)
        legacy_plan = self._legacy_plan(plan.model_dump(mode="json"))
        plan_location = self.artifact_store.write_json(workflow_id, "validation_plan", legacy_plan).resolve()
        context["plan"] = legacy_plan
        context.setdefault("insights", []).append(
            f"Planning stage derived {len(plan.expectations)} testable expectation(s) and {len(plan.metrics)} metrics."
        )
        metadata = context.setdefault("metadata", {})
        metadata.setdefault("plan", {}).update({"artifact": str(plan_location), "artifact_uri": plan_location.as_uri()})
        self._append_workflow_log(context, f"Plan v{plan.version}: objective '{plan.objective}'; datasets {plan.datasets}.")
        expectations = "; ".join(f"{e.expectation_id} {e.metric or e.concept} {e.direction}" for e in plan.expectations) or "none parsed"
        milestone = WorkflowMilestone(
            name="plan_generation",
            status=MilestoneStatus.COMPLETED,
            detail=f"Plan v{plan.version}: expectations [{expectations}]; datasets: {', '.join(plan.datasets)}.",
        )
        evidence = [EvidenceReference(type="validation_plan", uri=plan_location.as_uri())]
        self._record_evidence(context, evidence[0])
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _legacy_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        tools = []
        for dataset in plan.get("datasets", []):
            slug, extra = DATASETS[dataset]
            arguments: Dict[str, Any] = {"symbol": plan["ticker"], **extra}
            if dataset == "price_history":
                arguments["period"] = plan.get("price_period", "1y")
            definition = self.tool_catalog.get(slug)
            tools.append({"slug": slug, "dataset": dataset, "arguments": arguments,
                          "description": definition.description if definition else slug})
        analysis = [{"name": e["expectation_id"], "description": e["rationale"], "metric": e["metric"], "direction": e["direction"]}
                    for e in plan.get("expectations", [])]
        return {"hypothesis": plan["hypothesis"], "target_symbol": plan["ticker"], "data_fetch_tools": tools,
                "analysis_plan": analysis, "research_plan": plan}

    def _run_data_collection(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        if not context.get("plan"):
            raise RuntimeError("Validation plan missing from context; run plan_generation first.")
        state = research_state(context)
        outcome = self.agent.collect(request, context, workflow_id)
        context["plan"] = self._legacy_plan(state["plan"])
        evidence: List[EvidenceReference] = []
        data_sources: Dict[str, Any] = {}
        for snapshot in (state.get("registry") or {}).get("snapshots", []):
            if snapshot.get("artifact_uri"):
                data_sources[snapshot["tool_slug"] if snapshot["tool_slug"] not in data_sources else snapshot["dataset"]] = {
                    "artifact_path": snapshot["artifact_path"], "artifact_uri": snapshot["artifact_uri"],
                    "arguments": snapshot["parameters"], "snapshot_id": snapshot["snapshot_id"],
                }
                reference = EvidenceReference(type="data_fetch", uri=snapshot["artifact_uri"])
                if reference.model_dump(mode="json") not in context.get("evidence", []):
                    evidence.append(reference)
                    self._record_evidence(context, reference)
        context["data_sources"] = data_sources
        lines = "; ".join(f"{r['dataset']}: {r['status']}" for r in outcome["results"])
        self._append_workflow_log(context, f"Iteration {state['iteration']} data collection: {lines}; +{outcome['new_evidence']} evidence records.")
        milestone = WorkflowMilestone(
            name="data_collection",
            status=MilestoneStatus.COMPLETED,
            detail=f"Iteration {state['iteration']}: {lines} (+{outcome['new_evidence']} evidence records).",
        )
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_hybrid_analysis(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        if not context.get("plan"):
            raise RuntimeError("Validation plan missing from context; run plan_generation first.")
        if not context.get("data_sources") and not (context.get("research") or {}).get("registry", {}).get("snapshots"):
            raise RuntimeError("Data sources missing; run data_collection stage first.")
        state = research_state(context)
        outcome = self.agent.analyze(request, context, workflow_id)
        verdict = outcome["verdict"]
        analysis_payload: Dict[str, Any] = {
            "steps": [
                {"name": a.expectation_id, "metric": a.metric, "status": a.status, "summary": a.summary,
                 "evidence_ids": a.evidence_ids}
                for a in outcome["assessments"]
            ],
            "aggregated": {"verdict": verdict.verdict, "evidence_score": verdict.primary_score},
            "insights": [a.summary for a in outcome["assessments"]][:5],
            "charts": [c["artifact_uri"] for c in state["charts"]],
            "tool_calls": len(state["tool_calls"]),
            "llm_tool_steps": outcome["llm_steps"],
        }
        if self.settings.enable_codegen_analysis:
            analysis_payload["supplementary_unverified"] = self._supplementary_codegen(request, context, workflow_id)
        metrics_uri = self.artifact_store.write_json(workflow_id, "analysis_metrics", analysis_payload).resolve().as_uri()
        evidence = [EvidenceReference(type="analysis_metrics", uri=metrics_uri)]
        self._record_evidence(context, evidence[0])
        context["analysis_results"] = analysis_payload
        context.setdefault("insights", []).extend(analysis_payload["insights"][:3])
        self._append_workflow_log(context, f"Iteration {state['iteration']} analysis: verdict {verdict.verdict}; {len(state['tool_calls'])} tool calls so far.")
        statuses = ", ".join(f"{a.expectation_id}={a.status}" for a in outcome["assessments"]) or "no expectations"
        milestone = WorkflowMilestone(
            name="hybrid_analysis",
            status=MilestoneStatus.COMPLETED,
            detail=f"Iteration {state['iteration']}: {statuses}; provisional verdict {verdict.verdict}.",
        )
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _supplementary_codegen(self, request: HypothesisRequest, context: StageContext, workflow_id: str) -> Dict[str, Any]:
        """Legacy LLM code generation, kept opt-in. Its numbers are not evidence and are labelled as such."""

        data_sources = context.get("data_sources") or {}
        plan = context.get("plan") or {}
        try:
            result, history, log_uri = self._execute_llm_analysis(
                workflow_id=workflow_id, request=request, analysis_plan=plan.get("analysis_plan", []), data_sources=data_sources,
            )
        except Exception as exc:
            research_state(context)["errors"].append({"at": datetime.now(timezone.utc).isoformat(), "stage": "codegen", "error": str(exc)})
            return {"status": "failed", "error": str(exc)}
        return {"status": "ok", "result": result, "attempts": len(history), "log_uri": log_uri,
                "note": "LLM-generated code output; not part of the evidence registry and not used for the verdict."}

    def _run_critique(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        result = self.agent.critique(request, context, workflow_id)
        context["plan"] = self._legacy_plan(research_state(context)["plan"])
        uri = self.artifact_store.write_json(workflow_id, f"critique_iteration_{result.iteration}", result.model_dump(mode="json")).resolve().as_uri()
        evidence = [EvidenceReference(type="critique", uri=uri)]
        self._record_evidence(context, evidence[0])
        self._append_workflow_log(context, f"Critique iteration {result.iteration}: {result.status} - {result.rationale}")
        milestone = WorkflowMilestone(
            name="critique",
            status=MilestoneStatus.COMPLETED,
            detail=f"Iteration {result.iteration}: {result.status} ({len(result.issues)} issue(s)) - {result.rationale}",
        )
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_detailed_analysis(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        analysis_results = context.get("analysis_results")
        if not analysis_results:
            raise RuntimeError("Analysis results missing; run hybrid_analysis stage first.")
        workflow_id = self._ensure_workflow_id(context)
        entry = self.agent.narrate(request, context, workflow_id)
        narrative = entry["narrative"]
        context["detailed_analysis"] = {"narrative": str(narrative.get("narrative", "")), "structured": narrative,
                                        "metrics": analysis_results, "attempt": entry["attempt"]}
        context.setdefault("insights", []).append("Narrative generated from the evidence briefing.")
        self._append_workflow_log(context, f"Narrative attempt {entry['attempt']} generated ({len(str(narrative.get('narrative', '')))} characters).")
        milestone = WorkflowMilestone(
            name="detailed_analysis",
            status=MilestoneStatus.COMPLETED,
            detail=f"Narrative attempt {entry['attempt']} written from {len(analysis_results.get('steps', []))} assessed expectation(s).",
        )
        evidence: List[EvidenceReference] = []
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_claim_verification(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        report = self.agent.verify_claims(request, context, workflow_id)
        state = research_state(context)
        uri = self.artifact_store.write_json(workflow_id, "claim_verification", state["claims"]).resolve().as_uri()
        evidence = [EvidenceReference(type="claim_verification", uri=uri)]
        self._record_evidence(context, evidence[0])
        counts = ", ".join(f"{k}={v}" for k, v in sorted(report.counts.items())) or "no claims"
        retry = " Narrative sent back for correction." if state.get("narrative_retry") else ""
        self._append_workflow_log(context, f"Claim verification: {counts}.{retry}")
        milestone = WorkflowMilestone(name="claim_verification", status=MilestoneStatus.COMPLETED,
                                      detail=f"{len(report.claims)} claim(s): {counts}.{retry}")
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_report_generation(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        if not context.get("plan"):
            raise RuntimeError("Validation plan missing; run plan_generation stage first")
        if not context.get("detailed_analysis"):
            raise RuntimeError("Detailed analysis missing; run detailed_analysis stage first")
        if not context.get("analysis_results"):
            raise RuntimeError("Analysis results missing; run hybrid_analysis stage first")
        state = research_state(context)
        # Never publish unverified text: verify here if the claim_verification stage was skipped
        # (e.g. direct stage invocation) or if it checked an older narrative attempt.
        if not state.get("claims") or state.get("claims_attempt") != len(state.get("narratives", [])):
            self.agent.verify_claims(request, context, workflow_id)
            state = research_state(context)
        state["narrative_retry"] = False
        self.agent.finalize(context, workflow_id)
        state = research_state(context)
        pdf_path, sections = render_research_report(request=request, workflow_id=workflow_id, state=state,
                                                    artifact_store=self.artifact_store)
        latest = state["narratives"][-1]
        context["report"] = {"payload": latest.get("report_payload") or {}, "pdf_path": pdf_path, "sections": sections}
        pdf_evidence = EvidenceReference(type="report_document", uri=pdf_path)
        self._record_evidence(context, pdf_evidence)
        summary = self._build_summary(context)
        self._append_workflow_log(context, f"Report generated at {pdf_path}; verdict {summary.verdict}; evidence score {summary.score}; data completeness {summary.confidence}.")
        milestone = WorkflowMilestone(
            name="report_generation",
            status=MilestoneStatus.COMPLETED,
            detail=f"Evidence-backed report compiled; verdict {summary.verdict}.",
        )
        evidence = [pdf_evidence]
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence, summary=summary)

    def _build_summary(self, context: StageContext) -> ValidationSummary:
        state = research_state(context)
        verdict = state.get("verdict") or {}
        uncertainty = state.get("uncertainty") or {}
        completeness = uncertainty.get("data_completeness") or {}
        balance = uncertainty.get("evidence_balance") or {}
        label = verdict.get("verdict", "INSUFFICIENT_EVIDENCE")
        human = state.get("human_review") or {}
        return ValidationSummary(
            score=float(verdict.get("primary_score") or 0.0),
            conclusion=VERDICT_TEXT.get(label, label),
            confidence=float(completeness.get("coverage_ratio") or 0.0),
            evidence=[EvidenceReference.model_validate(ev) for ev in context.get("evidence", [])],
            current_stage="report_generation",
            milestones=[],
            verdict=label,
            claim_verification=(state.get("claims") or {}).get("counts", {}),
            uncertainty={
                "coverage_ratio": completeness.get("coverage_ratio"),
                "metrics_unavailable": [u["metric"] for u in completeness.get("metrics_unavailable", [])],
                **{key: len(value) for key, value in balance.items()},
                "staleness_days": (uncertainty.get("source_quality") or {}).get("staleness_days"),
            },
            research_iterations=len(state.get("iterations", [])),
            human_review_reason=human.get("reason") if human.get("requested") else None,
        )

    def _run_delivery(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        report = context.get("report") or {}
        pdf_path = report.get("pdf_path")
        payload = report.get("payload") or {}
        if not pdf_path or not payload:
            raise RuntimeError("Report assets missing; run report_generation stage first.")
        context.setdefault("insights", []).append("Report ready for download via the JASPER portal.")
        self._append_workflow_log(context, "Stage delivery completed; report published for download.")
        milestone = WorkflowMilestone(name="delivery", status=MilestoneStatus.COMPLETED, detail="Report available for download.")
        evidence: List[EvidenceReference] = []
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    # ------------------------------------------------------------------
    # Context and logging helpers
    # ------------------------------------------------------------------
    def _ensure_workflow_id(self, context: StageContext) -> str:
        metadata = context.get("metadata") or {}
        workflow_id = metadata.get("workflow_id")
        if not workflow_id:
            raise RuntimeError("Workflow metadata must include a workflow_id")
        return str(workflow_id)

    def _append_workflow_log(self, context: StageContext, message: str) -> str:
        workflow_id = self._ensure_workflow_id(context)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] {message.rstrip()}\n"
        path = self.artifact_store.append_text(workflow_id, "workflow.log", line)
        log_uri = path.resolve().as_uri()
        metadata = context.setdefault("metadata", {})
        log_info = metadata.setdefault("workflow_log", {})
        log_info["artifact"] = log_uri
        context["workflow_log_uri"] = log_uri
        return log_uri

    def _maybe_add_workflow_log_evidence(self, context: StageContext, evidence: List[EvidenceReference]) -> None:
        metadata = context.setdefault("metadata", {})
        log_info = metadata.setdefault("workflow_log", {})
        log_uri = log_info.get("artifact")
        if not log_uri or log_info.get("evidence_recorded"):
            return
        reference = EvidenceReference(type="workflow_log", uri=log_uri)
        evidence.append(reference)
        self._record_evidence(context, reference)
        log_info["evidence_recorded"] = True

    def _record_evidence(self, context: StageContext, evidence: EvidenceReference) -> None:
        context.setdefault("evidence", []).append(evidence.model_dump(mode="json"))
        context.setdefault("artifacts", []).append(evidence.model_dump(mode="json"))

    @staticmethod
    def _clone_context(context: StageContext | None) -> StageContext:
        if not context:
            return cast(StageContext, {})
        return copy.deepcopy(context)

    # ------------------------------------------------------------------
    # Legacy code-generation helpers (opt-in via ENABLE_CODEGEN_ANALYSIS)
    # ------------------------------------------------------------------
    def _execute_llm_analysis(
        self,
        *,
        workflow_id: str,
        request: HypothesisRequest,
        analysis_plan: List[Dict[str, Any]],
        data_sources: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        artifact_map: Dict[str, str] = {}
        for slug, metadata in data_sources.items():
            uri = metadata.get("artifact_path") or metadata.get("uri")
            if not uri:
                raise RuntimeError(f"Artifact path missing for tool '{slug}'")
            path = Path(uri).resolve()
            if not path.exists():
                raise RuntimeError(f"Artifact path for tool '{slug}' does not exist: {path}")
            artifact_map[slug] = str(path)

        history_records: List[Dict[str, Any]] = []
        prompt_history: List[Dict[str, str]] = []
        log_entries: List[str] = []
        log_uri: Optional[str] = None
        data_format = get_data_format(artifact_map)

        def _persist_log() -> str:
            nonlocal log_uri
            log_text = "\n\n".join(log_entries) if log_entries else "No attempts recorded."
            path = self.artifact_store.write_text(workflow_id, "hybrid_analysis_attempts.log", log_text)
            log_uri = path.resolve().as_uri()
            return log_uri

        for attempt in range(1, MAX_REPL_ATTEMPTS + 1):
            response = self.llm.generate_analysis_code(
                request=request, analysis_plan=analysis_plan, data_artifacts=artifact_map,
                data_format=data_format, attempt=attempt, history=prompt_history,
            )
            code = self._extract_python_code(response)
            stdout_text, stderr_text, result_payload, feedback = self._run_analysis_code(
                code=code, artifact_map=artifact_map, workflow_id=workflow_id,
            )
            record_status = "success" if result_payload is not None else "error"
            record: Dict[str, Any] = {
                "attempt": attempt,
                "status": record_status,
                "code": self._tail_text(code, STDOUT_CHARACTER_LIMIT),
                "stdout": self._tail_text(stdout_text, STDOUT_CHARACTER_LIMIT),
                "stderr": self._tail_text(stderr_text, STDOUT_CHARACTER_LIMIT),
            }
            if feedback:
                record["feedback"] = self._tail_text(feedback, STDOUT_CHARACTER_LIMIT)
            history_records.append(record)
            log_entries.append(
                f"Attempt {attempt} - {record_status.upper()}\n{'=' * 80}\nGenerated code:\n{code.rstrip() or '<empty>'}\n\n"
                f"STDOUT:\n{stdout_text.strip() or '<none>'}\n\nSTDERR:\n{stderr_text.strip() or '<none>'}\n\n"
                f"Feedback:\n{(feedback or 'None').strip()}\n\nResult payload:\n"
                f"{json.dumps(result_payload, indent=2) if result_payload is not None else 'None'}"
            )
            _persist_log()
            prompt_history.append({
                "attempt": str(attempt), "status": record_status,
                "stdout": self._tail_text(stdout_text, 600), "stderr": self._tail_text(stderr_text, 600),
                "feedback": self._tail_text(feedback or record_status, 600),
            })
            if result_payload is not None:
                return result_payload, history_records, log_uri or _persist_log()
        _persist_log()
        raise RuntimeError("LLM analysis failed to produce a valid result after maximum retries")

    def _run_analysis_code(
        self, *, code: str, artifact_map: Dict[str, str], workflow_id: str,
    ) -> tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
        stdout_text = ""
        stderr_text = ""
        result_payload: Optional[Dict[str, Any]] = None
        feedback: Optional[str] = None
        preamble = self._prepare_analysis_environment(workflow_id, artifact_map)
        sandbox = PythonSandbox()
        try:
            sandbox.run(preamble)
        except Exception as exc:
            error_message = f"Failed to initialise analysis runtime: {exc}"
            return "", error_message, None, error_message
        try:
            stdout_text = sandbox.run(code)
        except Exception as exc:
            stderr_text = str(exc)
            feedback = stderr_text or "Python sandbox execution raised an exception"
        result_line: Optional[str] = None
        for line in stdout_text.splitlines():
            if line.startswith(RESULT_PREFIX):
                result_line = line[len(RESULT_PREFIX):].strip()
                break
        if result_line is not None:
            try:
                result_payload = json.loads(result_line)
                feedback = None
            except json.JSONDecodeError as exc:
                feedback = f"Failed to parse RESULT payload: {exc}"
        elif feedback is None:
            feedback = stderr_text.strip() or "No RESULT:: line emitted in stdout"
        return stdout_text, stderr_text, result_payload, feedback

    def _prepare_analysis_environment(self, workflow_id: str, artifact_map: Dict[str, str]) -> str:
        workflow_dir = self.artifact_store.root / workflow_id.replace("/", "_")
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return textwrap.dedent(
            f"""
            import json
            import math
            import statistics
            from pathlib import Path

            DATA_ARTIFACTS = {artifact_map!r}
            ARTIFACT_OUTPUT_DIR = Path({str(workflow_dir.resolve())!r})
            RESULT_PREFIX = {RESULT_PREFIX!r}

            def load_json_artifact(slug: str):
                if slug not in DATA_ARTIFACTS:
                    raise KeyError(f"Unknown data artifact slug: {{slug}}")
                path = Path(DATA_ARTIFACTS[slug])
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            """
        )

    @staticmethod
    def _extract_python_code(response: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    @staticmethod
    def _tail_text(text: str, limit: int) -> str:
        snippet = text.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[-limit:]

"""The research agent: stateful nodes plus a router with bounded, conditional loops.

Graph (each arrow is chosen at run time by :func:`next_stage` from recorded state)::

    plan_generation -> data_collection -> hybrid_analysis -> critique
    critique --CONTINUE / INSUFFICIENT_EVIDENCE / HUMAN_REVIEW--> detailed_analysis
    critique --COLLECT_MORE_DATA / REVISE_PLAN--> data_collection   (plan revised, new data fetched)
    critique --REANALYZE--> hybrid_analysis                         (inconsistent values discarded)
    detailed_analysis -> claim_verification
    claim_verification --failed claims, attempts left--> detailed_analysis (with corrections)
    claim_verification -> report_generation -> [human_review] -> delivery

All state lives in ``context["research"]`` as JSON so stages can run in separate workers and
every decision, revision and tool call is preserved for the report.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from hypothesis_agent.llm import BaseLLM, LLMError
from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.research import analytics
from hypothesis_agent.research.assessment import ExpectationAssessment, VerdictResult, assess_plan, determine_verdict
from hypothesis_agent.research.claims import ClaimReport, narrative_feedback, verify_text
from hypothesis_agent.research.critique import (
    CritiqueResult,
    action_key,
    decide,
    deterministic_critique,
    merge_llm_critique,
)
from hypothesis_agent.research.evidence import EvidenceRegistry, utc_now_iso
from hypothesis_agent.research.metrics import METRICS, RATIO_RECIPES
from hypothesis_agent.research.planner import ResearchPlan, build_plan, revise_plan
from hypothesis_agent.research.tools import ResearchToolbox, summarise_tool_log
from hypothesis_agent.research.uncertainty import build_uncertainty
from hypothesis_agent.storage.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

LOOP_STAGES = {"data_collection", "hybrid_analysis", "critique", "detailed_analysis", "claim_verification"}
PRICE_METRICS = {"price_return", "annualized_volatility", "max_drawdown"}
VALUATION_METRICS = {"pe_computed", "peg_computed"}


class ResearchLimits:
    def __init__(self, max_iterations: int = 3, max_tool_calls: int = 80, max_llm_tool_steps: int = 4,
                 max_narrative_attempts: int = 2, max_reanalysis: int = 1) -> None:
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_llm_tool_steps = max_llm_tool_steps
        self.max_narrative_attempts = max_narrative_attempts
        self.max_reanalysis = max_reanalysis

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def research_state(context: Dict[str, Any]) -> Dict[str, Any]:
    state = context.setdefault("research", {})
    state.setdefault("tool_calls", [])
    state.setdefault("currencies", {})
    state.setdefault("charts", [])
    state.setdefault("computed", [])
    state.setdefault("critiques", [])
    state.setdefault("iterations", [])
    state.setdefault("attempted_actions", [])
    state.setdefault("narratives", [])
    state.setdefault("iteration", 1)
    state.setdefault("reanalysis_count", 0)
    state.setdefault("errors", [])
    return state


def resolve_ticker(request: HypothesisRequest) -> str:
    if not request.entities:
        return "SPY"
    return request.entities[0].strip().upper() or "SPY"


class ResearchAgent:
    """Executes research nodes against a context dict."""

    def __init__(self, *, llm: BaseLLM, toolset: Any, artifact_store: ArtifactStore, limits: Optional[ResearchLimits] = None) -> None:
        self.llm = llm
        self.toolset = toolset
        self.artifact_store = artifact_store
        self.limits = limits or ResearchLimits()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load(self, context: Dict[str, Any]) -> tuple[Dict[str, Any], ResearchPlan, EvidenceRegistry]:
        state = research_state(context)
        if "plan" not in state:
            raise RuntimeError("Research plan missing; run plan_generation first.")
        plan = ResearchPlan.model_validate(state["plan"])
        registry = EvidenceRegistry.from_dict(state.get("registry"))
        return state, plan, registry

    def _save(self, state: Dict[str, Any], plan: ResearchPlan, registry: EvidenceRegistry, workflow_id: str) -> None:
        state["plan"] = plan.model_dump(mode="json")
        state["registry"] = registry.to_dict()
        self.artifact_store.write_json(workflow_id, "evidence_registry", state["registry"])
        self.artifact_store.write_json(workflow_id, "research_plan", state["plan"])

    def _toolbox(self, state: Dict[str, Any], plan: ResearchPlan, registry: EvidenceRegistry, workflow_id: str) -> ResearchToolbox:
        return ResearchToolbox(
            registry=registry, toolset=self.toolset, artifact_store=self.artifact_store, workflow_id=workflow_id,
            allowed_tickers=[plan.ticker], call_log=state["tool_calls"], currencies=state["currencies"],
            charts=state["charts"], iteration=int(state["iteration"]), max_calls=self.limits.max_tool_calls,
        )

    def _llm_call(self, state: Dict[str, Any], name: str, fn, *args, **kwargs):
        """Call an optional LLM capability; failures are recorded (never hidden) and research continues."""

        try:
            return fn(*args, **kwargs)
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            message = f"LLM {name} failed: {type(exc).__name__}: {exc}"
            logger.warning(message)
            state["errors"].append({"at": utc_now_iso(), "stage": name, "error": message})
            return None

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def plan(self, request: HypothesisRequest, context: Dict[str, Any], workflow_id: str) -> ResearchPlan:
        state = research_state(context)
        ticker = resolve_ticker(request)
        catalogue = {key: {"label": d.label, "category": d.category, "unit": d.unit} for key, d in METRICS.items()}
        proposal = self._llm_call(state, "plan_proposal", self.llm.propose_research_plan, request, catalogue)
        plan = build_plan(request.hypothesis_text, ticker, proposal if isinstance(proposal, dict) else None)
        state["limits"] = self.limits.as_dict()
        state["registry"] = EvidenceRegistry().to_dict()
        state["plan_initial"] = plan.model_dump(mode="json")
        self._save(state, plan, EvidenceRegistry(), workflow_id)
        return plan

    def collect(self, request: HypothesisRequest, context: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        state, plan, registry = self._load(context)
        toolbox = self._toolbox(state, plan, registry, workflow_id)
        before = len(registry.records)
        results = []
        ordered = sorted(plan.datasets, key=lambda d: (d != "company_info", plan.datasets.index(d)))
        for dataset in ordered:
            arguments: Dict[str, Any] = {"dataset": dataset, "ticker": plan.ticker}
            if dataset == "price_history":
                arguments["period"] = plan.price_period
            result = toolbox.invoke("fetch_dataset", arguments, origin="plan", rationale=f"Plan v{plan.version} requires {dataset}.")
            results.append({"dataset": dataset, "status": result.status, "summary": result.output.get("summary", result.error)})
        self._save(state, plan, registry, workflow_id)
        return {"results": results, "new_evidence": len(registry.records) - before}

    def analyze(self, request: HypothesisRequest, context: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        state, plan, registry = self._load(context)
        toolbox = self._toolbox(state, plan, registry, workflow_id)
        computed: List[str] = state["computed"]

        def run(tool: str, args: Dict[str, Any], rationale: str) -> None:
            key = f"{tool}|{json.dumps(args, sort_keys=True)}"
            if key in computed:
                return
            result = toolbox.invoke(tool, args, origin="agent", rationale=rationale)
            if result.status in {"ok", "skipped"}:
                computed.append(key)  # failed computations are retried once new data arrives

        ticker = plan.ticker
        for planned in plan.metrics:
            metric, frequency = planned.metric, planned.frequency
            reason = f"Plan v{planned.added_in_version} ({planned.origin}): {planned.rationale}"
            base = metric[: -len("_growth_yoy")] if metric.endswith("_growth_yoy") else metric
            if base in RATIO_RECIPES or base == "free_cash_flow_computed":
                run("compute_metric", {"metric": base, "ticker": ticker, "frequency": frequency}, reason)
            if metric.endswith("_growth_yoy"):
                run("compute_growth", {"metric": base, "ticker": ticker, "frequency": frequency}, reason)
            if metric in PRICE_METRICS:
                run("price_performance", {"ticker": ticker}, reason)
            if metric in VALUATION_METRICS:
                run("valuation_check", {"ticker": ticker}, reason)
            if frequency == "quarterly" and metric in METRICS:
                run("assess_trend", {"metric": metric, "ticker": ticker, "frequency": "quarterly"},
                    "Check the most recent quarters for a persisting reversal.")
        for expectation in plan.expectations:
            if not expectation.measurable or expectation.metric is None or expectation.direction not in {"increase", "decrease"}:
                continue
            args: Dict[str, Any] = {"metric": expectation.metric, "ticker": ticker}
            if expectation.window_start_year:
                args["start_year"] = expectation.window_start_year
            if expectation.window_end_year:
                args["end_year"] = expectation.window_end_year
            run("assess_trend", args, f"Evaluate {expectation.expectation_id}: {expectation.concept} should {expectation.direction}.")

        llm_steps = self._llm_tool_loop(request, state, plan, registry, toolbox)
        for metric in self._chart_metrics(plan, registry):
            run("plot_metric", {"metric": metric, "ticker": ticker}, "Visualise evidence behind a tested expectation.")

        assessments = assess_plan(plan, registry)
        verdict = determine_verdict(assessments)
        state["assessments"] = [a.model_dump(mode="json") for a in assessments]
        state["verdict"] = verdict.model_dump(mode="json")
        state["last_consistency"] = analytics.check_consistency(registry, toolbox.payload)
        self._save(state, plan, registry, workflow_id)
        return {"assessments": assessments, "verdict": verdict, "llm_steps": llm_steps}

    def _chart_metrics(self, plan: ResearchPlan, registry: EvidenceRegistry) -> List[str]:
        metrics: List[str] = []
        candidates = [e.metric for e in plan.expectations if e.metric and e.measurable and e.metric not in VALUATION_METRICS]
        candidates += ["total_revenue", "operating_margin", "free_cash_flow"]
        if any(e.metric == "peg_computed" for e in plan.expectations):
            candidates.insert(0, "diluted_eps")
        for metric in candidates:
            if metric in metrics or metric in PRICE_METRICS:
                continue
            if len(registry.series(plan.ticker, metric)) >= 2:
                metrics.append(metric)
            if len(metrics) >= 4:
                break
        return metrics

    def _llm_tool_loop(self, request: HypothesisRequest, state: Dict[str, Any], plan: ResearchPlan,
                       registry: EvidenceRegistry, toolbox: ResearchToolbox) -> int:
        steps = 0
        for _ in range(self.limits.max_llm_tool_steps):
            briefing = {
                "hypothesis": plan.hypothesis,
                "ticker": plan.ticker,
                "expectations": [e.model_dump(include={"expectation_id", "metric", "direction", "concept"}) for e in plan.expectations],
                "tools": toolbox.describe_tools(compact=True),
                "available_metrics": registry.metrics_available(plan.ticker),
                "recent_tool_calls": summarise_tool_log(state["tool_calls"]),
                "remaining_calls": self.limits.max_tool_calls - toolbox.calls_made,
            }
            action = self._llm_call(state, "choose_next_analysis_action", self.llm.choose_next_analysis_action, request, briefing)
            if not isinstance(action, dict) or action.get("tool") in {None, "finish", "stop"}:
                break
            toolbox.invoke(str(action.get("tool")), action.get("arguments") or {}, origin="llm",
                           rationale=str(action.get("rationale") or "")[:300])
            steps += 1
        return steps

    def critique(self, request: HypothesisRequest, context: Dict[str, Any], workflow_id: str) -> CritiqueResult:
        state, plan, registry = self._load(context)
        iteration = int(state["iteration"])
        assessments = [ExpectationAssessment.model_validate(a) for a in state.get("assessments", [])]
        verdict = VerdictResult.model_validate(state["verdict"]) if state.get("verdict") else determine_verdict(assessments)
        collected = sorted({s.dataset for s in registry.snapshots.values() if s.ticker == plan.ticker})
        issues, actions = deterministic_critique(
            plan=plan, registry=registry, assessments=assessments, verdict=verdict,
            consistency_problems=state.get("last_consistency", []), collected_datasets=collected, iteration=iteration,
        )
        briefing = {
            "hypothesis": plan.hypothesis,
            "verdict": verdict.model_dump(),
            "assessments": [a.model_dump(include={"expectation_id", "metric", "status", "summary", "caveats"}) for a in assessments],
            "deterministic_issues": [i.model_dump(include={"type", "description"}) for i in issues],
            "available_metrics": registry.metrics_available(plan.ticker),
            "allowed_metrics": sorted(METRICS),
        }
        llm_payload = self._llm_call(state, "critique_analysis", self.llm.critique_analysis, request, briefing)
        llm_notes = merge_llm_critique(llm_payload if isinstance(llm_payload, dict) else None, issues, actions, plan, iteration)

        new_evidence = len(registry.records) - int(state.get("evidence_at_iteration_start", 0))
        decision, rationale, to_execute = decide(
            issues=issues, actions=actions, assessments=assessments, iteration=iteration,
            max_iterations=self.limits.max_iterations, reanalysis_count=int(state["reanalysis_count"]),
            max_reanalysis=self.limits.max_reanalysis, new_evidence_last_iteration=new_evidence,
            attempted_actions=state["attempted_actions"],
        )
        result = CritiqueResult(iteration=iteration, status=decision, issues=issues, required_actions=actions,
                                rationale=rationale, llm_notes=llm_notes)
        revision = None
        if decision in {"COLLECT_MORE_DATA", "REVISE_PLAN"}:
            pairs = [pair for action in to_execute for pair in action.metrics]
            for action in to_execute:
                state["attempted_actions"].append(action_key(action))
                for dataset in action.datasets:
                    if dataset not in plan.datasets:
                        plan.datasets.append(dataset)
            triggered = sorted({t for action in to_execute for t in action.triggered_by})
            revision = revise_plan(plan, iteration=iteration, reason=rationale, triggered_by=triggered, metrics=pairs)
        if decision == "REANALYZE":
            for problem in state.get("last_consistency", []):
                registry.remove(problem["evidence_id"])
            state["computed"] = []
            state["reanalysis_count"] = int(state["reanalysis_count"]) + 1
        if decision == "HUMAN_REVIEW":
            state["human_review"] = {"requested": True, "reason": rationale}
        state["critiques"].append(result.model_dump(mode="json"))
        state["iterations"].append({
            "iteration": iteration,
            "plan_version": plan.version,
            "evidence_records": len(registry.records),
            "new_evidence": new_evidence,
            "tool_calls": sum(1 for c in state["tool_calls"] if c["iteration"] == iteration),
            "verdict": verdict.verdict,
            "decision": decision,
            "rationale": rationale,
            "revision": revision.model_dump(mode="json") if revision else None,
            "issues": [i.type for i in issues],
            "at": utc_now_iso(),
        })
        state["decision"] = {"status": decision, "rationale": rationale, "iteration": iteration}
        if decision in {"COLLECT_MORE_DATA", "REVISE_PLAN", "REANALYZE"}:
            state["iteration"] = iteration + 1
            state["evidence_at_iteration_start"] = len(registry.records)
        state["iteration_limit_hit"] = iteration >= self.limits.max_iterations and bool(
            [a for a in actions if action_key(a) not in state["attempted_actions"] and a.action != "recompute"])
        self._save(state, plan, registry, workflow_id)
        self.artifact_store.write_json(workflow_id, f"critique_iteration_{iteration}", result.model_dump(mode="json"))
        return result

    def briefing(self, plan: ResearchPlan, registry: EvidenceRegistry, state: Dict[str, Any]) -> Dict[str, Any]:
        assessments = state.get("assessments", [])
        ids: List[str] = []
        for a in assessments:
            ids.extend(a.get("evidence_ids", []))
        for planned in plan.metrics:
            latest = registry.latest(plan.ticker, planned.metric)
            if latest is not None:
                ids.append(latest.evidence_id)
        ids = list(dict.fromkeys(i for i in ids if registry.has(i)))[:60]
        key_evidence = []
        for eid in ids:
            record = registry.get(eid)
            key_evidence.append({
                "evidence_id": eid, "metric": record.metric, "label": record.label, "period": record.period,
                "value": record.value, "display": display_value(record.value, record.unit), "unit": record.unit,
                "qualifier": record.qualifier,
            })
        critique = state["critiques"][-1] if state.get("critiques") else {}
        return {
            "hypothesis": plan.hypothesis,
            "ticker": plan.ticker,
            "verdict": state.get("verdict"),
            "assessments": assessments,
            "key_evidence": key_evidence,
            "critique_issues": [i["description"] for i in critique.get("issues", [])],
            "critique_issue_types": [i["type"] for i in critique.get("issues", [])],
            "decision": state.get("decision"),
            "plan_revisions": [r.model_dump(mode="json") for r in plan.revisions],
            "missing": [m.model_dump(mode="json") for m in registry.missing if m.metric != "*"][-15:],
            "rules": [
                "Use only figures listed in key_evidence and cite each one with its evidence_id in square brackets, e.g. [EV-1a2b3c4d].",
                "State the verdict exactly as given in verdict.verdict; do not upgrade or downgrade it.",
                "Name contradicting evidence explicitly. Do not invent data that was not collected.",
                "Always state the fiscal year of each figure.",
            ],
        }

    def narrate(self, request: HypothesisRequest, context: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        state, plan, registry = self._load(context)
        brief = self.briefing(plan, registry, state)
        attempts = state["narratives"]
        if attempts and attempts[-1].get("feedback"):
            brief["corrections"] = attempts[-1]["feedback"]
        narrative = self.llm.generate_research_narrative(request, brief)
        if not isinstance(narrative, dict) or not str(narrative.get("narrative", "")).strip():
            raise RuntimeError("LLM failed to generate a research narrative")
        report_payload = self.llm.generate_report(
            request=request, metrics_overview=brief, analysis_summary=str(narrative["narrative"]),
            artifact_paths=[c["artifact_uri"] for c in state["charts"]],
        )
        if not isinstance(report_payload, dict):
            raise RuntimeError("LLM report payload must be a dictionary")
        entry = {"attempt": len(attempts) + 1, "narrative": narrative, "report_payload": report_payload, "at": utc_now_iso()}
        attempts.append(entry)
        self._save(state, plan, registry, workflow_id)
        return entry

    def verify_claims(self, request: HypothesisRequest, context: Dict[str, Any], workflow_id: str) -> ClaimReport:
        state, plan, registry = self._load(context)
        if not state["narratives"]:
            raise RuntimeError("No narrative to verify; run detailed_analysis first.")
        latest = state["narratives"][-1]
        sections = narrative_sections(latest)
        verdict = (state.get("verdict") or {}).get("verdict")
        loader = analytics.default_payload_loader(registry)
        report = verify_text(sections, registry, plan.ticker, verdict=verdict, loader=loader)
        latest["claim_counts"] = report.counts
        failed = narrative_feedback(report)
        if failed and len(state["narratives"]) < self.limits.max_narrative_attempts:
            latest["feedback"] = failed
            state["narrative_retry"] = True
        else:
            state["narrative_retry"] = False
        state["claims"] = report.model_dump(mode="json")
        state["claims_attempt"] = len(state["narratives"])
        self.artifact_store.write_json(workflow_id, "claim_verification", state["claims"])
        self._save(state, plan, registry, workflow_id)
        return report

    def finalize(self, context: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        state, plan, registry = self._load(context)
        assessments = [ExpectationAssessment.model_validate(a) for a in state.get("assessments", [])]
        issues = state["critiques"][-1]["issues"] if state.get("critiques") else []
        uncertainty = build_uncertainty(plan=plan, registry=registry, assessments=assessments, critique_issues=issues,
                                        iteration_limit_hit=bool(state.get("iteration_limit_hit")))
        state["uncertainty"] = uncertainty.model_dump(mode="json")
        self._save(state, plan, registry, workflow_id)
        manifest = reproducibility_manifest(state, registry)
        self.artifact_store.write_json(workflow_id, "reproducibility_manifest", manifest)
        return state["uncertainty"]


def narrative_sections(entry: Dict[str, Any]) -> Dict[str, str]:
    narrative = entry.get("narrative") or {}
    payload = entry.get("report_payload") or {}
    sections: Dict[str, str] = {}
    for key in ("executive_summary", "narrative"):
        if narrative.get(key):
            sections[f"narrative.{key}"] = str(narrative[key])
    for key in ("supporting_points", "contradicting_points", "limitations"):
        if narrative.get(key):
            sections[f"narrative.{key}"] = " ".join(_as_sentences(narrative[key]))
    if payload.get("executive_summary"):
        sections["report.executive_summary"] = str(payload["executive_summary"])
    for key in ("key_findings", "risks", "next_steps"):
        if payload.get(key):
            sections[f"report.{key}"] = " ".join(_as_sentences(payload[key]))
    return sections


def _as_sentences(items: Any) -> List[str]:
    if isinstance(items, str):
        return [items]
    out = []
    for item in items or []:
        text = str(item).strip()
        if text and text[-1] not in ".!?":
            text += "."
        out.append(text)
    return out


def display_value(value: float, unit: str) -> str:
    if unit == "ratio":
        return f"{value * 100:.1f}%"
    if unit == "ratio_points":
        return f"{value * 100:+.1f} percentage points"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit in {"x"}:
        return f"{value:.2f}x"
    if unit == "days":
        return f"{value:.0f} days"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if unit.endswith("/share"):
        return f"{sign}${magnitude:,.2f}" if unit.startswith("USD") else f"{value:,.2f} {unit}"
    if unit == "shares":
        return f"{value / 1e9:,.2f}B shares" if magnitude >= 1e9 else f"{value:,.0f} shares"
    prefix = "$" if unit == "USD" else ""
    suffix = "" if unit == "USD" else f" {unit}"
    if magnitude >= 1e9:
        return f"{sign}{prefix}{magnitude / 1e9:,.2f} billion{suffix}"
    if magnitude >= 1e6:
        return f"{sign}{prefix}{magnitude / 1e6:,.1f} million{suffix}"
    return f"{sign}{prefix}{magnitude:,.2f}{suffix}"


def reproducibility_manifest(state: Dict[str, Any], registry: EvidenceRegistry) -> Dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "instructions": ("Each snapshot's raw payload is stored at artifact_path with its SHA-256. Raw evidence records "
                         "locate their value via locator; derived records list the operation, formula and input IDs. "
                         "Run `jasper reproduce <run_dir>` to re-extract and recompute everything from the stored payloads."),
        "snapshots": [s.model_dump(mode="json") for s in registry.snapshots.values()],
        "derived": [
            {"evidence_id": r.evidence_id, "metric": r.metric, "value": r.value, "operation": r.derivation.operation,
             "formula": r.derivation.formula, "inputs": r.derivation.inputs, "input_snapshots": r.derivation.input_snapshots,
             "parameters": r.derivation.parameters}
            for r in registry.records.values() if r.derivation is not None
        ],
        "verdict": state.get("verdict"),
        "limits": state.get("limits"),
    }


def next_stage(stage: str, context: Dict[str, Any], *, human_review_required: bool = False) -> Optional[str]:
    """Router: decide the next stage from recorded research state."""

    state = context.get("research") or {}
    if stage == "plan_generation":
        return "data_collection"
    if stage == "data_collection":
        return "hybrid_analysis"
    if stage == "hybrid_analysis":
        return "critique"
    if stage == "critique":
        decision = (state.get("decision") or {}).get("status", "CONTINUE")
        if decision in {"COLLECT_MORE_DATA", "REVISE_PLAN"}:
            return "data_collection"
        if decision == "REANALYZE":
            return "hybrid_analysis"
        return "detailed_analysis"
    if stage == "detailed_analysis":
        return "claim_verification"
    if stage == "claim_verification":
        return "detailed_analysis" if state.get("narrative_retry") else "report_generation"
    if stage == "report_generation":
        requested = bool((state.get("human_review") or {}).get("requested"))
        return "human_review" if (human_review_required or requested) else "delivery"
    if stage == "human_review":
        return "delivery"
    return None

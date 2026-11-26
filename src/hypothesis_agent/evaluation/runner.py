"""Run JASPER over a benchmark suite and write machine- and human-readable results."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, List, Optional

from hypothesis_agent.config import AppSettings
from hypothesis_agent.evaluation.benchmark import BenchmarkCase, BenchmarkSuite
from hypothesis_agent.evaluation.judge import DeterministicRubricJudge, LLMRubricJudge
from hypothesis_agent.evaluation.scoring import judge_view, score_case
from hypothesis_agent.llm import BaseLLM, NvidiaLLM
from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.orchestration.fixture_tools import FixtureToolSet
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.research.offline_llm import DeterministicResearchLLM
from hypothesis_agent.storage.artifact_store import ArtifactStore

SCORE_KEYS = ["verdict_accuracy", "numerical_accuracy", "metric_relevance", "logical_coherence",
              "evidence_coverage", "unsupported_claim_rate", "caveat_recall"]
RESULTS_SCHEMA_VERSION = "1"


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def jasper_version() -> Dict[str, Any]:
    try:
        from importlib.metadata import version

        package_version = version("hypothesis-agent")
    except Exception:  # package not installed in editable mode
        package_version = "unknown"
    status = _git("status", "--porcelain")
    return {
        "package_version": package_version,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
    }


def make_llm(mode: str, settings: AppSettings) -> BaseLLM:
    if mode == "offline":
        return DeterministicResearchLLM()
    if mode == "nvidia":
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is required for --llm nvidia")
        return NvidiaLLM(api_key=settings.nvidia_api_key, model=settings.nvidia_model)
    raise ValueError(f"Unknown LLM mode '{mode}'")


def run_case(
    case: BenchmarkCase,
    *,
    suite: BenchmarkSuite,
    llm: BaseLLM,
    settings: AppSettings,
    runs_root: Path,
    judges: List[Any],
) -> Dict[str, Any]:
    workflow_id = f"bench-{case.id}"
    run_dir = runs_root / workflow_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    store = ArtifactStore.from_path(str(runs_root))
    orchestrator = LangGraphValidationOrchestrator(
        settings=settings, llm=llm, artifact_store=store, toolset=FixtureToolSet(suite.fixture_dir or ""),
    )
    # The agent receives only what a user would type: hypothesis text and ticker.
    request = HypothesisRequest(
        user_id="benchmark", hypothesis_text=case.hypothesis, entities=[case.ticker],
        time_horizon=TimeHorizon(start=date(2020, 1, 1), end=date(2026, 12, 31)),
    )
    started = time.perf_counter()
    base = {"case_id": case.id, "category": case.category, "ticker": case.ticker, "hypothesis": case.hypothesis,
            "expected_verdict": case.expected_verdict, "acceptable_verdicts": case.acceptable_verdicts}
    try:
        context = orchestrator.run_research(request, {"metadata": {"workflow_id": workflow_id}})
    except Exception as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}", "failure_categories": ["RUN_ERROR"],
                "duration_s": round(time.perf_counter() - started, 2), "scores": {}}
    duration = round(time.perf_counter() - started, 2)
    scored = score_case(case, context, run_dir)
    rubric = {}
    for judge in judges:
        if isinstance(judge, LLMRubricJudge):
            result = judge.score(judge_view(case, context))
        else:
            result = judge.score(scored["signals"])
        rubric[result.judge] = result.model_dump(mode="json")
    state = context.get("research") or {}
    scored.pop("signals", None)
    return {
        **base,
        "status": "ok",
        **scored,
        "rubric": rubric,
        "route": context.get("metadata", {}).get("route", []),
        "iterations": [{"iteration": it["iteration"], "decision": it["decision"], "rationale": it["rationale"],
                        "added_metrics": (it.get("revision") or {}).get("added_metrics", [])} for it in state.get("iterations", [])],
        "plan_revisions": len((state.get("plan") or {}).get("revisions", [])),
        "narrative_attempts": len(state.get("narratives", [])),
        "human_review_requested": bool((state.get("human_review") or {}).get("requested")),
        "tool_calls": len(state.get("tool_calls", [])),
        "llm_tool_calls": sum(1 for c in state.get("tool_calls", []) if c.get("origin") == "llm"),
        "llm_errors": state.get("errors", []),
        "report_pdf": (context.get("report") or {}).get("pdf_path"),
        "run_dir": str(run_dir),
        "duration_s": duration,
    }


def aggregate(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [c for c in cases if c.get("status") == "ok"]
    means: Dict[str, Optional[float]] = {}
    for key in SCORE_KEYS:
        values = [c["scores"][key] for c in ok if c["scores"].get(key) is not None]
        means[key] = round(fmean(values), 4) if values else None
    by_category: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cases": 0, "verdict_correct": 0})
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        bucket = by_category[c["category"]]
        bucket["cases"] += 1
        bucket["verdict_correct"] += int(bool(c.get("verdict_correct")))
        confusion[c["expected_verdict"]][c.get("verdict") or "ERROR"] += 1
    failures = Counter(f for c in cases for f in c.get("failure_categories", []))
    claim_totals: Counter = Counter()
    for c in ok:
        claim_totals.update(c.get("claim_counts", {}))
    return {
        "cases": len(cases),
        "completed": len(ok),
        "errors": len(cases) - len(ok),
        "verdict_accuracy_acceptable": round(sum(bool(c.get("verdict_correct")) for c in cases) / len(cases), 4) if cases else None,
        "verdict_accuracy_exact": round(sum(bool(c.get("verdict_exact")) for c in cases) / len(cases), 4) if cases else None,
        "mean_scores": means,
        "by_category": {k: {**v, "accuracy": round(v["verdict_correct"] / v["cases"], 4)} for k, v in sorted(by_category.items())},
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "failure_categories": dict(failures.most_common()),
        "claim_totals": dict(claim_totals),
        "numerical_discrepancies": sum(len(c.get("numerical_discrepancies", [])) for c in ok),
        "unsupported_claims": sum(len(c.get("unsupported_claims", [])) for c in ok),
        "reproducible_runs": sum(1 for c in ok if c.get("reproducibility", {}).get("reproducible")),
        "plan_revisions": sum(c.get("plan_revisions", 0) for c in ok),
        "human_review_escalations": sum(1 for c in ok if c.get("human_review_requested")),
    }


def run_suite(
    suite: BenchmarkSuite,
    *,
    llm_mode: str = "offline",
    judge_mode: str = "deterministic",
    label: Optional[str] = None,
    out_root: str | Path = "eval_results",
    settings: Optional[AppSettings] = None,
    progress: Callable[[str], None] = print,
) -> Path:
    settings = settings or AppSettings()
    label = label or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{llm_mode}"
    out_dir = Path(out_root).resolve() / label
    runs_root = out_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_settings = settings.model_copy(update={"artifact_store_path": str(runs_root), "llm_provider": "offline" if llm_mode == "offline" else "nvidia"})
    llm = make_llm(llm_mode, settings)
    judges: List[Any] = [DeterministicRubricJudge()]
    if judge_mode == "nvidia":
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is required for --judge nvidia")
        judges.append(LLMRubricJudge(NvidiaLLM(api_key=settings.nvidia_api_key, model=settings.nvidia_model)))
    started = datetime.now(timezone.utc)
    results: List[Dict[str, Any]] = []
    for index, case in enumerate(suite.cases, start=1):
        progress(f"[{index}/{len(suite.cases)}] {case.id} ...")
        result = run_case(case, suite=suite, llm=llm, settings=run_settings, runs_root=runs_root, judges=judges)
        progress(f"    -> {result.get('verdict') or result.get('error')} (expected {case.expected_verdict}); "
                 f"failures: {', '.join(result.get('failure_categories', [])) or 'none'}")
        results.append(result)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "label": label,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jasper": jasper_version(),
        "suite": {"name": suite.suite, "version": suite.version, "cases": len(suite.cases), "content_sha256": suite.content_sha256,
                  "root": suite.root},
        "configuration": {
            "llm_mode": llm_mode,
            "llm_model": getattr(llm, "model", getattr(llm, "name", type(llm).__name__)),
            "judges": [getattr(j, "name", type(j).__name__) for j in judges],
            "limits": {k: getattr(run_settings, k) for k in ("research_max_iterations", "research_max_tool_calls",
                                                               "research_max_llm_tool_steps", "research_max_narrative_attempts")},
        },
        "aggregate": aggregate(results),
        "cases": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    return out_dir


def _fmt(value: Any, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.0%}" if pct else f"{value:.2f}"
    return str(value)


def render_summary(payload: Dict[str, Any]) -> str:
    agg = payload["aggregate"]
    lines = [
        f"# JASPER benchmark results: {payload['label']}",
        "",
        f"- Suite: {payload['suite']['name']} v{payload['suite']['version']} ({payload['suite']['cases']} cases, content sha256 `{(payload['suite']['content_sha256'] or '')[:16]}`)",
        f"- JASPER: package {payload['jasper']['package_version']}, commit `{payload['jasper']['git_commit']}`"
        f"{' (uncommitted changes)' if payload['jasper']['git_dirty'] else ''}",
        f"- LLM: {payload['configuration']['llm_mode']} ({payload['configuration']['llm_model']}); judges: {', '.join(payload['configuration']['judges'])}",
        f"- Run: {payload['started_at']} to {payload['finished_at']}",
        "",
        "## Aggregate",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Verdict within acceptable range | {_fmt(agg['verdict_accuracy_acceptable'])} |",
        f"| Verdict exactly as expected | {_fmt(agg['verdict_accuracy_exact'])} |",
    ]
    for key, value in agg["mean_scores"].items():
        lines.append(f"| Mean {key.replace('_', ' ')} | {_fmt(value)} |")
    lines += [
        f"| Numerical discrepancies (contradicted figures) | {agg['numerical_discrepancies']} |",
        f"| Unsupported or contradicted claims | {agg['unsupported_claims']} |",
        f"| Reproducible runs | {agg['reproducible_runs']}/{agg['completed']} |",
        f"| Plan revisions across suite | {agg['plan_revisions']} |",
        f"| Human-review escalations | {agg['human_review_escalations']} |",
        f"| Run errors | {agg['errors']} |",
        "",
        "## Accuracy by category",
        "",
        "| Category | Cases | Verdict correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for category, row in agg["by_category"].items():
        lines.append(f"| {category} | {row['cases']} | {row['verdict_correct']} | {_fmt(row['accuracy'])} |")
    lines += ["", "## Confusion (expected -> actual)", "", "| Expected | Actual counts |", "|---|---|"]
    for expected, actual in agg["confusion"].items():
        lines.append(f"| {expected} | {', '.join(f'{k}: {v}' for k, v in actual.items())} |")
    lines += ["", "## Failure categories", ""]
    lines += [f"- {k}: {v}" for k, v in agg["failure_categories"].items()] or ["- none"]
    lines += ["", "## Cases", "",
              "| Case | Category | Expected | Actual | OK | Numeric acc. | Coverage | Relevance | Coherence | Caveats | Unsupported | Failures |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in payload["cases"]:
        s = case.get("scores", {})
        lines.append(
            f"| {case['case_id']} | {case['category']} | {case['expected_verdict']} | {case.get('verdict') or 'ERROR'} | "
            f"{'yes' if case.get('verdict_correct') else 'no'} | {_fmt(s.get('numerical_accuracy'))} | {_fmt(s.get('evidence_coverage'))} | "
            f"{_fmt(s.get('metric_relevance'))} | {_fmt(s.get('logical_coherence'))} | {_fmt(s.get('caveat_recall'))} | "
            f"{len(case.get('unsupported_claims', []))} | {', '.join(case.get('failure_categories', [])) or '-'} |"
        )
    details = [c for c in payload["cases"] if c.get("numerical_discrepancies") or c.get("unsupported_claims") or c.get("status") == "error"]
    if details:
        lines += ["", "## Discrepancies and unsupported claims", ""]
        for case in details:
            if case.get("status") == "error":
                lines.append(f"- **{case['case_id']}** run error: {case.get('error')}")
                continue
            for item in case.get("unsupported_claims", []):
                lines.append(f"- **{case['case_id']} {item['claim_id']}** [{item['status']}] {item['text']} -> {item['reason']}")
    missed = [(c["case_id"], cav["description"]) for c in payload["cases"] for cav in c.get("caveats", []) if not cav["covered"]]
    if missed:
        lines += ["", "## Caveats the run did not surface", ""]
        lines += [f"- {cid}: {desc}" for cid, desc in missed]
    lines += ["", "## How to read this", "",
              "- Verdict accuracy compares the rule-based verdict with the benchmark's expected verdict; 'acceptable' allows a stated range.",
              "- Numerical accuracy is the share of figures in generated text that independently re-verify against evidence reloaded from disk.",
              "- Coherence, relevance, coverage and caveat recall are deterministic checks; rubric scores (1-5) are in results.json.",
              "- Expected conclusions were written from the frozen data in the same development cycle as the rule-based components; "
              "treat offline-mode verdict accuracy as an optimistic upper bound and see the paraphrase probe for robustness."]
    return "\n".join(lines) + "\n"

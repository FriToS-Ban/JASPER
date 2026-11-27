"""Command-line entry points.

    jasper research --ticker KO --hypothesis "..." [--llm offline|nvidia] [--fixtures benchmarks/v1/fixtures]
    jasper eval run --suite benchmarks/v1 [--llm offline|nvidia] [--judge deterministic|nvidia] [--label NAME]
    jasper eval compare eval_results/A/results.json eval_results/B/results.json
    jasper eval human-template eval_results/A/results.json review.csv
    jasper eval human-import eval_results/A/results.json review.csv
    jasper reproduce data/artifacts/<workflow_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

from hypothesis_agent.config import AppSettings


def _settings(args: argparse.Namespace) -> AppSettings:
    overrides = {}
    if getattr(args, "llm", None):
        overrides["llm_provider"] = args.llm
    if getattr(args, "fixtures", None):
        overrides["research_fixture_dir"] = str(Path(args.fixtures).resolve())
    if getattr(args, "out", None) and args.command == "research":
        overrides["artifact_store_path"] = str(Path(args.out).resolve())
    if getattr(args, "max_iterations", None):
        overrides["research_max_iterations"] = args.max_iterations
    return AppSettings(**overrides)


def cmd_research(args: argparse.Namespace) -> int:
    from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
    from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator

    settings = _settings(args)
    orchestrator = LangGraphValidationOrchestrator(settings=settings)
    request = HypothesisRequest(
        user_id="cli", hypothesis_text=args.hypothesis, entities=[args.ticker],
        time_horizon=TimeHorizon(start=date(2020, 1, 1), end=date.today()), requires_human_review=args.human_review,
    )
    workflow_id = args.workflow_id or f"cli-{args.ticker.upper()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    context = orchestrator.run_research(request, {"metadata": {"workflow_id": workflow_id}})
    state = context["research"]
    verdict = state["verdict"]
    print(f"Verdict: {verdict['verdict']}  ({verdict['rationale']})")
    for assessment in state["assessments"]:
        print(f"  {assessment['expectation_id']} {assessment['status']}: {assessment['summary']}")
    print(f"Route: {' > '.join(context['metadata']['route'])}")
    for iteration in state["iterations"]:
        print(f"  iteration {iteration['iteration']}: {iteration['decision']} - {iteration['rationale']}")
    print(f"Claims: {state['claims']['counts']}")
    if (state.get("human_review") or {}).get("requested"):
        print(f"Human review requested: {state['human_review']['reason']}")
    print(f"Report: {context['report']['pdf_path']}")
    print(f"Run directory: {Path(settings.artifact_store_path).resolve() / workflow_id}")
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    from hypothesis_agent.evaluation.benchmark import load_suite
    from hypothesis_agent.evaluation.runner import run_suite

    suite = load_suite(args.suite, include_paraphrases=args.include_paraphrases, include_holdout=args.include_holdout,
                       core=not args.only_extra, case_ids=args.cases.split(",") if args.cases else None)
    settings = AppSettings()
    if args.max_iterations:
        settings = settings.model_copy(update={"research_max_iterations": args.max_iterations})
    out_dir = run_suite(suite, llm_mode=args.llm, judge_mode=args.judge, label=args.label, out_root=args.out, settings=settings)
    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    agg = payload["aggregate"]
    print(f"\nVerdict accuracy (acceptable): {agg['verdict_accuracy_acceptable']}, exact: {agg['verdict_accuracy_exact']}")
    print(f"Mean scores: {json.dumps(agg['mean_scores'])}")
    print(f"Failure categories: {agg['failure_categories']}")
    print(f"Results: {out_dir / 'results.json'}\nSummary: {out_dir / 'summary.md'}")
    return 0 if agg["errors"] == 0 else 1


def cmd_eval_compare(args: argparse.Namespace) -> int:
    from hypothesis_agent.evaluation.compare import compare_results, render_comparison

    result = compare_results(args.baseline, args.candidate)
    text = render_comparison(result)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(text)
    return 0


def cmd_human_template(args: argparse.Namespace) -> int:
    from hypothesis_agent.evaluation.human import export_template

    path = export_template(args.results, args.out_csv)
    print(f"Wrote {path} and {path.with_suffix('.rubric.md')}")
    return 0


def cmd_human_import(args: argparse.Namespace) -> int:
    from hypothesis_agent.evaluation.human import import_scores

    print(json.dumps(import_scores(args.results, args.scores_csv), indent=2))
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    from hypothesis_agent.research.reproduce import verify_run_directory

    report = verify_run_directory(args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["reproducible"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jasper", description="JASPER evidence-grounded investment research agent")
    sub = parser.add_subparsers(dest="command", required=True)

    research = sub.add_parser("research", help="Run the research agent on one hypothesis")
    research.add_argument("--ticker", required=True)
    research.add_argument("--hypothesis", required=True)
    research.add_argument("--llm", choices=["offline", "nvidia"], help="Override LLM_PROVIDER")
    research.add_argument("--fixtures", help="Replay frozen fixtures from this directory instead of live Yahoo Finance")
    research.add_argument("--out", help="Artifact directory (default ARTIFACT_STORE_PATH)")
    research.add_argument("--workflow-id")
    research.add_argument("--max-iterations", type=int)
    research.add_argument("--human-review", action="store_true", help="Mark the request as requiring human review")
    research.set_defaults(func=cmd_research)

    evaluation = sub.add_parser("eval", help="Benchmark evaluation")
    eval_sub = evaluation.add_subparsers(dest="eval_command", required=True)
    run = eval_sub.add_parser("run", help="Run the benchmark suite")
    run.add_argument("--suite", default="benchmarks/v1")
    run.add_argument("--llm", choices=["offline", "nvidia"], default="offline")
    run.add_argument("--judge", choices=["deterministic", "nvidia"], default="deterministic")
    run.add_argument("--label")
    run.add_argument("--out", default="eval_results")
    run.add_argument("--cases", help="Comma-separated case ids")
    run.add_argument("--include-paraphrases", action="store_true", help="Add the reworded-hypothesis robustness probe")
    run.add_argument("--include-holdout", action="store_true", help="Add held-out cases never used during development")
    run.add_argument("--only-extra", action="store_true", help="Run only the included paraphrase/holdout files")
    run.add_argument("--max-iterations", type=int)
    run.set_defaults(func=cmd_eval_run)
    compare = eval_sub.add_parser("compare", help="Compare two results.json files")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--out")
    compare.set_defaults(func=cmd_eval_compare)
    template = eval_sub.add_parser("human-template", help="Export a blind human-review CSV")
    template.add_argument("results")
    template.add_argument("out_csv")
    template.set_defaults(func=cmd_human_template)
    human = eval_sub.add_parser("human-import", help="Import human scores into results.json")
    human.add_argument("results")
    human.add_argument("scores_csv")
    human.set_defaults(func=cmd_human_import)

    reproduce = sub.add_parser("reproduce", help="Re-verify a run directory from stored payloads")
    reproduce.add_argument("run_dir")
    reproduce.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

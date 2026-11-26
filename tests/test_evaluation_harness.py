"""Benchmark loading, reproducible execution, scoring, failure reporting, comparison and human review."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.evaluation.benchmark import load_suite
from hypothesis_agent.evaluation.compare import compare_results, render_comparison
from hypothesis_agent.evaluation.human import export_template, import_scores
from hypothesis_agent.evaluation.runner import run_suite
from hypothesis_agent.evaluation.scoring import verdict_score

SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "v1"
CASES = ["KO-01", "MRNA-01"]


def _quiet(_: str) -> None:
    return None


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("eval")
    suite = load_suite(SUITE, case_ids=CASES)
    return run_suite(suite, llm_mode="offline", label="a", out_root=out, settings=AppSettings(), progress=_quiet)


def test_benchmark_suite_loads_and_is_well_formed() -> None:
    suite = load_suite(SUITE)
    assert 15 <= len(suite.cases) <= 20
    categories = {c.category for c in suite.cases}
    assert categories == {"supported", "contradicted", "partially_supported", "insufficient_evidence",
                          "conflicting_metrics", "period_dependent"}
    assert all(c.expected_verdict in c.acceptable_verdicts and c.expected_evidence_categories for c in suite.cases)
    assert suite.content_sha256 and len(suite.content_sha256) == 64
    extended = load_suite(SUITE, include_paraphrases=True, include_holdout=True)
    assert len(extended.cases) > len(suite.cases)


def test_benchmark_loader_rejects_bad_suites(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    (bad / "fixtures").mkdir(parents=True)
    payload = json.loads((SUITE / "cases.json").read_text(encoding="utf-8"))
    payload["cases"] = payload["cases"][:1] * 2
    (bad / "cases.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_suite(bad)
    payload["cases"] = payload["cases"][:1]
    (bad / "cases.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_suite(bad)


def test_expected_answers_never_reach_the_agent(offline_run: Path) -> None:
    results = json.loads((offline_run / "results.json").read_text(encoding="utf-8"))
    for case in results["cases"]:
        plan = json.loads((Path(case["run_dir"]) / "research_plan.json").read_text(encoding="utf-8"))
        text = json.dumps(plan)
        assert case["expected_verdict"] not in text


def test_run_produces_scored_results_and_summary(offline_run: Path) -> None:
    results = json.loads((offline_run / "results.json").read_text(encoding="utf-8"))
    assert results["schema_version"] == "1" and results["jasper"]["git_commit"] is not None
    by_id = {c["case_id"]: c for c in results["cases"]}
    assert by_id["KO-01"]["verdict"] == "SUPPORTED" and by_id["KO-01"]["verdict_correct"]
    assert by_id["MRNA-01"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    for case in results["cases"]:
        assert set(case["scores"]) == {"verdict_accuracy", "numerical_accuracy", "metric_relevance", "logical_coherence",
                                       "evidence_coverage", "unsupported_claim_rate", "caveat_recall"}
        assert case["reproducibility"]["reproducible"]
        assert case["verification_disagreements"] == []
        assert "deterministic-rubric" in case["rubric"]
    assert by_id["KO-01"]["plan_revisions"] >= 1
    summary = (offline_run / "summary.md").read_text(encoding="utf-8")
    assert "| KO-01 |" in summary and "Failure categories" in summary


def test_offline_runs_are_reproducible(offline_run: Path, tmp_path: Path) -> None:
    suite = load_suite(SUITE, case_ids=CASES)
    second = run_suite(suite, llm_mode="offline", label="b", out_root=tmp_path, settings=AppSettings(), progress=_quiet)
    first_cases = json.loads((offline_run / "results.json").read_text(encoding="utf-8"))["cases"]
    second_cases = json.loads((second / "results.json").read_text(encoding="utf-8"))["cases"]
    for a, b in zip(first_cases, second_cases):
        assert a["verdict"] == b["verdict"] and a["scores"] == b["scores"] and a["claim_counts"] == b["claim_counts"]


def test_scoring_detects_failures(offline_run: Path, tmp_path: Path) -> None:
    from hypothesis_agent.evaluation.benchmark import BenchmarkCase
    from hypothesis_agent.evaluation.scoring import score_case

    results = json.loads((offline_run / "results.json").read_text(encoding="utf-8"))
    ko = next(c for c in results["cases"] if c["case_id"] == "KO-01")
    run_dir = tmp_path / "ko"
    shutil.copytree(ko["run_dir"], run_dir)
    # Rebuild the context the scorer needs from disk, then inject a wrong figure into the narrative.
    from hypothesis_agent.config import AppSettings as _S  # noqa: F401
    suite = load_suite(SUITE, case_ids=["KO-01"])
    case: BenchmarkCase = suite.cases[0]
    state = {
        "registry": json.loads((run_dir / "evidence_registry.json").read_text(encoding="utf-8")),
        "plan": json.loads((run_dir / "research_plan.json").read_text(encoding="utf-8")),
        "assessments": [], "verdict": {"verdict": "NOT_SUPPORTED"},
        "narratives": [{"narrative": {"narrative": "Operating margin was 45.0% in FY2025. The hypothesis is supported."}}],
        "claims": {"claims": []}, "uncertainty": {}, "critiques": [],
    }
    scored = score_case(case, {"research": state}, run_dir)
    assert "VERDICT_ERROR" in scored["failure_categories"]
    assert "NUMERICAL_DISCREPANCY" in scored["failure_categories"]
    assert "INCOHERENT_REASONING" in scored["failure_categories"]
    assert "VERIFICATION_DISAGREEMENT" in scored["failure_categories"]  # run reported no claims; independent check finds some
    assert scored["numerical_discrepancies"][0]["reported"] == "45.0%"


def test_verdict_score_scale() -> None:
    suite = load_suite(SUITE, case_ids=["AMZN-01"])
    case = suite.cases[0]  # expected PARTIALLY_SUPPORTED, acceptable also NOT_SUPPORTED
    assert verdict_score("PARTIALLY_SUPPORTED", case) == 1.0
    assert verdict_score("NOT_SUPPORTED", case) == 0.75
    assert verdict_score("SUPPORTED", case) == 0.25
    assert verdict_score("INSUFFICIENT_EVIDENCE", case) == 0.0


def test_version_comparison_reports_regressions(offline_run: Path, tmp_path: Path) -> None:
    baseline = offline_run / "results.json"
    candidate_payload = json.loads(baseline.read_text(encoding="utf-8"))
    candidate_payload["label"] = "candidate"
    ko = next(c for c in candidate_payload["cases"] if c["case_id"] == "KO-01")
    ko.update({"verdict": "NOT_SUPPORTED", "verdict_correct": False, "failure_categories": ["VERDICT_ERROR"]})
    ko["scores"]["numerical_accuracy"] = 0.5
    candidate_payload["aggregate"]["verdict_accuracy_acceptable"] = 0.5
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
    result = compare_results(baseline, candidate)
    change = next(c for c in result["case_changes"] if c["case_id"] == "KO-01")
    assert "regressed" in change["verdict"] and change["new_failures"] == ["VERDICT_ERROR"]
    assert result["aggregate_deltas"]["verdict_accuracy_acceptable"]["delta"] == -0.5
    assert "Same suite content" in result["suite_note"]
    assert "KO-01" in render_comparison(result)


def test_human_review_round_trip(offline_run: Path, tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    shutil.copy(offline_run / "results.json", results)
    sheet = export_template(results, tmp_path / "review.csv")
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    assert "expected_verdict" not in rows[0]  # blind review
    for row in rows:
        row.update({"logical_coherence_1to5": "5", "verdict_justification_1to5": "4", "metric_relevance_1to5": "5",
                    "numbers_traceable_yes_no": "yes", "reviewer": "analyst-1"})
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = import_scores(results, sheet)
    assert summary["imported_cases"] == 2
    assert summary["agreement_with_deterministic_rubric"]["logical_coherence"]["n"] == 2
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert stored["cases"][0]["human"]["numbers_traceable"] is True

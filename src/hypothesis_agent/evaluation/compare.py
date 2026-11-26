"""Compare two benchmark result files (e.g. two JASPER versions or LLM configurations)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from hypothesis_agent.evaluation.runner import SCORE_KEYS


def compare_results(baseline_path: str | Path, candidate_path: str | Path, *, threshold: float = 0.05) -> Dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if baseline["suite"].get("content_sha256") != candidate["suite"].get("content_sha256"):
        suite_note = "WARNING: suites differ (content hash mismatch); case-level comparisons may not be like-for-like."
    else:
        suite_note = "Same suite content."
    agg_b, agg_c = baseline["aggregate"], candidate["aggregate"]
    deltas: Dict[str, Any] = {}
    for key in ["verdict_accuracy_acceptable", "verdict_accuracy_exact"]:
        deltas[key] = _delta(agg_b.get(key), agg_c.get(key))
    for key in SCORE_KEYS:
        deltas[key] = _delta(agg_b["mean_scores"].get(key), agg_c["mean_scores"].get(key))
    for key in ["numerical_discrepancies", "unsupported_claims", "errors"]:
        deltas[key] = _delta(agg_b.get(key), agg_c.get(key))

    by_id_b = {c["case_id"]: c for c in baseline["cases"]}
    changes: List[Dict[str, Any]] = []
    for case in candidate["cases"]:
        previous = by_id_b.get(case["case_id"])
        if previous is None:
            changes.append({"case_id": case["case_id"], "change": "new case"})
            continue
        entry: Dict[str, Any] = {"case_id": case["case_id"]}
        if bool(previous.get("verdict_correct")) != bool(case.get("verdict_correct")):
            entry["verdict"] = f"{previous.get('verdict')} -> {case.get('verdict')} ({'fixed' if case.get('verdict_correct') else 'regressed'})"
        for key in SCORE_KEYS:
            d = _delta(previous.get("scores", {}).get(key), case.get("scores", {}).get(key))
            if d.get("delta") is not None and abs(d["delta"]) >= threshold:
                entry[key] = d
        added = sorted(set(case.get("failure_categories", [])) - set(previous.get("failure_categories", [])))
        removed = sorted(set(previous.get("failure_categories", [])) - set(case.get("failure_categories", [])))
        if added:
            entry["new_failures"] = added
        if removed:
            entry["resolved_failures"] = removed
        if len(entry) > 1:
            changes.append(entry)
    return {
        "baseline": {"label": baseline["label"], "commit": baseline["jasper"].get("git_commit"), "llm": baseline["configuration"]["llm_mode"]},
        "candidate": {"label": candidate["label"], "commit": candidate["jasper"].get("git_commit"), "llm": candidate["configuration"]["llm_mode"]},
        "suite_note": suite_note,
        "aggregate_deltas": deltas,
        "case_changes": changes,
    }


def _delta(a: Any, b: Any) -> Dict[str, Any]:
    if a is None or b is None:
        return {"baseline": a, "candidate": b, "delta": None}
    return {"baseline": a, "candidate": b, "delta": round(b - a, 4)}


def render_comparison(result: Dict[str, Any]) -> str:
    lines = [
        f"# Comparison: {result['baseline']['label']} -> {result['candidate']['label']}",
        "",
        f"- Baseline: commit `{result['baseline']['commit']}`, LLM {result['baseline']['llm']}",
        f"- Candidate: commit `{result['candidate']['commit']}`, LLM {result['candidate']['llm']}",
        f"- {result['suite_note']}",
        "",
        "| Measure | Baseline | Candidate | Delta |",
        "|---|---|---|---|",
    ]
    for key, d in result["aggregate_deltas"].items():
        lines.append(f"| {key} | {d['baseline']} | {d['candidate']} | {d['delta']} |")
    lines += ["", "## Case-level changes", ""]
    if not result["case_changes"]:
        lines.append("No case-level changes above threshold.")
    for change in result["case_changes"]:
        details = "; ".join(f"{k}: {v}" for k, v in change.items() if k != "case_id")
        lines.append(f"- **{change['case_id']}**: {details}")
    return "\n".join(lines) + "\n"

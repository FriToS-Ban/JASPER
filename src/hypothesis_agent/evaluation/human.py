"""Blind human evaluation: export a rubric sheet, import scores, measure agreement with automation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List

from hypothesis_agent.evaluation.judge import RUBRICS

HUMAN_COLUMNS = [f"{dimension}_1to5" for dimension in RUBRICS] + ["numbers_traceable_yes_no", "reviewer", "notes"]


def export_template(results_path: str | Path, out_csv: str | Path) -> Path:
    """Write one row per case. The expected verdict is deliberately omitted so review stays blind."""

    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out = Path(out_csv)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "ticker", "hypothesis", "system_verdict", "report_pdf"] + HUMAN_COLUMNS)
        for case in payload["cases"]:
            writer.writerow([case["case_id"], case["ticker"], case["hypothesis"], case.get("verdict"), case.get("report_pdf")]
                            + [""] * len(HUMAN_COLUMNS))
    rubric_path = out.with_suffix(".rubric.md")
    lines = ["# Rubric for human reviewers", ""]
    for dimension, anchors in RUBRICS.items():
        lines += [f"## {dimension}", ""] + [f"- **{score}**: {text}" for score, text in sorted(anchors.items(), reverse=True)] + [""]
    rubric_path.write_text("\n".join(lines), encoding="utf-8")
    return out


def import_scores(results_path: str | Path, scores_csv: str | Path) -> Dict[str, Any]:
    """Attach human scores to results.json and report agreement with the deterministic rubric."""

    path = Path(results_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in payload["cases"]}
    imported = 0
    with Path(scores_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = by_id.get(row.get("case_id", ""))
            if case is None:
                continue
            human: Dict[str, Any] = {"reviewer": row.get("reviewer") or None, "notes": row.get("notes") or None}
            for dimension in RUBRICS:
                raw = (row.get(f"{dimension}_1to5") or "").strip()
                if raw:
                    value = int(raw)
                    if not 1 <= value <= 5:
                        raise ValueError(f"{case['case_id']}: {dimension} score {value} outside 1-5")
                    human[dimension] = value
            traceable = (row.get("numbers_traceable_yes_no") or "").strip().lower()
            if traceable:
                human["numbers_traceable"] = traceable in {"y", "yes", "true", "1"}
            case["human"] = human
            imported += 1
    agreement: Dict[str, Any] = {}
    for dimension in RUBRICS:
        diffs: List[float] = []
        for case in payload["cases"]:
            human_score = (case.get("human") or {}).get(dimension)
            auto = ((case.get("rubric") or {}).get("deterministic-rubric") or {}).get("scores", {}).get(dimension, {}).get("score")
            if human_score is not None and auto is not None:
                diffs.append(abs(human_score - auto))
        agreement[dimension] = {"n": len(diffs), "mean_abs_diff": round(fmean(diffs), 3) if diffs else None,
                                "exact_agreement": round(sum(d == 0 for d in diffs) / len(diffs), 3) if diffs else None}
    payload["human_evaluation"] = {"imported_cases": imported, "agreement_with_deterministic_rubric": agreement}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload["human_evaluation"]

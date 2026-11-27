"""Replay the LLM-critic escalation rule over recorded benchmark runs, without new LLM calls.

Usage::

    PYTHONPATH=src python scripts/replay_critic_escalation.py eval_results/<label>/runs [...]

For each run it reads the final critique (``critique_iteration_<n>.json``) and the assessments
(``research_state.json`` when present, else ``analysis_metrics.json``) and reports whether the
recorded decision would change under :func:`escalation_metrics`. Severity only matters at the
last branch of ``decide()`` (after every data-collection and iteration-limit branch), so a run
whose final critique escalated on LLM issues alone would have continued to the narrative instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from hypothesis_agent.research.critique import CritiqueIssue, escalation_metrics
from hypothesis_agent.research.metrics import is_known_metric


def _assessments(run_dir: Path) -> List[SimpleNamespace]:
    state_path = run_dir / "research_state.json"
    if state_path.exists():
        rows = json.loads(state_path.read_text(encoding="utf-8")).get("assessments", [])
        return [SimpleNamespace(name=a["expectation_id"], metric=a.get("metric"), status=a["status"]) for a in rows]
    steps = json.loads((run_dir / "analysis_metrics.json").read_text(encoding="utf-8"))["steps"]
    return [SimpleNamespace(name=s["name"], metric=s.get("metric"), status=s["status"]) for s in steps]


def replay(run_dir: Path) -> Optional[Dict[str, Any]]:
    critiques = sorted(run_dir.glob("critique_iteration_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not critiques:
        return None
    final = json.loads(critiques[-1].read_text(encoding="utf-8"))
    issues = [CritiqueIssue.model_validate(i) for i in final["issues"]]
    deterministic = [i for i in issues if i.source == "deterministic"]
    assessments = _assessments(run_dir)
    escalable = escalation_metrics(assessments, deterministic)
    llm_high = [i for i in issues if i.source == "llm" and i.severity == "high"]
    kept = [i for i in llm_high if {m for m in i.metrics if is_known_metric(m)} & escalable]
    conflicts = [i for i in issues if i.type == "SOURCE_CONFLICT" and i.severity == "high"]
    recorded = final["status"]
    llm_only_escalation = recorded == "HUMAN_REVIEW" and final["rationale"].startswith("High-impact ambiguity") and not conflicts
    return {
        "run": run_dir.name,
        "iteration": final["iteration"],
        "recorded": recorded,
        "replayed": "CONTINUE" if llm_only_escalation and not kept else recorded,
        "assessments": [(a.name, a.metric, a.status) for a in assessments],
        "escalable_metrics": sorted(escalable),
        "llm_high": [{"issue_id": i.issue_id, "metrics": i.metrics, "description": i.description,
                      "keeps_high": i in kept} for i in llm_high],
    }


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    changed = escalated = 0
    for root in argv:
        for run_dir in sorted(Path(root).glob("bench-*")):
            result = replay(run_dir)
            if result is None:
                continue
            escalated += result["recorded"] == "HUMAN_REVIEW"
            changed += result["recorded"] != result["replayed"]
            print(f"{result['run']}: iteration {result['iteration']} {result['recorded']} -> {result['replayed']}")
            print(f"  assessments: {result['assessments']}")
            print(f"  escalable metrics: {result['escalable_metrics']}")
            for item in result["llm_high"]:
                verdict = "keeps high" if item["keeps_high"] else "lowered"
                print(f"  {item['issue_id']} {item['metrics']} ({verdict}): {item['description'][:200]}")
    print(f"\nRecorded HUMAN_REVIEW decisions: {escalated}; changed by the replay: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

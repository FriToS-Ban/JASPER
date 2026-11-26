"""Independent reproducibility check for a completed research run.

Given a run directory, this re-hashes every stored raw payload, re-reads every raw evidence value
through its locator, and recomputes every derived value from its inputs. It uses only the files on
disk, so a reviewer can confirm the report's numbers without trusting the running system.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from hypothesis_agent.research.analytics import AnalyticsError, recompute
from hypothesis_agent.research.evidence import EvidenceRegistry, sha256_of
from hypothesis_agent.research.extraction import read_locator


def verify_run_directory(run_dir: str | Path, *, rel_tol: float = 1e-9) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    registry_path = run_dir / "evidence_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"No evidence_registry.json in {run_dir}")
    registry = EvidenceRegistry.from_dict(json.loads(registry_path.read_text(encoding="utf-8")))
    payloads: Dict[str, Any] = {}
    problems: List[Dict[str, Any]] = []

    for snapshot in registry.snapshots.values():
        if snapshot.status != "ok":
            continue
        local = run_dir / Path(snapshot.artifact_path or "").name
        path = local if local.exists() else Path(snapshot.artifact_path or "")
        if not path.exists():
            problems.append({"kind": "snapshot_missing", "snapshot_id": snapshot.snapshot_id, "path": str(path)})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[snapshot.snapshot_id] = payload
        if sha256_of(payload) != snapshot.content_sha256:
            problems.append({"kind": "hash_mismatch", "snapshot_id": snapshot.snapshot_id})

    raw_checked = derived_checked = 0
    for record in registry.records.values():
        if record.kind == "raw":
            payload = payloads.get(record.snapshot_id or "")
            if payload is None:
                problems.append({"kind": "raw_unreadable", "evidence_id": record.evidence_id})
                continue
            value = read_locator(payload, record.locator)
            raw_checked += 1
            if value is None or not math.isclose(value, record.value, rel_tol=rel_tol, abs_tol=1e-12):
                problems.append({"kind": "raw_mismatch", "evidence_id": record.evidence_id, "stored": record.value, "reread": value})
        else:
            try:
                fresh = recompute(registry, record, lambda sid: payloads[sid])
            except (AnalyticsError, KeyError, ZeroDivisionError) as exc:
                problems.append({"kind": "derived_error", "evidence_id": record.evidence_id, "error": str(exc)})
                continue
            derived_checked += 1
            if not math.isclose(fresh, record.value, rel_tol=rel_tol, abs_tol=1e-12):
                problems.append({"kind": "derived_mismatch", "evidence_id": record.evidence_id, "stored": record.value, "recomputed": fresh})

    return {
        "run_dir": str(run_dir),
        "snapshots_checked": len(payloads),
        "raw_records_checked": raw_checked,
        "derived_records_checked": derived_checked,
        "problems": problems,
        "reproducible": not problems,
    }

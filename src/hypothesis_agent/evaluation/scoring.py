"""Per-case scoring. Deterministic wherever possible, and independent of the run's own verdicts.

Independence: scores are computed from the files the run left on disk. The evidence registry is
reloaded and checked for reproducibility, the verdict is recomputed from the recorded
assessments, and every narrative claim is re-verified from scratch. Where the run's own
verification disagrees with the independent re-verification, the disagreement is reported.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional

from hypothesis_agent.evaluation.benchmark import BenchmarkCase
from hypothesis_agent.research.agent import narrative_sections
from hypothesis_agent.research.assessment import ExpectationAssessment, determine_verdict
from hypothesis_agent.research.claims import verify_text
from hypothesis_agent.research.evidence import EvidenceRegistry
from hypothesis_agent.research.metrics import metric_category
from hypothesis_agent.research.reproduce import verify_run_directory

ORDINAL = {"NOT_SUPPORTED": 0, "PARTIALLY_SUPPORTED": 1, "SUPPORTED": 2}
FAILURE_THRESHOLDS = {"metric_relevance": 0.8, "caveat_recall": 0.5}


def verdict_score(actual: Optional[str], case: BenchmarkCase) -> float:
    if actual is None:
        return 0.0
    if actual == case.expected_verdict:
        return 1.0
    if actual in case.acceptable_verdicts:
        return 0.75
    if actual in ORDINAL and case.expected_verdict in ORDINAL and abs(ORDINAL[actual] - ORDINAL[case.expected_verdict]) == 1:
        return 0.25
    return 0.0


def _audit_text(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    uncertainty = state.get("uncertainty") or {}
    parts += uncertainty.get("analytical_limitations", [])
    parts += (uncertainty.get("source_quality") or {}).get("limitations", [])
    for assessment in state.get("assessments", []):
        parts.append(assessment.get("summary", ""))
        parts += assessment.get("caveats", [])
    for critique in state.get("critiques", []):
        parts.append(critique.get("rationale", ""))
        parts += [i.get("description", "") for i in critique.get("issues", [])]
    plan = state.get("plan") or {}
    parts += [r.get("reason", "") for r in plan.get("revisions", [])]
    parts += [m.get("reason", "") for m in (state.get("registry") or {}).get("missing", [])]
    for narrative in state.get("narratives", [])[-1:]:
        body = narrative.get("narrative") or {}
        parts += [str(body.get("narrative", "")), str(body.get("executive_summary", ""))]
        parts += [str(x) for x in body.get("limitations", []) or []]
    return " ".join(str(p) for p in parts).lower()


def score_case(case: BenchmarkCase, context: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    state = context.get("research") or {}
    ticker = case.ticker.upper()
    reproduction = verify_run_directory(run_dir)
    registry = EvidenceRegistry.from_dict(json.loads((run_dir / "evidence_registry.json").read_text(encoding="utf-8")))

    assessments = [ExpectationAssessment.model_validate(a) for a in state.get("assessments", [])]
    recomputed = determine_verdict(assessments)
    reported_verdict = (state.get("verdict") or {}).get("verdict")

    # Independent claim re-verification from disk.
    narratives = state.get("narratives", [])
    sections = narrative_sections(narratives[-1]) if narratives else {}

    def _loader(snapshot_id: str) -> Any:
        snapshot = registry.snapshots[snapshot_id]
        local = run_dir / Path(snapshot.artifact_path or "").name
        return json.loads(local.read_text(encoding="utf-8"))

    independent = verify_text(sections, registry, ticker, verdict=recomputed.verdict, loader=_loader)
    run_claims = {c["claim_id"]: c["status"] for c in (state.get("claims") or {}).get("claims", [])}
    disagreements = [
        {"claim_id": c.claim_id, "run_status": run_claims.get(c.claim_id), "independent_status": c.status}
        for c in independent.claims if run_claims.get(c.claim_id) != c.status
    ]
    numeric = [c for c in independent.claims if c.reported_value is not None and c.metric is not None]
    numerical_accuracy = (sum(1 for c in numeric if c.status == "VERIFIED") / len(numeric)) if numeric else None
    numeric_discrepancies = [
        {"claim_id": c.claim_id, "text": c.text, "reported": c.reported_text, "evidence": c.expected_display, "reason": c.reason}
        for c in numeric if c.status == "CONTRADICTED"
    ]
    unsupported = [
        {"claim_id": c.claim_id, "status": c.status, "text": c.text, "reason": c.reason}
        for c in independent.claims if c.status in {"UNSUPPORTED", "CONTRADICTED"}
    ]
    checkable = [c for c in independent.claims if c.status != "NOT_VERIFIABLE"]
    unsupported_rate = (len(unsupported) / len(checkable)) if checkable else 0.0

    # Metric relevance: metrics the verdict and narrative actually rely on.
    used = {a.metric for a in assessments if a.metric}
    used |= {c.metric for c in independent.claims if c.metric}
    expected_categories = set(case.expected_evidence_categories)
    relevant = {m for m in used if metric_category(m) in expected_categories}
    irrelevant = sorted(used - relevant)
    metric_relevance = (len(relevant) / len(used)) if used else None

    # Evidence coverage: expected categories actually investigated in the registry.
    investigated = {metric_category(r.metric) for r in registry.records.values() if r.ticker == ticker}
    covered = expected_categories & investigated
    evidence_coverage = len(covered) / len(expected_categories) if expected_categories else 1.0

    # Logical coherence checks.
    verdict_claims = [c for c in independent.claims if c.category == "verdict"]
    contradicting = [a for a in assessments if a.status in {"CONTRADICTS", "MIXED_NEGATIVE"}]
    narrative = (narratives[-1].get("narrative") or {}) if narratives else {}
    narrative_text = " ".join(str(v) for v in narrative.values()).lower()
    disclosed = all(
        bool(narrative.get("contradicting_points")) or (a.metric or "").replace("_", " ") in narrative_text
        for a in contradicting
    )
    checks = {
        "verdict_recomputes_from_assessments": recomputed.verdict == reported_verdict,
        "narrative_states_verdict": bool(verdict_claims),
        "narrative_verdict_consistent": all(c.status == "VERIFIED" for c in verdict_claims),
        "contradicting_evidence_disclosed": disclosed,
        "no_contradicted_numeric_claims": not numeric_discrepancies,
    }
    logical_coherence = sum(checks.values()) / len(checks)

    audit = _audit_text(state)
    caveat_hits = [
        {"description": cav.description, "covered": any(k.lower() in audit for k in cav.keywords)}
        for cav in case.important_caveats
    ]
    caveat_recall = fmean(1.0 if h["covered"] else 0.0 for h in caveat_hits) if caveat_hits else None

    vscore = verdict_score(reported_verdict, case)
    failures: List[str] = []
    if reported_verdict not in case.acceptable_verdicts:
        failures.append("VERDICT_ERROR")
    if numeric_discrepancies:
        failures.append("NUMERICAL_DISCREPANCY")
    if any(u["status"] == "UNSUPPORTED" for u in unsupported):
        failures.append("UNSUPPORTED_CLAIMS")
    if evidence_coverage < 1.0:
        failures.append("MISSING_EVIDENCE_CATEGORY")
    if metric_relevance is not None and metric_relevance < FAILURE_THRESHOLDS["metric_relevance"]:
        failures.append("IRRELEVANT_METRICS")
    if logical_coherence < 1.0:
        failures.append("INCOHERENT_REASONING")
    if caveat_recall is not None and caveat_recall < FAILURE_THRESHOLDS["caveat_recall"]:
        failures.append("MISSING_CAVEAT")
    if not reproduction["reproducible"]:
        failures.append("NOT_REPRODUCIBLE")
    if disagreements:
        failures.append("VERIFICATION_DISAGREEMENT")

    uncertainty = state.get("uncertainty") or {}
    return {
        "verdict": reported_verdict,
        "verdict_correct": reported_verdict in case.acceptable_verdicts,
        "verdict_exact": reported_verdict == case.expected_verdict,
        "scores": {
            "verdict_accuracy": vscore,
            "numerical_accuracy": numerical_accuracy,
            "metric_relevance": metric_relevance,
            "logical_coherence": logical_coherence,
            "evidence_coverage": evidence_coverage,
            "unsupported_claim_rate": unsupported_rate,
            "caveat_recall": caveat_recall,
        },
        "claim_counts": independent.counts,
        "numeric_claims": len(numeric),
        "numerical_discrepancies": numeric_discrepancies,
        "unsupported_claims": unsupported,
        "irrelevant_metrics": irrelevant,
        "missing_categories": sorted(expected_categories - investigated),
        "coherence_checks": checks,
        "caveats": caveat_hits,
        "verification_disagreements": disagreements,
        "reproducibility": {k: v for k, v in reproduction.items() if k != "run_dir"},
        "failure_categories": failures,
        "signals": {
            "coherence_checks": checks,
            "metric_relevance": metric_relevance,
            "evidence_balance": uncertainty.get("evidence_balance"),
            "verdict": reported_verdict,
            "limitations_count": len(uncertainty.get("analytical_limitations", [])),
        },
    }


def judge_view(case: BenchmarkCase, context: Dict[str, Any]) -> Dict[str, Any]:
    """What a blind judge sees: the hypothesis and the report content, never the expected answer."""

    state = context.get("research") or {}
    narratives = state.get("narratives", [])
    return {
        "hypothesis": case.hypothesis,
        "ticker": case.ticker,
        "verdict": state.get("verdict"),
        "assessments": [{k: a.get(k) for k in ("expectation_id", "metric", "direction", "status", "summary", "caveats")}
                        for a in state.get("assessments", [])],
        "narrative": (narratives[-1].get("narrative") if narratives else None),
        "limitations": (state.get("uncertainty") or {}).get("analytical_limitations", []),
        "unverified_claims": [c for c in (state.get("claims") or {}).get("claims", []) if c["status"] != "VERIFIED"][:10],
    }

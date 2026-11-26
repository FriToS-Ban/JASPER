"""Deterministic, template-based narrator used when no LLM is configured.

This is a real (if plain) implementation of the LLM interface, not a test stub: it writes the
research narrative strictly from the evidence briefing, citing every figure. It makes JASPER
runnable offline and gives the benchmark a reproducible baseline against which LLM-driven runs
can be compared.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from hypothesis_agent.llm import BaseLLM
from hypothesis_agent.models.hypothesis import HypothesisRequest

VERDICT_SENTENCES = {
    "SUPPORTED": "The hypothesis is supported by the collected evidence.",
    "PARTIALLY_SUPPORTED": "The hypothesis is partially supported: some expectations hold and others do not.",
    "NOT_SUPPORTED": "The hypothesis is not supported by the collected evidence.",
    "INSUFFICIENT_EVIDENCE": "There is insufficient evidence to evaluate the hypothesis with the available data.",
}
STATUS_WORDS = {
    "SUPPORTS": "consistent with the hypothesis",
    "CONTRADICTS": "inconsistent with the hypothesis",
    "MIXED_POSITIVE": "consistent over the full window but reversed in the latest period",
    "MIXED_NEGATIVE": "inconsistent over the full window although the latest period moved the expected way",
    "NEUTRAL": "broadly flat, so it neither supports nor contradicts the hypothesis",
    "INSUFFICIENT": "not measurable with the collected data",
}
RISK_TEMPLATES = {
    "CONTRADICTORY_EVIDENCE": "Accounting profitability and cash generation point in different directions.",
    "TIME_PERIOD_SENSITIVITY": "The conclusion is sensitive to the period chosen; the most recent period departs from the longer trend.",
    "UNSUPPORTED_ASSUMPTION": "Part of the thesis rests on assumptions the financial statements cannot confirm.",
    "METRIC_NOT_MEANINGFUL": "Some standard ratios are not meaningful for this company (for example negative equity).",
    "SOURCE_CONFLICT": "Yahoo's trailing figures and fiscal-year statements disagree materially.",
    "DATA_UNAVAILABLE": "Some planned metrics could not be collected from the data source.",
    "INSUFFICIENT_PERIODS": "Too few periods were available to establish a trend.",
}


def _evidence_index(briefing: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["evidence_id"]: item for item in briefing.get("key_evidence", [])}


def _short_period(period: str) -> str:
    return period.split(" (")[0]


def _cite(item: Dict[str, Any]) -> str:
    return f"{item['display']} in {_short_period(item['period'])} [{item['evidence_id']}]"


def _assessment_sentence(assessment: Dict[str, Any], evidence: Dict[str, Dict[str, Any]]) -> Optional[str]:
    items = [evidence[e] for e in assessment.get("evidence_ids", []) if e in evidence]
    series = [i for i in items if i.get("metric") == assessment.get("metric")]
    label = series[0]["label"] if series else (items[0]["label"] if items else assessment.get("concept", "The metric"))
    status = STATUS_WORDS.get(assessment.get("status", ""), "")
    direction = assessment.get("direction")
    if assessment.get("metric") is None:
        return f"{assessment['concept'].capitalize()}: {assessment['summary']}"
    if direction in {"increase", "decrease"} and len(series) >= 2:
        first, last = series[0], series[-1]
        return f"{label} moved from {_cite(first)} to {_cite(last)}, which is {status}."
    if direction in {"above", "below", "positive", "negative"} and series:
        return f"{label} was {_cite(series[-1])}, which is {status} (threshold set by the hypothesis wording)."
    if direction in {"overvalued", "undervalued"} and items:
        readings = " and ".join(f"{i['label']} of {i['display']} [{i['evidence_id']}]" for i in items)
        return f"Valuation relative to growth: {readings}; this is {status}."
    if direction == "base_effect" and items:
        revenue = [i for i in items if i.get("metric") == "total_revenue"]
        path = ", ".join(_cite(i) for i in revenue)
        return f"Total revenue was {path}; the base-effect test is {status}."
    return None


class DeterministicResearchLLM(BaseLLM):
    """Evidence-only narrator implementing the full LLM interface without network access."""

    name = "deterministic-template-narrator"

    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:
        return ["Collect company info, annual statements and price history for the named ticker."]

    def generate_analysis_plan(self, request: HypothesisRequest, data_overview: Dict[str, Any]) -> List[str]:
        return ["Compute planned ratios, growth rates and trends; assess each expectation."]

    def generate_detailed_analysis(self, request: HypothesisRequest, metrics_overview: Dict[str, Any]) -> str:
        return self.generate_research_narrative(request, metrics_overview)["narrative"]

    def generate_analysis_code(self, **_: Any) -> str:
        return "print('RESULT::' + json.dumps({'steps': [], 'aggregated': {}, 'insights': [], 'artifacts': []}))"

    def generate_research_narrative(self, request: HypothesisRequest, briefing: Dict[str, Any]) -> Dict[str, Any]:
        evidence = _evidence_index(briefing)
        verdict = (briefing.get("verdict") or {}).get("verdict", "INSUFFICIENT_EVIDENCE")
        assessments = briefing.get("assessments", [])
        supporting, contradicting, other = [], [], []
        for assessment in assessments:
            sentence = _assessment_sentence(assessment, evidence)
            if not sentence:
                continue
            status = assessment.get("status")
            if status == "SUPPORTS":
                supporting.append(sentence)
            elif status in {"CONTRADICTS", "MIXED_NEGATIVE"}:
                contradicting.append(sentence)
            else:
                other.append(sentence)
        counts = {k: sum(1 for a in assessments if a.get("status") == k) for k in ("SUPPORTS", "CONTRADICTS")}
        executive = (f"{VERDICT_SENTENCES[verdict]} Of {len(assessments)} tested expectation(s), {counts['SUPPORTS']} "
                     f"were consistent with the evidence and {counts['CONTRADICTS']} were contradicted.")
        paragraphs = [" ".join(supporting + other + contradicting)] if (supporting or other or contradicting) else []
        decision = (briefing.get("decision") or {}).get("rationale")
        if briefing.get("plan_revisions"):
            paragraphs.append(f"The research plan was revised {len(briefing['plan_revisions'])} time(s) after critique.")
        if decision:
            paragraphs.append(f"Final research decision: {decision}")
        limitations = []
        for assessment in assessments:
            limitations.extend(assessment.get("caveats", []))
        return {
            "executive_summary": executive,
            "narrative": "\n\n".join(paragraphs) or VERDICT_SENTENCES[verdict],
            "supporting_points": supporting,
            "contradicting_points": contradicting,
            "limitations": list(dict.fromkeys(limitations))[:6],
        }

    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
        analysis_summary: str,
        artifact_paths: List[str],
    ) -> Dict[str, Any]:
        narrative = self.generate_research_narrative(request, metrics_overview)
        risk_types: List[str] = []
        for issue_type in metrics_overview.get("critique_issue_types", []):
            template = RISK_TEMPLATES.get(issue_type)
            if template and template not in risk_types:
                risk_types.append(template)
        missing = metrics_overview.get("missing", [])
        next_steps = ["Review the unverified claims section before relying on any figure."]
        if missing:
            next_steps.append("Source the unavailable metrics listed under data completeness from company filings.")
        return {
            "executive_summary": narrative["executive_summary"],
            "key_findings": narrative["supporting_points"] + narrative["contradicting_points"],
            "risks": risk_types or ["No additional risks were identified by the deterministic critique."],
            "next_steps": next_steps,
        }

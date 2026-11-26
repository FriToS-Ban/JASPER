"""Rubric-based judges for the subjective evaluation dimensions.

Two judges apply the *same* written rubric:

* :class:`DeterministicRubricJudge` maps measured signals (coherence checks, metric categories)
  onto the rubric anchors. It is reproducible and cannot be talked into a higher score.
* :class:`LLMRubricJudge` asks a separate LLM call to grade the run blind (it never sees the
  benchmark's expected verdict). It provides an independent read that is not the generator
  judging itself, and is reported alongside, never instead of, deterministic scores.

Human reviewers use the same anchors via the CSV exported by :mod:`hypothesis_agent.evaluation.human`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from hypothesis_agent.llm import LLMError, NvidiaLLM

RUBRICS: Dict[str, Dict[int, str]] = {
    "logical_coherence": {
        5: "Every conclusion follows from the cited evidence; contradicting evidence is named and weighed; the stated verdict matches the evidence balance.",
        4: "Reasoning follows from the evidence with a minor gap (e.g. a caveat not discussed).",
        3: "Some conclusions are asserted without evidence, or contradicting evidence is mentioned but not weighed.",
        2: "Key reasoning steps contradict the evidence or the stated verdict differs from what the evidence implies.",
        1: "Narrative is disconnected from the evidence.",
    },
    "verdict_justification": {
        5: "The verdict is the natural reading of the presented evidence, with its limits stated.",
        4: "The verdict is defensible, but a reasonable analyst could weigh one item differently.",
        3: "The verdict is plausible but under-justified (important evidence or caveats missing).",
        2: "The verdict is hard to reconcile with the presented evidence.",
        1: "The verdict contradicts the presented evidence.",
    },
    "metric_relevance": {
        5: "Every metric used directly tests a component of the hypothesis.",
        4: "Metrics are relevant with minor padding.",
        3: "About half of the metrics used address the hypothesis.",
        2: "Mostly tangential metrics.",
        1: "Metrics do not address the hypothesis.",
    },
}


class RubricScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class RubricResult(BaseModel):
    judge: str
    scores: Dict[str, RubricScore]
    error: Optional[str] = None


class DeterministicRubricJudge:
    name = "deterministic-rubric"

    def score(self, signals: Dict[str, Any]) -> RubricResult:
        checks: Dict[str, bool] = signals["coherence_checks"]
        if not checks.get("verdict_recomputes_from_assessments", True):
            coherence = RubricScore(score=1, justification="Reported verdict does not follow from the recorded assessments.")
        elif not checks.get("narrative_verdict_consistent", True):
            coherence = RubricScore(score=2, justification="Narrative states a verdict different from the evidence-based verdict.")
        elif not checks.get("contradicting_evidence_disclosed", True) or not checks.get("no_contradicted_numeric_claims", True):
            coherence = RubricScore(score=3, justification="Contradicting evidence not disclosed, or narrative contains contradicted figures.")
        elif not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            coherence = RubricScore(score=4, justification=f"Minor gap: {', '.join(failed)}.")
        else:
            coherence = RubricScore(score=5, justification="All coherence checks passed.")

        relevance_value = signals.get("metric_relevance")
        if relevance_value is None:
            relevance = RubricScore(score=3, justification="No metrics were used, relevance cannot be shown.")
        else:
            band = 5 if relevance_value >= 0.95 else 4 if relevance_value >= 0.8 else 3 if relevance_value >= 0.6 else 2 if relevance_value >= 0.4 else 1
            relevance = RubricScore(score=band, justification=f"{relevance_value:.0%} of used metrics fall in the case's expected evidence categories.")

        balance = signals.get("evidence_balance") or {}
        verdict = signals.get("verdict")
        supporting, contradicting = len(balance.get("supporting", [])), len(balance.get("contradicting", []))
        consistent = (
            (verdict == "SUPPORTED" and contradicting == 0)
            or (verdict == "NOT_SUPPORTED" and supporting == 0)
            or verdict in {"PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"}
        )
        caveats_stated = bool(signals.get("limitations_count"))
        if not consistent:
            justification = RubricScore(score=2, justification="Verdict sits awkwardly with the supporting/contradicting balance.")
        elif caveats_stated:
            justification = RubricScore(score=5, justification="Verdict follows from the evidence balance and limitations are stated.")
        else:
            justification = RubricScore(score=4, justification="Verdict follows from the evidence balance; limitations not stated.")
        return RubricResult(judge=self.name, scores={
            "logical_coherence": coherence, "verdict_justification": justification, "metric_relevance": relevance,
        })


class LLMRubricJudge:
    """Blind LLM grader using the same rubric (requires NVIDIA_API_KEY)."""

    def __init__(self, llm: NvidiaLLM) -> None:
        self.llm = llm
        self.name = f"llm-rubric:{llm.model}"

    def score(self, view: Dict[str, Any]) -> RubricResult:
        system_prompt = (
            "You are an independent reviewer grading an equity research report against a fixed rubric. Grade only "
            "what is presented; you do not know the correct answer. For each dimension return an integer 1-5 and a "
            "one-sentence justification. Respond with JSON: {\"logical_coherence\": {\"score\": int, \"justification\": str}, "
            "\"verdict_justification\": {...}, \"metric_relevance\": {...}}."
        )
        user_prompt = json.dumps({"rubrics": RUBRICS, "report": view}, ensure_ascii=False, default=str)
        try:
            content = self.llm._chat(system_prompt, user_prompt, operation="rubric_judge", json_mode=True)
            payload = self.llm._parse_json(content)
            scores = {k: RubricScore.model_validate(payload[k]) for k in RUBRICS}
        except (LLMError, ValidationError, KeyError, TypeError) as exc:
            return RubricResult(judge=self.name, scores={}, error=f"{type(exc).__name__}: {exc}")
        return RubricResult(judge=self.name, scores=scores)

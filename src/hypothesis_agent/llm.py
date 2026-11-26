"""LLM integration helpers for the JASPER hypothesis workflow."""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import BadRequestError, OpenAI
logger = logging.getLogger(__name__)

from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.metrics import LLMTelemetry, PIPELINE_TELEMETRY


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an unexpected payload."""


class BaseLLM(ABC):
    """Interface defining the planner / report generation contract."""

    @abstractmethod
    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:  # pragma: no cover - interface
        """Return an ordered list of data collection tasks."""

    @abstractmethod
    def generate_analysis_plan(
        self,
        request: HypothesisRequest,
        data_overview: Dict[str, Any],
    ) -> List[str]:  # pragma: no cover - interface
        """Return an ordered list of analysis steps given the available data."""

    @abstractmethod
    def generate_detailed_analysis(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
    ) -> str:  # pragma: no cover - interface
        """Return a narrative describing hypothesis validation results."""

    @abstractmethod
    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
        analysis_summary: str,
        artifact_paths: List[str],
    ) -> Dict[str, Any]:  # pragma: no cover - interface
        """Return a structured report payload summarising the findings."""

    @abstractmethod
    def generate_analysis_code(
        self,
        *,
        request: HypothesisRequest,
        analysis_plan: List[Dict[str, Any]],
        data_artifacts: Dict[str, str],
        data_format: Dict[str, str],
        attempt: int,
        history: List[Dict[str, str]],
    ) -> str:  # pragma: no cover - interface
        """Return Python source for executing the requested analysis."""

    # ------------------------------------------------------------------
    # Research-agent capabilities. Defaults keep older adapters working: an adapter that does not
    # implement them simply contributes nothing, and the deterministic agent still runs.
    # ------------------------------------------------------------------
    def propose_research_plan(
        self, request: HypothesisRequest, catalogue: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Optionally propose expectations/metrics. Output is validated before use."""

        return None

    def choose_next_analysis_action(
        self, request: HypothesisRequest, briefing: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Optionally pick the next analytical tool call ({"tool", "arguments", "rationale"})."""

        return None

    def critique_analysis(
        self, request: HypothesisRequest, briefing: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Optionally add critique issues / metric requests. Cannot override deterministic checks."""

        return None

    def generate_research_narrative(
        self, request: HypothesisRequest, briefing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return {"narrative": str, ...}. Defaults to the legacy free-text narrative."""

        return {"narrative": self.generate_detailed_analysis(request, briefing)}


@dataclass(slots=True)
class NvidiaLLM(BaseLLM):
    """LLM adapter backed by NVIDIA NIM (OpenAI-compatible)"""

    api_key: str
    model: str
    temperature: float | None = 0.2
    telemetry: LLMTelemetry | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise LLMError("NVIDIA API key is not configured")
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
            timeout=180.0,
            max_retries=2,
        )
        if self.telemetry is None:
            self.telemetry = PIPELINE_TELEMETRY

    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:
        system_prompt = (
            "You are a financial research operations planner. Generate a concise list of data collection "
            "tasks needed to validate an investment hypothesis. Respond with a JSON array of strings."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nEntities: {entities}\nTime horizon: {start} to {end}\n"
            "Include market, fundamental, macro, and narrative data collection tasks.".format(
                hypothesis=request.hypothesis_text,
                entities=", ".join(request.entities) or "(none specified)",
                start=request.time_horizon.start.isoformat(),
                end=request.time_horizon.end.isoformat(),
            )
        )
        content = self._chat(system_prompt, user_prompt, operation="generate_data_plan", json_mode=True)
        return self._parse_list(content)

    def generate_analysis_plan(
        self,
        request: HypothesisRequest,
        data_overview: Dict[str, Any],
    ) -> List[str]:
        system_prompt = (
            "You are a senior quantitative analyst. Based on the available data inventory, produce an ordered list "
            "of analytical steps to validate the hypothesis. Respond with a JSON array of strings."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nRisk appetite: {risk}\nData overview: {overview}".format(
                hypothesis=request.hypothesis_text,
                risk=request.risk_appetite.value,
                overview=json.dumps(data_overview, ensure_ascii=False),
            )
        )
        content = self._chat(system_prompt, user_prompt, operation="generate_analysis_plan", json_mode=True)
        return self._parse_list(content)

    def generate_detailed_analysis(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
    ) -> str:
        system_prompt = (
            "You are an equity research associate. Write a concise narrative (max 6 sentences) summarising the current "
            "state of the hypothesis given the quantitative metrics and sentiment signals. Avoid repeated figures."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nMetrics overview: {metrics}".format(
                hypothesis=request.hypothesis_text,
                metrics=json.dumps(metrics_overview, ensure_ascii=False),
            )
        )
        return self._chat(system_prompt, user_prompt, operation="generate_detailed_analysis")

    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
        analysis_summary: str,
        artifact_paths: List[str],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are an analyst preparing a client-facing investment memo. Respond with a JSON object containing the "
            "keys 'executive_summary', 'key_findings' (array), 'risks' (array), and 'next_steps' (array). "
            "If the metrics include 'key_evidence', use only those figures, cite each with its evidence_id in square "
            "brackets (e.g. [EV-1a2b3c4d]), state fiscal years, and state the verdict exactly as given. Every figure "
            "you write will be checked automatically against the evidence registry."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nMetrics: {metrics}\nNarrative summary: {summary}\nArtifacts: {artifacts}".format(
                hypothesis=request.hypothesis_text,
                metrics=json.dumps(metrics_overview, ensure_ascii=False),
                summary=analysis_summary,
                artifacts=artifact_paths,
            )
        )
        content = self._chat(system_prompt, user_prompt, operation="generate_report", json_mode=True)
        payload = self._parse_json(content)
        if not isinstance(payload, dict):
            raise LLMError("Report generation returned non-dict response")
        return payload

    def generate_analysis_code(
        self,
        *,
        request: HypothesisRequest,
        analysis_plan: List[Dict[str, Any]],
        data_artifacts: Dict[str, str],
        data_format: Dict[str, str],
        attempt: int,
        history: List[Dict[str, str]],
    ) -> str:
        system_prompt = (
            "You are an autonomous quantitative analyst operating in a constrained Python REPL. "
            "Write straight-line, top-level Python (no function or class definitions) that inspects provided datasets, "
            "computes the requested analytics, and prints the final dictionary using "
            "print(\"RESULT::\" + json.dumps(result, default=str)). Respond only with Python code inside a "
            "fenced block; do not include commentary."
        )
        user_payload = {
            "hypothesis": request.hypothesis_text,
            "attempt": attempt,
            "analysis_plan": analysis_plan,
            "data_artifacts": data_artifacts,
            "data_format": data_format,
            "execution_history": history,
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
        return self._chat(system_prompt, user_prompt, operation="generate_analysis_code")

    def propose_research_plan(self, request: HypothesisRequest, catalogue: Dict[str, Any]) -> Dict[str, Any] | None:
        system_prompt = (
            "You plan quantitative tests of investment hypotheses. Translate the hypothesis into expectations about "
            "metrics from the provided catalogue. Respond with JSON: {\"expectations\": [{\"concept\": str, "
            "\"metric\": catalogue key or null if no catalogue metric can measure it, \"direction\": one of "
            "increase|decrease|above|below|positive|negative|overvalued|undervalued|unmeasurable, \"threshold\": "
            "number or null, \"rationale\": str}], \"additional_metrics\": [catalogue keys]}. Never invent metric keys."
        )
        user_prompt = json.dumps({
            "hypothesis": request.hypothesis_text,
            "entities": request.entities,
            "catalogue": catalogue,
        }, ensure_ascii=False)
        content = self._chat(system_prompt, user_prompt, operation="propose_research_plan", json_mode=True)
        payload = self._parse_json(content)
        return payload if isinstance(payload, dict) else None

    def choose_next_analysis_action(self, request: HypothesisRequest, briefing: Dict[str, Any]) -> Dict[str, Any] | None:
        system_prompt = (
            "You are a quantitative analyst driving a toolbox of deterministic financial tools. Look at the "
            "expectations, the metrics already available and the recent tool calls, then choose ONE next tool call "
            "that would most improve the test of the hypothesis (e.g. a missing ratio, a growth rate, a trend over a "
            "specific window, a correlation). Do not repeat calls already made. If the evidence is sufficient, choose "
            "\"finish\". Respond with JSON: {\"tool\": name or \"finish\", \"arguments\": {...}, \"rationale\": str}."
        )
        user_prompt = json.dumps(briefing, ensure_ascii=False, default=str)
        content = self._chat(system_prompt, user_prompt, operation="choose_next_analysis_action", json_mode=True)
        payload = self._parse_json(content)
        return payload if isinstance(payload, dict) else None

    def critique_analysis(self, request: HypothesisRequest, briefing: Dict[str, Any]) -> Dict[str, Any] | None:
        system_prompt = (
            "You are a sceptical reviewer of an equity analysis. Deterministic checks have already run (listed). "
            "Identify additional problems: missing metrics that matter for this hypothesis, ignored contradictory "
            "evidence, inappropriate time periods, unsupported assumptions. Only reference metric keys from "
            "allowed_metrics. Respond with JSON: {\"issues\": [{\"severity\": low|medium|high, \"description\": str, "
            "\"metrics\": [keys]}], \"additional_metrics\": [keys], \"notes\": str}. Return empty lists if nothing is wrong."
        )
        user_prompt = json.dumps(briefing, ensure_ascii=False, default=str)
        content = self._chat(system_prompt, user_prompt, operation="critique_analysis", json_mode=True)
        payload = self._parse_json(content)
        return payload if isinstance(payload, dict) else None

    def generate_research_narrative(self, request: HypothesisRequest, briefing: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are an equity research associate writing the analysis section of an auditable research report. "
            "Follow the rules in the briefing exactly: use only figures from key_evidence, cite every figure with "
            "its evidence_id in square brackets, state fiscal years, state the verdict exactly as given, and describe "
            "contradicting evidence. If 'corrections' are present, they list claims that failed automated "
            "verification in your previous draft: fix or remove them. Respond with JSON: {\"executive_summary\": str "
            "(2-3 sentences), \"narrative\": str (one to three paragraphs), \"supporting_points\": [str], "
            "\"contradicting_points\": [str], \"limitations\": [str]}."
        )
        user_prompt = json.dumps(briefing, ensure_ascii=False, default=str)
        content = self._chat(system_prompt, user_prompt, operation="generate_research_narrative", json_mode=True)
        payload = self._parse_json(content)
        if not isinstance(payload, dict) or not payload.get("narrative"):
            raise LLMError("Research narrative response missing 'narrative'")
        return payload

    def _chat(self, system_prompt: str, user_prompt: str, *, operation: str, json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            start_time = time.perf_counter()
            response = self._client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - start_time
        except BadRequestError as exc:
            message = str(exc)
            if self.temperature is not None and "temperature" in message.lower():
                logger.warning(
                    "OpenAI model %s rejected temperature=%s; retrying with default",
                    self.model,
                    self.temperature,
                )
                self.temperature = None
                kwargs.pop("temperature", None)
                try:
                    start_time = time.perf_counter()
                    response = self._client.chat.completions.create(**kwargs)
                    latency = time.perf_counter() - start_time
                except Exception as retry_exc:  # pragma: no cover - network failures
                    raise LLMError(f"LLM request failed ({self.model}): {type(retry_exc).__name__}: {retry_exc}") from retry_exc
            else:
                raise LLMError(f"LLM request failed ({self.model}): {type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # pragma: no cover - network failures
            raise LLMError(f"LLM request failed ({self.model}): {type(exc).__name__}: {exc}") from exc

        try:
            message = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
            raise LLMError("Malformed OpenAI response") from exc

        if self.telemetry is not None:
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            self.telemetry.record_llm_call(
                operation=operation,
                model=self.model,
                latency_seconds=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        return message.strip()

    @staticmethod
    def _parse_json(content: str) -> Any:
        # Strip markdown code blocks if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Handle ```json ... ``` or just ``` ... ```
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM response as JSON. Content: %s", content)
            raise LLMError(f"LLM response was not valid JSON: {str(exc)}") from exc

    def _parse_list(self, content: str) -> List[str]:
        parsed = self._parse_json(content)
        if not isinstance(parsed, list) or not parsed:
            raise LLMError("LLM response must be a non-empty list of steps")
        if not all(isinstance(item, str) and item.strip() for item in parsed):
            raise LLMError("LLM response list entries must be non-empty strings")
        return parsed

OpenAILLM = NvidiaLLM
GeminiLLM = NvidiaLLM
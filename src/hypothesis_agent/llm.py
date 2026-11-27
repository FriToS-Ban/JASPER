"""LLM integration for the JASPER hypothesis workflow, built on LangChain chat models.

:class:`NvidiaLLM` talks to NVIDIA NIM's OpenAI-compatible endpoint through
``langchain_openai.ChatOpenAI``. Structured responses are parsed into pydantic models with
``with_structured_output`` and the analysis tool choice uses native function calling
(``bind_tools``). The LLM only proposes: plans, critique issues and tool calls are validated
downstream, every figure comes from the evidence registry, and the verdict is computed
deterministically (``research.assessment.determine_verdict``).
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type
from uuid import UUID

import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, BeforeValidator, Field, ValidationError, field_validator

from hypothesis_agent.metrics import LLMTelemetry, PIPELINE_TELEMETRY
from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.research.critique import LLMCritique
from hypothesis_agent.research.planner import PlanProposal

logger = logging.getLogger(__name__)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 2
StructuredMethod = Literal["json_schema", "function_calling", "json_mode"]


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an unexpected payload."""


# ----------------------------------------------------------------------------------------------
# Structured-output schemas (planner and critique schemas live next to their validators)
# ----------------------------------------------------------------------------------------------
def _as_str_list(value: Any) -> Any:
    """Accept a bare string or null where a list of prose items is expected."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return value


StrList = Annotated[List[str], BeforeValidator(_as_str_list)]


class ResearchNarrative(BaseModel):
    """Analysis section of an auditable research report."""

    executive_summary: str = Field(default="", description="2-3 sentences.")
    narrative: str = Field(description="One to three paragraphs.")
    supporting_points: StrList = Field(default_factory=list)
    contradicting_points: StrList = Field(default_factory=list)
    limitations: StrList = Field(default_factory=list)

    @field_validator("narrative")
    @classmethod
    def _narrative_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("narrative must not be empty")
        return value


class ReportPayload(BaseModel):
    """Client-facing investment memo."""

    executive_summary: str = ""
    key_findings: StrList = Field(default_factory=list)
    risks: StrList = Field(default_factory=list)
    next_steps: StrList = Field(default_factory=list)


class AnalysisAction(BaseModel):
    """One analytical tool call chosen in JSON mode (used when native function calling is unavailable)."""

    tool: str = Field(description="Tool name, or 'finish'.")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class StepList(BaseModel):
    """An ordered list of steps."""

    steps: List[str]

    @field_validator("steps")
    @classmethod
    def _non_empty_steps(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("at least one step is required")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("steps must be non-empty strings")
        return value


# ----------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------
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

    def choose_next_tool_call(
        self, request: HypothesisRequest, briefing: Dict[str, Any], tools: Sequence[Any]
    ) -> Dict[str, Any] | None:
        """Pick the next tool call given LangChain tools (see ``ResearchToolbox.as_langchain_tools``).

        Returns the same action dict as :meth:`choose_next_analysis_action`; the caller executes it
        through the toolbox, which validates and logs it. The default delegates to
        :meth:`choose_next_analysis_action`, so adapters without native tool calling keep working.
        """

        return self.choose_next_analysis_action(request, briefing)

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


# ----------------------------------------------------------------------------------------------
# LangChain plumbing
# ----------------------------------------------------------------------------------------------
class NimChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` for NVIDIA NIM that also keeps NIM's non-standard ``reasoning_content``.

    ``ChatOpenAI`` follows the official OpenAI schema and drops provider extensions. Reasoning
    models on NIM return their reasoning there (with empty ``content`` on tool calls); JASPER uses it
    only as the recorded rationale of a native tool call that came without any message text.
    """

    def _create_chat_result(self, response: Any, generation_info: Optional[Dict[str, Any]] = None) -> Any:
        result = super()._create_chat_result(response, generation_info)
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        for generation, choice in zip(result.generations, choices or []):
            message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
            reasoning = message.get("reasoning_content") if isinstance(message, dict) else getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning.strip() and isinstance(generation.message, AIMessage):
                generation.message.additional_kwargs.setdefault("reasoning_content", reasoning)
        return result


def build_nim_chat_model(
    *, api_key: str, model: str, temperature: float | None = 0.2, http_client: Any = None
) -> ChatOpenAI:
    """Chat model for NIM's OpenAI-compatible API (180 s timeout, 2 SDK retries)."""

    return NimChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=NIM_BASE_URL,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
        use_responses_api=False,
        http_client=http_client,
    )


class _TelemetryCallback(BaseCallbackHandler):
    """Records latency and token usage (from ``usage_metadata``) of each successful chat-model call."""

    def __init__(self, telemetry: LLMTelemetry, *, operation: str, model: str) -> None:
        self.telemetry = telemetry
        self.operation = operation
        self.model = model
        self._started: Dict[UUID, float] = {}

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._started.pop(run_id, None)
        prompt_tokens, completion_tokens, total_tokens = _token_usage(response)
        self.telemetry.record_llm_call(
            operation=self.operation,
            model=self.model,
            latency_seconds=time.perf_counter() - started if started is not None else 0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._started.pop(run_id, None)


def _token_usage(response: LLMResult) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                return usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")
    usage = (response.llm_output or {}).get("token_usage") or {}
    if usage:
        return usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
    return None, None, None


def _without_temperature(chat_model: BaseChatModel) -> BaseChatModel:
    if getattr(chat_model, "temperature", None) is None:
        return chat_model
    return chat_model.model_copy(update={"temperature": None})


def _message_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return message.text or ""
    return str(message or "")


def _extract_json(text: str) -> Any:
    """Tolerant JSON extraction: strips <think> blocks, code fences and surrounding prose."""

    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1).strip()
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.find("{"): cleaned.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _recover(schema: Type[BaseModel], raw: Any, *, lenient: bool) -> Optional[Dict[str, Any]]:
    """Parse a raw message that the structured-output parser rejected into ``schema``."""

    if not isinstance(raw, BaseMessage):
        return None
    candidates: List[Any] = [call.get("args") for call in getattr(raw, "tool_calls", None) or []]
    payload = _extract_json(_message_text(raw))
    if payload is not None:
        candidates.append(payload)
    for candidate in candidates:
        try:
            return schema.model_validate(candidate).model_dump()
        except ValidationError:
            continue
    if lenient:  # the caller validates item by item and records rejections
        return next((c for c in candidates if isinstance(c, dict)), None)
    return None


def _describe(exc: Any) -> str:
    if exc is None:
        return "no parseable output"
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _tail(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    tail = collapsed[-limit:]
    boundary = tail.find(". ")
    return tail[boundary + 2:] if 0 <= boundary < limit // 2 else "..." + tail


def _rationale(message: AIMessage) -> str:
    text = re.sub(r"<think>.*?</think>", "", _message_text(message), flags=re.S).strip()
    if text:
        return text
    reasoning = message.additional_kwargs.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return "Model reasoning: " + _tail(reasoning)
    return ""


def action_from_tool_message(message: Any) -> Dict[str, Any]:
    """Turn a chat model's tool-calling reply into the agent's action dict.

    The first tool call wins (parallel calls are disabled). Arguments are passed through untouched:
    ``ResearchToolbox.invoke`` validates them and logs rejections.
    """

    if not isinstance(message, AIMessage):
        raise LLMError(f"Tool-calling reply was {type(message).__name__}, not an AIMessage")
    rationale = _rationale(message)
    if message.tool_calls:
        call = message.tool_calls[0]
        name = str(call.get("name") or "")
        arguments = dict(call.get("args") or {})
        if name == "finish":
            return {"tool": "finish", "arguments": {}, "rationale": rationale or str(arguments.get("reason") or "")}
        return {"tool": name, "arguments": arguments, "rationale": rationale}
    if message.invalid_tool_calls:
        bad = message.invalid_tool_calls[0]
        raise LLMError(f"Model returned a malformed tool call {bad.get('name')!r}: {bad.get('error')}")
    payload = _extract_json(_message_text(message))
    if isinstance(payload, dict) and payload.get("tool"):
        try:
            return AnalysisAction.model_validate(payload).model_dump()
        except ValidationError as exc:
            raise LLMError(f"Tool choice in message text did not validate: {_describe(exc)}") from exc
    # No tool call at all: the model chose to stop.
    return {"tool": "finish", "arguments": {}, "rationale": rationale}


# ----------------------------------------------------------------------------------------------
# NVIDIA NIM adapter
# ----------------------------------------------------------------------------------------------
@dataclass(slots=True)
class NvidiaLLM(BaseLLM):
    """LLM adapter backed by NVIDIA NIM (OpenAI-compatible) through a LangChain chat model.

    ``chat_model`` may be injected (any LangChain ``BaseChatModel``, e.g. a test double); otherwise a
    :class:`NimChatOpenAI` is built from ``api_key``/``model``. ``structured_method`` is the
    ``with_structured_output`` method tried first; JSON mode is the fallback.
    """

    api_key: str = field(repr=False)
    model: str
    temperature: float | None = 0.2
    telemetry: LLMTelemetry | None = None
    structured_method: StructuredMethod = "json_schema"
    chat_model: Any = None
    http_client: Any = None

    def __post_init__(self) -> None:
        if self.chat_model is None:
            if not self.api_key:
                raise LLMError("NVIDIA API key is not configured")
            self.chat_model = build_nim_chat_model(
                api_key=self.api_key, model=self.model, temperature=self.temperature, http_client=self.http_client,
            )
        if self.telemetry is None:
            self.telemetry = PIPELINE_TELEMETRY

    # ------------------------------------------------------------------
    # Legacy workflow capabilities
    # ------------------------------------------------------------------
    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:
        system_prompt = (
            "You are a financial research operations planner. Generate a concise list of data collection "
            "tasks needed to validate an investment hypothesis. Respond with JSON: {\"steps\": [str]}."
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
        return self._structured(StepList, system_prompt, user_prompt, operation="generate_data_plan")["steps"]

    def generate_analysis_plan(
        self,
        request: HypothesisRequest,
        data_overview: Dict[str, Any],
    ) -> List[str]:
        system_prompt = (
            "You are a senior quantitative analyst. Based on the available data inventory, produce an ordered list "
            "of analytical steps to validate the hypothesis. Respond with JSON: {\"steps\": [str]}."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nRisk appetite: {risk}\nData overview: {overview}".format(
                hypothesis=request.hypothesis_text,
                risk=request.risk_appetite.value,
                overview=json.dumps(data_overview, ensure_ascii=False),
            )
        )
        return self._structured(StepList, system_prompt, user_prompt, operation="generate_analysis_plan")["steps"]

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
        return self._text(system_prompt, user_prompt, operation="generate_detailed_analysis")

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
        return self._structured(ReportPayload, system_prompt, user_prompt, operation="generate_report")

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
        return self._text(system_prompt, user_prompt, operation="generate_analysis_code")

    # ------------------------------------------------------------------
    # Research-agent capabilities
    # ------------------------------------------------------------------
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
        # Lenient: build_plan validates each expectation and records the ones it rejects.
        return self._structured(PlanProposal, system_prompt, user_prompt, operation="propose_research_plan", lenient=True)

    def choose_next_analysis_action(self, request: HypothesisRequest, briefing: Dict[str, Any]) -> Dict[str, Any] | None:
        """JSON-mode tool choice: the fallback when native function calling is unavailable."""

        system_prompt = (
            "You are a quantitative analyst driving a toolbox of deterministic financial tools. Look at the "
            "expectations, the metrics already available and the recent tool calls, then choose ONE next tool call "
            "that would most improve the test of the hypothesis (e.g. a missing ratio, a growth rate, a trend over a "
            "specific window, a correlation). Do not repeat calls already made. If the evidence is sufficient, choose "
            "\"finish\". Respond with JSON: {\"tool\": name or \"finish\", \"arguments\": {...}, \"rationale\": str}."
        )
        user_prompt = json.dumps(briefing, ensure_ascii=False, default=str)
        # JSON mode: the free-form 'arguments' object cannot be expressed as a strict JSON schema.
        return self._structured(AnalysisAction, system_prompt, user_prompt, operation="choose_next_analysis_action",
                                method="json_mode")

    def choose_next_tool_call(
        self, request: HypothesisRequest, briefing: Dict[str, Any], tools: Sequence[Any]
    ) -> Dict[str, Any] | None:
        """Native function calling: the tools are bound to the model and its first tool call is returned."""

        if not tools:
            return self.choose_next_analysis_action(request, briefing)
        system_prompt = (
            "You are a quantitative analyst driving a toolbox of deterministic financial tools. Look at the "
            "expectations, the metrics already available and the recent tool calls, then call ONE tool that would "
            "most improve the test of the hypothesis (e.g. a missing ratio, a growth rate, a trend over a specific "
            "window, a correlation). Do not repeat calls already made. If the evidence is sufficient, call the "
            "'finish' tool. State in one sentence why you chose the call."
        )
        # The bound tool schemas replace the briefing's compact tool descriptions.
        user_prompt = json.dumps({k: v for k, v in briefing.items() if k != "tools"}, ensure_ascii=False, default=str)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        try:
            message = self._run(
                lambda chat: chat.bind_tools(list(tools), tool_choice="required", parallel_tool_calls=False),
                messages, operation="choose_next_analysis_action",
            )
        except (openai.BadRequestError, NotImplementedError) as exc:
            logger.warning("Native function calling unavailable for %s (%s); falling back to a JSON-mode tool choice",
                           self.model, _describe(exc))
            return self.choose_next_analysis_action(request, briefing)
        except Exception as exc:
            raise LLMError(f"LLM request failed ({self.model}): {type(exc).__name__}: {exc}") from exc
        action = action_from_tool_message(message)
        logger.info("LLM native tool call: %s(%s)", action["tool"], json.dumps(action["arguments"], default=str))
        return action

    def critique_analysis(self, request: HypothesisRequest, briefing: Dict[str, Any]) -> Dict[str, Any] | None:
        system_prompt = (
            "You are a sceptical reviewer of an equity analysis. Deterministic checks have already run (listed). "
            "Identify additional problems: missing metrics that matter for this hypothesis, ignored contradictory "
            "evidence, inappropriate time periods, unsupported assumptions. Only reference metric keys from "
            "allowed_metrics. Respond with JSON: {\"issues\": [{\"severity\": low|medium|high, \"description\": str, "
            "\"metrics\": [keys]}], \"additional_metrics\": [keys], \"notes\": str}. Return empty lists if nothing is wrong."
        )
        user_prompt = json.dumps(briefing, ensure_ascii=False, default=str)
        # Lenient: merge_llm_critique validates the payload and records rejections.
        return self._structured(LLMCritique, system_prompt, user_prompt, operation="critique_analysis", lenient=True)

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
        return self._structured(ResearchNarrative, system_prompt, user_prompt, operation="generate_research_narrative")

    # ------------------------------------------------------------------
    # Generic calls
    # ------------------------------------------------------------------
    def structured_output(
        self,
        schema: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str,
        method: Optional[StructuredMethod] = None,
    ) -> Dict[str, Any]:
        """Ask for ``schema`` and return the validated payload as a dict (raises :class:`LLMError`)."""

        return self._structured(schema, system_prompt, user_prompt, operation=operation, method=method)

    def _run(self, build: Callable[[Any], Any], messages: List[BaseMessage], *, operation: str) -> Any:
        """Invoke ``build(chat_model)``; retry once without temperature if the model rejects it."""

        callbacks = [_TelemetryCallback(self.telemetry, operation=operation, model=self.model)] if self.telemetry else []
        config = {"callbacks": callbacks, "run_name": operation}
        try:
            return build(self.chat_model).invoke(messages, config=config)
        except openai.BadRequestError as exc:
            if self.temperature is None or "temperature" not in str(exc).lower():
                raise
            logger.warning("NVIDIA model %s rejected temperature=%s; retrying with default", self.model, self.temperature)
            self.temperature = None
            self.chat_model = _without_temperature(self.chat_model)
            return build(self.chat_model).invoke(messages, config=config)

    def _text(self, system_prompt: str, user_prompt: str, *, operation: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        try:
            message = self._run(lambda chat: chat, messages, operation=operation)
        except Exception as exc:
            raise LLMError(f"LLM request failed ({self.model}): {type(exc).__name__}: {exc}") from exc
        return _message_text(message).strip()

    def _structured(
        self,
        schema: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str,
        method: Optional[StructuredMethod] = None,
        lenient: bool = False,
    ) -> Dict[str, Any]:
        """Structured output with ``method`` (default ``structured_method``), falling back to JSON mode.

        A reply the parser rejects is first re-read from the raw message (tool-call arguments or JSON
        in the text); only if that fails is JSON mode tried with the same prompt. Transport errors are
        not retried here (the SDK already retried them) and raise :class:`LLMError` immediately.
        """

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        primary = method or self.structured_method
        attempts: List[str] = [primary] if primary == "json_mode" else [primary, "json_mode"]
        failures: List[str] = []
        for attempt in attempts:
            try:
                out = self._run(
                    lambda chat, m=attempt: chat.with_structured_output(schema, method=m, include_raw=True),
                    messages, operation=operation,
                )
            # 400/422: e.g. this response_format is not supported for the model; try the next method.
            except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
                failures.append(f"{attempt}: {_describe(exc)}")
            except openai.APIError as exc:
                raise LLMError(f"LLM request failed ({self.model}): {type(exc).__name__}: {exc}") from exc
            # The reply could not be parsed (parser or SDK-side checks such as a truncated reply).
            # Other exceptions are bugs and propagate.
            except (OutputParserException, ValidationError, ValueError, openai.OpenAIError) as exc:
                failures.append(f"{attempt}: {_describe(exc)}")
            else:
                parsed = out.get("parsed") if isinstance(out, dict) else None
                if isinstance(parsed, schema):
                    return parsed.model_dump()
                recovered = _recover(schema, out.get("raw") if isinstance(out, dict) else None, lenient=lenient)
                if recovered is not None:
                    return recovered
                failures.append(f"{attempt}: {_describe(out.get('parsing_error') if isinstance(out, dict) else None)}")
            if attempt != attempts[-1]:
                logger.warning("%s: %s structured output failed (%s); retrying in JSON mode", operation, attempt, failures[-1])
        raise LLMError(f"{operation}: no valid {schema.__name__} from {self.model} ({'; '.join(failures)})")


OpenAILLM = NvidiaLLM
GeminiLLM = NvidiaLLM

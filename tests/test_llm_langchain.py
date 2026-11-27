"""The LangChain LLM layer: structured output, native tool calling, telemetry and fallbacks.

Everything runs offline: a small fake ``BaseChatModel`` stands in for the chat model, and one test
drives the real ``ChatOpenAI`` stack against a mocked NIM HTTP endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from hypothesis_agent.config import AppSettings
from hypothesis_agent.evaluation.judge import LLMRubricJudge
from hypothesis_agent.llm import NIM_BASE_URL, LLMError, NvidiaLLM, action_from_tool_message
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.research.evidence import EvidenceRegistry
from hypothesis_agent.research.planner import build_plan
from hypothesis_agent.research.tools import FINISH_TOOL, ResearchToolbox, ToolResult
from hypothesis_agent.storage.artifact_store import ArtifactStore
from tests.research_helpers import SyntheticToolSet, company_payloads, make_request

REQUEST = make_request("TEST's operating margins are improving.")
USAGE = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}


def bad_request(message: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", f"{NIM_BASE_URL}/chat/completions"))
    return openai.BadRequestError(message, response=response, body=None)


def tool_reply(name: str, args: Dict[str, Any], content: str = "", **kwargs: Any) -> AIMessage:
    return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"}],
                     usage_metadata=USAGE, **kwargs)


class FakeToolChatModel(BaseChatModel):
    """Scripted chat model supporting ``bind_tools`` and ``with_structured_output``.

    ``responses`` are consumed in order (an Exception is raised instead of returned); when they run
    out, ``responder(messages, kwargs)`` is called. Every request is recorded in ``calls``.
    """

    responses: List[Any] = Field(default_factory=list)
    responder: Optional[Callable[[List[Any], Dict[str, Any]], AIMessage]] = None
    temperature: Optional[float] = None
    reject_temperature: bool = False
    calls: List[Dict[str, Any]] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs, "temperature": self.temperature})
        if self.reject_temperature and self.temperature is not None:
            raise bad_request("Unsupported parameter: 'temperature' is not supported with this model.")
        item = self.responses.pop(0) if self.responses else self.responder(messages, kwargs)
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)

    def with_structured_output(self, schema, *, method="function_calling", include_raw=False, **kwargs):
        self.methods.append(method)
        return super().with_structured_output(schema, include_raw=include_raw)


class RecordingTelemetry:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


def make_llm(*responses: Any, responder=None, **kwargs: Any) -> NvidiaLLM:
    fake = FakeToolChatModel(responses=list(responses), responder=responder,
                             temperature=kwargs.pop("fake_temperature", None),
                             reject_temperature=kwargs.pop("reject_temperature", False))
    return NvidiaLLM(api_key="test-key", model="fake/model", chat_model=fake, telemetry=RecordingTelemetry(), **kwargs)


def make_toolbox(tmp_path: Path) -> ResearchToolbox:
    return ResearchToolbox(
        registry=EvidenceRegistry(), toolset=SyntheticToolSet(), artifact_store=ArtifactStore.from_path(str(tmp_path)),
        workflow_id="wf", allowed_tickers=["TEST"], call_log=[], currencies={}, charts=[], max_calls=40,
    )


# ----------------------------------------------------------------------------------------------
# Structured output
# ----------------------------------------------------------------------------------------------
NARRATIVE = {"executive_summary": "Supported.", "narrative": "Operating margin rose [EV-1].",
             "supporting_points": ["Margin up."], "contradicting_points": [], "limitations": ["One source."]}
REPORT = {"executive_summary": "Supported.", "key_findings": ["Margin up."], "risks": ["Cash."], "next_steps": ["Review."]}
CRITIQUE = {"issues": [{"severity": "high", "description": "Cash flow ignored.", "metrics": ["free_cash_flow"]}],
            "additional_metrics": ["capex_intensity"], "notes": "Check cash."}
PLAN = {"expectations": [{"concept": "cash", "metric": "free_cash_flow", "direction": "increase", "threshold": None,
                          "rationale": "Cash should follow profit."}], "additional_metrics": ["capex_intensity"]}
GRADE = {"score": 4, "justification": "Fine."}


@pytest.mark.parametrize("schema_name, payload, call", [
    ("PlanProposal", PLAN, lambda llm: llm.propose_research_plan(REQUEST, {"free_cash_flow": {}})),
    ("LLMCritique", CRITIQUE, lambda llm: llm.critique_analysis(REQUEST, {"allowed_metrics": ["free_cash_flow"]})),
    ("ResearchNarrative", NARRATIVE, lambda llm: llm.generate_research_narrative(REQUEST, {"rules": []})),
    ("ReportPayload", REPORT, lambda llm: llm.generate_report(REQUEST, {}, "Narrative.", [])),
])
def test_structured_output_parses_each_type(schema_name, payload, call) -> None:
    llm = make_llm(tool_reply(schema_name, payload))
    assert call(llm) == payload
    fake = llm.chat_model
    assert fake.methods == ["json_schema"]
    assert [t["function"]["name"] for t in fake.calls[0]["kwargs"]["tools"]] == [schema_name]


def test_rubric_judge_uses_structured_output() -> None:
    grades = {"logical_coherence": GRADE, "verdict_justification": GRADE, "metric_relevance": {"score": 2, "justification": "Thin."}}
    llm = make_llm(tool_reply("RubricGrades", grades))
    result = LLMRubricJudge(llm).score({"hypothesis": "x"})
    assert result.error is None
    assert {k: v.score for k, v in result.scores.items()} == {"logical_coherence": 4, "verdict_justification": 4, "metric_relevance": 2}
    out_of_range = make_llm(tool_reply("RubricGrades", {**grades, "metric_relevance": {"score": 9, "justification": "?"}}),
                            AIMessage(content="no json"))
    assert LLMRubricJudge(out_of_range).score({}).error  # recorded, not raised


def test_structured_output_recovers_json_from_raw_text_without_another_call() -> None:
    reply = AIMessage(content="<think>drafting</think>\n```json\n" + json.dumps(NARRATIVE) + "\n```", usage_metadata=USAGE)
    llm = make_llm(reply)
    assert llm.generate_research_narrative(REQUEST, {}) == NARRATIVE
    assert len(llm.chat_model.calls) == 1


def test_structured_output_falls_back_to_json_mode() -> None:
    llm = make_llm(AIMessage(content="I would rather write prose."), tool_reply("ReportPayload", REPORT))
    assert llm.generate_report(REQUEST, {}, "Narrative.", []) == REPORT
    assert llm.chat_model.methods == ["json_schema", "json_mode"]


def test_structured_output_falls_back_when_method_is_rejected_by_the_endpoint() -> None:
    llm = make_llm(bad_request("response_format json_schema is not supported"), tool_reply("ReportPayload", REPORT))
    assert llm.generate_report(REQUEST, {}, "Narrative.", []) == REPORT
    assert llm.chat_model.methods == ["json_schema", "json_mode"]


def test_structured_output_falls_back_when_the_endpoint_answers_422() -> None:
    response = httpx.Response(422, request=httpx.Request("POST", f"{NIM_BASE_URL}/chat/completions"))
    rejected = openai.UnprocessableEntityError("response_format not supported", response=response, body=None)
    llm = make_llm(rejected, tool_reply("ReportPayload", REPORT))
    assert llm.generate_report(REQUEST, {}, "Narrative.", []) == REPORT
    assert llm.chat_model.methods == ["json_schema", "json_mode"]


def test_programming_errors_are_not_mistaken_for_unparseable_replies() -> None:
    llm = make_llm(TypeError("bug in schema handling"), tool_reply("ReportPayload", REPORT))
    with pytest.raises(TypeError, match="bug in schema handling"):
        llm.generate_report(REQUEST, {}, "Narrative.", [])
    assert len(llm.chat_model.calls) == 1


def test_structured_output_raises_llm_error_when_every_attempt_fails() -> None:
    llm = make_llm(AIMessage(content="nope"), tool_reply("ResearchNarrative", {**NARRATIVE, "narrative": "  "}))
    with pytest.raises(LLMError, match="ResearchNarrative"):
        llm.generate_research_narrative(REQUEST, {})


def test_transport_errors_are_not_retried_by_the_fallback() -> None:
    error = openai.APIConnectionError(request=httpx.Request("POST", NIM_BASE_URL))
    llm = make_llm(error, tool_reply("ReportPayload", REPORT))
    with pytest.raises(LLMError, match="APIConnectionError"):
        llm.generate_report(REQUEST, {}, "Narrative.", [])
    assert len(llm.chat_model.calls) == 1


def test_lenient_plan_proposal_keeps_per_item_validation() -> None:
    proposal = {"expectations": [PLAN["expectations"][0],
                                 {"concept": "debt", "metric": "total_debt", "direction": "shrinking"}],
                "additional_metrics": []}
    bad = AIMessage(content=json.dumps(proposal))
    llm = make_llm(bad, bad)
    payload = llm.propose_research_plan(REQUEST, {})
    assert payload == proposal  # schema failed twice; raw dict returned for item-level validation
    plan = build_plan("TEST's operating margins are improving.", "TEST", payload)
    assert any(e.metric == "free_cash_flow" and e.source == "llm_planner" for e in plan.expectations)
    assert not any(e.metric == "total_debt" for e in plan.expectations)
    assert any("invalid schema" in note for note in plan.notes)


def test_legacy_list_methods_return_steps() -> None:
    llm = make_llm(tool_reply("StepList", {"steps": ["Collect statements.", "Collect prices."]}))
    assert llm.generate_data_plan(REQUEST) == ["Collect statements.", "Collect prices."]


# ----------------------------------------------------------------------------------------------
# Telemetry and the temperature fallback
# ----------------------------------------------------------------------------------------------
def test_telemetry_records_usage_metadata() -> None:
    llm = make_llm(tool_reply("ReportPayload", REPORT))
    llm.generate_report(REQUEST, {}, "Narrative.", [])
    (record,) = llm.telemetry.records
    assert record["operation"] == "generate_report" and record["model"] == "fake/model"
    assert (record["prompt_tokens"], record["completion_tokens"], record["total_tokens"]) == (120, 30, 150)
    assert record["latency_seconds"] >= 0


def test_rejected_temperature_is_dropped_and_the_call_retried() -> None:
    reply = AIMessage(content="Margins improved.", usage_metadata=USAGE)
    llm = make_llm(reply, reply, fake_temperature=0.2, reject_temperature=True, temperature=0.2)
    assert llm.generate_detailed_analysis(REQUEST, {}) == "Margins improved."
    assert llm.temperature is None
    assert [c["temperature"] for c in llm.chat_model.calls] == [0.2, None]
    assert len(llm.telemetry.records) == 1  # the rejected attempt is not recorded
    assert llm.generate_detailed_analysis(REQUEST, {}) == "Margins improved."  # later calls skip temperature
    assert [c["temperature"] for c in llm.chat_model.calls] == [0.2, None, None]


def test_missing_api_key_is_rejected_and_the_key_never_appears_in_repr() -> None:
    with pytest.raises(LLMError):
        NvidiaLLM(api_key="", model="nvidia/x")
    llm = NvidiaLLM(api_key="secret-test-key", model="nvidia/x")
    assert "secret-test-key" not in repr(llm)


# ----------------------------------------------------------------------------------------------
# Tools and native function calling
# ----------------------------------------------------------------------------------------------
def _without_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _without_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_without_titles(v) for v in schema]
    return schema


def test_as_langchain_tools_schemas_match_arg_models(tmp_path: Path) -> None:
    toolbox = make_toolbox(tmp_path)
    tools = toolbox.as_langchain_tools()
    assert [t.name for t in tools] == list(toolbox.specs) + [FINISH_TOOL]
    for tool in tools[:-1]:
        spec = toolbox.specs[tool.name]
        assert tool.args_schema is spec.args_model and tool.description == spec.description
        function = convert_to_openai_tool(tool)["function"]
        schema = spec.args_model.model_json_schema()
        assert function["name"] == spec.name
        # The bound schema is the arg model's JSON schema (LangChain drops cosmetic titles); unknown
        # fields are still rejected by the toolbox (see the next test).
        assert function["parameters"]["properties"] == _without_titles(schema["properties"])
        assert sorted(function["parameters"].get("required", [])) == sorted(schema.get("required", []))
    assert toolbox.call_log == []  # building tools has no side effects


def test_langchain_tools_execute_through_the_toolbox_and_log_rejections(tmp_path: Path) -> None:
    toolbox = make_toolbox(tmp_path)
    tools = {t.name: t for t in toolbox.as_langchain_tools(origin="llm", rationale="via langchain")}
    message = tools["fetch_dataset"].invoke({"name": "fetch_dataset", "args": {"dataset": "company_info", "ticker": "TEST"},
                                             "id": "c1", "type": "tool_call"})
    assert isinstance(message, ToolMessage) and isinstance(message.artifact, ToolResult)
    assert message.artifact.status == "ok" and json.loads(message.content)["status"] == "ok"
    # Invalid arguments are not pre-validated by LangChain: the toolbox rejects and logs them.
    rejected_extra = json.loads(tools["fetch_dataset"].invoke({"dataset": "company_info", "ticker": "TEST", "path": "/etc"}))
    rejected_ticker = json.loads(tools["fetch_dataset"].invoke({"dataset": "company_info", "ticker": "MSFT"}))
    assert rejected_extra["status"] == "rejected" and rejected_ticker["status"] == "rejected"
    assert "outside the research universe" in rejected_ticker["error"]
    assert [(c["tool"], c["status"], c["origin"], c["rationale"]) for c in toolbox.call_log] == [
        ("fetch_dataset", "ok", "llm", "via langchain"),
        ("fetch_dataset", "rejected", "llm", "via langchain"),
        ("fetch_dataset", "rejected", "llm", "via langchain"),
    ]
    assert tools[FINISH_TOOL].invoke({"reason": "done"}).startswith("Finished")
    assert len(toolbox.call_log) == 3  # finish is never logged


def test_tool_call_conversion() -> None:
    action = action_from_tool_message(tool_reply("compute_metric", {"metric": "net_margin", "ticker": "TEST"}, content="Check net margin."))
    assert action == {"tool": "compute_metric", "arguments": {"metric": "net_margin", "ticker": "TEST"}, "rationale": "Check net margin."}
    reasoning = tool_reply("compute_growth", {"metric": "net_income", "ticker": "TEST"},
                           additional_kwargs={"reasoning_content": "Margins are covered. Net income growth is missing, so compute it."})
    assert action_from_tool_message(reasoning)["rationale"].startswith("Model reasoning: Margins are covered.")
    finish = action_from_tool_message(tool_reply(FINISH_TOOL, {"reason": "Enough evidence."}))
    assert finish == {"tool": "finish", "arguments": {}, "rationale": "Enough evidence."}
    assert action_from_tool_message(AIMessage(content="All done."))["tool"] == "finish"
    malformed = AIMessage(content="", invalid_tool_calls=[{"name": "compute_metric", "args": "{bad", "id": "x",
                                                            "error": "bad json", "type": "invalid_tool_call"}])
    with pytest.raises(LLMError, match="malformed tool call"):
        action_from_tool_message(malformed)


def test_choose_next_tool_call_binds_tools_natively(tmp_path: Path) -> None:
    tools = make_toolbox(tmp_path).as_langchain_tools()
    llm = make_llm(tool_reply("compute_metric", {"metric": "net_margin", "ticker": "TEST"}))
    briefing = {"hypothesis": "h", "tools": [{"name": "compact"}], "available_metrics": ["total_revenue"]}
    action = llm.choose_next_tool_call(REQUEST, briefing, tools)
    assert action["tool"] == "compute_metric" and action["arguments"] == {"metric": "net_margin", "ticker": "TEST"}
    kwargs = llm.chat_model.calls[0]["kwargs"]
    assert [t["function"]["name"] for t in kwargs["tools"]] == [t.name for t in tools]
    assert kwargs["tool_choice"] == "required" and kwargs["parallel_tool_calls"] is False
    prompt = json.loads(llm.chat_model.calls[0]["messages"][-1].content)
    assert "tools" not in prompt and prompt["available_metrics"] == ["total_revenue"]
    assert llm.chat_model.methods == []  # no structured-output call was needed


def test_choose_next_tool_call_falls_back_to_json_mode_when_tools_are_rejected(tmp_path: Path) -> None:
    tools = make_toolbox(tmp_path).as_langchain_tools()
    fallback = tool_reply("AnalysisAction", {"tool": "compute_metric", "arguments": {"metric": "net_margin", "ticker": "TEST"},
                                             "rationale": "json mode"})
    llm = make_llm(bad_request("tool_choice 'required' is not supported"), fallback)
    action = llm.choose_next_tool_call(REQUEST, {"tools": [{"name": "compute_metric"}]}, tools)
    assert action == {"tool": "compute_metric", "arguments": {"metric": "net_margin", "ticker": "TEST"}, "rationale": "json mode"}
    assert llm.chat_model.methods == ["json_mode"]
    assert "compute_metric" in llm.chat_model.calls[1]["messages"][-1].content  # briefing keeps tool descriptions


def test_native_llm_tool_calls_are_validated_logged_and_llm_failures_recorded(tmp_path: Path) -> None:
    """End to end with NvidiaLLM: bad native tool calls are rejected by the toolbox, a failed critique is recorded."""

    steps = iter([
        tool_reply("compute_metric", {"metric": "net_margin", "ticker": "TEST"}, content="check net margin"),
        tool_reply("delete_files", {"path": "/"}),
        tool_reply("fetch_dataset", {"dataset": "company_info", "ticker": "MSFT"}),
        tool_reply(FINISH_TOOL, {"reason": "enough"}),
    ])

    def responder(messages, kwargs):
        names = [t["function"]["name"] for t in kwargs.get("tools", [])]
        if "PlanProposal" in names:
            return tool_reply("PlanProposal", {"expectations": [], "additional_metrics": []})
        if "LLMCritique" in names:
            return AIMessage(content="not json")
        if "ResearchNarrative" in names:
            return tool_reply("ResearchNarrative", {"narrative": "The hypothesis is supported."})
        if "ReportPayload" in names:
            return tool_reply("ReportPayload", {"executive_summary": "Supported."})
        assert "compute_metric" in names and kwargs["tool_choice"] == "required"
        return next(steps)

    llm = make_llm(responder=responder)
    config = AppSettings(artifact_store_path=str(tmp_path / "artifacts"), llm_provider="offline")
    orchestrator = LangGraphValidationOrchestrator(settings=config, llm=llm, artifact_store=ArtifactStore.from_path(config.artifact_store_path),
                                                   toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = orchestrator.run_research(REQUEST, {"metadata": {"workflow_id": "wf-native"}})
    state = context["research"]
    llm_calls = [c for c in state["tool_calls"] if c["origin"] == "llm"]
    assert [(c["tool"], c["status"]) for c in llm_calls] == [
        ("compute_metric", "ok"), ("delete_files", "rejected"), ("fetch_dataset", "rejected")]
    assert llm_calls[0]["rationale"] == "check net margin"
    assert "Unknown tool" in llm_calls[1]["error"] and "outside the research universe" in llm_calls[2]["error"]
    assert EvidenceRegistry.from_dict(state["registry"]).find(metric="net_margin")
    assert [e["stage"] for e in state["errors"]] == ["critique_analysis"]
    assert state["verdict"]["verdict"] == "SUPPORTED"  # deterministic, unaffected by the LLM
    assert state["narratives"][0]["narrative"]["narrative"] == "The hypothesis is supported."


# ----------------------------------------------------------------------------------------------
# The real ChatOpenAI stack against a mocked NIM endpoint
# ----------------------------------------------------------------------------------------------
def test_chat_openai_against_mock_nim_endpoint() -> None:
    requests: List[Dict[str, Any]] = []

    def completion(message: Dict[str, Any], finish: str = "stop") -> httpx.Response:
        return httpx.Response(200, json={
            "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "nvidia/test",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
        })

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{NIM_BASE_URL}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        requests.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "temperature is not supported for this model"}})
        if body.get("tools"):
            return completion({"content": None, "reasoning_content": "Net margin is missing. Compute it.",
                               "tool_calls": [{"id": "call-1", "type": "function", "function": {
                                   "name": "compute_metric", "arguments": json.dumps({"metric": "net_margin", "ticker": "TEST"})}}]},
                              finish="tool_calls")
        return completion({"content": json.dumps(NARRATIVE)})

    telemetry = RecordingTelemetry()
    llm = NvidiaLLM(api_key="test-key", model="nvidia/test", telemetry=telemetry,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.generate_research_narrative(REQUEST, {}) == NARRATIVE
    assert requests[0]["temperature"] == 0.2 and "temperature" not in requests[1]
    assert requests[1]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"]["json_schema"]["name"] == "ResearchNarrative"
    assert telemetry.records[-1]["total_tokens"] == 52 and telemetry.records[-1]["operation"] == "generate_research_narrative"

    tool = convert_to_openai_tool({"title": "compute_metric", "description": "Compute a ratio.", "type": "object",
                                   "properties": {"metric": {"type": "string"}, "ticker": {"type": "string"}}})
    action = llm.choose_next_tool_call(REQUEST, {"hypothesis": "h"}, [tool])
    assert action["tool"] == "compute_metric" and action["arguments"] == {"metric": "net_margin", "ticker": "TEST"}
    assert action["rationale"] == "Model reasoning: Net margin is missing. Compute it."
    assert requests[-1]["tool_choice"] == "required" and requests[-1]["parallel_tool_calls"] is False
    assert len(requests) == 3


def test_tool_loop_supports_adapters_without_native_tool_calling(tmp_path: Path) -> None:
    from hypothesis_agent.config import AppSettings
    from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
    from tests.research_helpers import ScriptedLLM, company_payloads

    class LegacyAdapter:
        """Duck-typed adapter (no BaseLLM base class, no choose_next_tool_call)."""

        def __init__(self) -> None:
            self._inner = ScriptedLLM(tool_actions=[
                {"tool": "compute_metric", "arguments": {"metric": "net_margin", "ticker": "TEST"}, "rationale": "legacy"},
                {"tool": "finish"},
            ])

        def __getattr__(self, name: str):
            if name == "choose_next_tool_call":
                raise AttributeError(name)
            return getattr(self._inner, name)

    settings = AppSettings(artifact_store_path=str(tmp_path / "artifacts"), llm_provider="offline")
    orchestrator = LangGraphValidationOrchestrator(settings=settings, llm=LegacyAdapter(),
                                                   artifact_store=ArtifactStore.from_path(settings.artifact_store_path),
                                                   toolset=SyntheticToolSet(company_payloads(healthy_cash=True)))
    context = orchestrator.run_research(REQUEST, {"metadata": {"workflow_id": "wf-legacy"}})
    llm_calls = [c for c in context["research"]["tool_calls"] if c["origin"] == "llm"]
    assert llm_calls and llm_calls[0]["rationale"] == "legacy"

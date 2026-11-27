# Design decisions

This records the main choices behind JASPER's evidence-grounded research agent, what they buy,
and what they cost.

## 1. Deterministic numbers, LLM language

All figures come from deterministic functions over a registry of stored data. The LLM may choose
which analytical tool to call next, propose extra expectations or critique issues, and write prose.
It never computes a number that enters the evidence registry, and it cannot set the verdict.

*Why:* the original pipeline executed LLM-written Python and printed whatever came back. When that
code returned an unexpected schema (as in the stored `analysis_metrics.json` from earlier runs, with
empty `steps` and `aggregated`), the report had nothing traceable to rely on. Deterministic tools
are testable, reproducible, and lineage-preserving.

*Cost:* the analytical vocabulary is limited to the catalogue in `research/metrics.py`. Questions
outside it end in `INSUFFICIENT_EVIDENCE`, which is the intended behaviour, not a crash. The legacy
code-generation path remains available behind `ENABLE_CODEGEN_ANALYSIS`, and its outputs are
labelled unverified and excluded from the verdict.

**The LLM layer is LangChain.** `NvidiaLLM` wraps `langchain_openai.ChatOpenAI` pointed at NVIDIA
NIM's OpenAI-compatible endpoint. Plan proposals, critique, narrative, report and rubric-judge
responses are parsed into pydantic models with `with_structured_output(method="json_schema")`,
which NIM enforces during generation; this was checked live on the benchmark generator
(`nemotron-3.5-lightning`) and judge (`minimax-m3.1-flash`). If a reply still does not parse,
the raw message is re-read, then the call is retried once in JSON mode; plan and critique
payloads that fail both fall through to the existing per-item validators, which record
rejections. The research tools are exposed as LangChain `StructuredTool`s built from the same
pydantic argument models (`ResearchToolbox.as_langchain_tools`), and the tool-choice step binds
them for native function calling. The tool functions call `ResearchToolbox.invoke`, and LangChain's
own argument parsing is bypassed, so validation, the ticker allow-list, the call budget and the
call log stay in one place: a malformed or disallowed call is rejected and logged exactly as
before. Adapters without function calling (the offline narrator, test doubles) use the default
`BaseLLM.choose_next_tool_call`, which falls back to the JSON action.

*Why:* the previous adapter described each schema in prose and parsed the reply by hand, so a
reply wrapped in text or with a missing key cost that step its LLM contribution, and the old
tool-choice prompt described tools in free text. Structured output and bound tool schemas give
the model the exact contract, while the deterministic layer keeps the final say. The port was
checked for parity: offline benchmark results are identical, and a live NVDA-01 run completed with
the expected verdict, six native tool calls and no LLM errors. When reasoning models return empty
message content with a tool call, the logged rationale is the tail of their reasoning text.

## 2. Rule-based verdict

The verdict is a function of expectation statuses (`SUPPORTS`=1, `MIXED_POSITIVE`=0.65,
`NEUTRAL`=0.5, `MIXED_NEGATIVE`=0.35, `CONTRADICTS`=0, averaged within each concept group and then
across groups; >=0.8 supported, <=0.2 not supported). Any unmeasurable core expectation gives
`INSUFFICIENT_EVIDENCE`.

*Why:* a verdict that an LLM writes cannot be audited or reproduced. A rule can be challenged
line by line, and the claim verifier flags any narrative that states a different verdict.

*Cost:* thresholds (2% relative change for amounts, 0.5 percentage points for ratios, PEG 1/2) are
conventions. The report says so, and assessments carry caveats when readings sit near a threshold.

## 3. "Contradicted" requires an unambiguous referent

A mismatch becomes `CONTRADICTED` only when the claim cites an evidence ID, names a period, or
refers to a single-valued metric. Otherwise it is `UNSUPPORTED`, explicitly "not proof the claim
is false".

*Why:* the task requires that verification failure never be read as falsity. Conservative
contradiction keeps that line clean; unsupported claims are still surfaced prominently.

## 4. Evidence IDs are content-addressed

Snapshot IDs hash the call and payload; evidence IDs hash ticker, metric, period, snapshot and
locator (raw) or operation, inputs and parameters (derived). Re-registration of identical evidence
is idempotent; a different value for the same metric and period from another snapshot is recorded
as a conflict, which the critique can escalate.

*Why:* identical inputs produce identical IDs across runs, so reports and benchmark results are
diffable and claims can cite stable identifiers.

## 5. The stage router is compiled into a LangGraph StateGraph

The routing is declared once, as a table (`research/agent.py:ROUTES`): fixed successors for the
linear steps and three routers that read recorded state (after critique, after claim
verification, after the report). `LangGraphValidationOrchestrator.build_graph` compiles that table
into a LangGraph `StateGraph` with one node per stage (each wrapping `run_stage`) and conditional
edges calling the same routers; `next_stage` resolves single steps from the same table for tests
and stage-by-stage callers. Stage names and the `run_stage` contract are unchanged.

The graph state is plain JSON: the request, the stage context, and the last node's milestone,
evidence and summary. That keeps checkpoints serialisable and the research state as auditable as
before; LangGraph decides the order of execution, never the content.

*Why:* the loop stays explicit and testable stage by stage, but the runtime is a standard one:
conditional edges instead of a hand-written `while` loop, a `recursion_limit` (40 node executions
per invocation) as defence in depth on top of the iteration, reanalysis, narrative-attempt and
tool-call limits, streaming of node start and finish events for the milestone timeline, and
checkpointing for human review (section 6). The port was checked for parity: the offline core,
held-out and paraphrase runs give identical verdicts, scores, claim counts and routes before and
after.

*Note:* the original code carried the LangGraph name and dependency without using its runtime.

## 6. Critique-triggered human review reuses the existing gate

When the critique decides `HUMAN_REVIEW` (high-severity source conflict, LLM-flagged high-impact
ambiguity that names real metrics, or calculations that stay inconsistent after reanalysis), the
agent still finishes the report, then pauses before delivery with the reason attached. The pause
is LangGraph's `interrupt()` inside the `human_review` node; the workflow client owns an
`InMemorySaver` checkpointer (thread id = workflow id + run id), and the resume endpoint continues
the checkpointed thread with `Command(resume=decision)`, which records the decision and runs
delivery. Finished, failed and cancelled threads are deleted.

*Why:* reviewers get a complete report to judge rather than a half-finished state, the existing
resume API works unchanged, and the paused run is a real checkpoint rather than hand-kept copies
of the summary. Checkpoints are in memory, like the workflow client's run table, so a server
restart loses paused runs; a persistent checkpointer (e.g. SQLite) is the upgrade path.

## 7. Frozen fixtures for evaluation

Benchmark cases run against payloads captured through the same `YFinanceTool` code path the live
system uses, including captured failures (the news endpoint was down at capture time, and that gap
is replayed). Snapshot metadata records `frozen_fixture` mode and the capture time.

*Why:* expected conclusions must stay valid while Yahoo's data changes, and two runs of the suite
must be comparable.

## 8. Independent, disk-based scoring

The evaluator reloads the run's files, checks reproducibility, recomputes the verdict from recorded
assessments and re-verifies every claim. Disagreement with the run's own verification is a failure
category. Rubric judges are reported alongside deterministic scores, never instead of them.

## Known limitations

- The rule-based hypothesis parser covers common phrasings only. The blind probes measured this:
  on the paraphrase probe, 4 of 6 verdicts were acceptable before vocabulary fixes (one of those
  four for the wrong reason). On the held-out set, 6 of 8 were acceptable, and both misses were
  `INSUFFICIENT_EVIDENCE` abstentions rather than wrong directional calls. The LLM planner can add
  expectations; it was not separately benchmarked at scale because of endpoint latency.
- Expected benchmark conclusions were written by the same developer who built the rules, from the
  same frozen data. In-sample offline accuracy is therefore an optimistic upper bound; the
  held-out result is the honest estimate.
- Claim extraction is pattern-based. On the first live-LLM run, 22 of 26 flagged claims were
  checker misreads of free-form prose (fixed and covered by regression tests); new phrasings will
  still produce some false flags. Citations reduce this because they pin the comparison, and a
  flag always shows the sentence and the evidence so a reviewer can judge it.
- The LLM critic escalated to human review in 3 of 6 live cases; its severity threshold may need
  tightening.
- Relative trends for amounts measured from a near-zero base (e.g. quarterly free cash flow)
  produce extreme percentages; they are correct arithmetic but easy to misread.
- Yahoo Finance offers roughly four annual periods, no segment data, and undisclosed
  methodology for TTM ratios. These limits are stated in every report.
- No causal controls (peers, macro, FX, M&A) are applied; associations are descriptive.

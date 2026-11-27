# JASPER: Evidence-Grounded Investment Research Agent

JASPER tests an investment hypothesis against financial data and produces an auditable research
report. It plans what evidence the hypothesis needs, collects it from Yahoo Finance through
validated tools, computes every figure deterministically, critiques its own analysis, widens the
investigation when the evidence is incomplete or contradictory, verifies every number the LLM
writes, and reports its uncertainty in structured form.

The design goal: a skeptical quantitative reviewer can trace each number in the PDF to a stored
data snapshot, recompute it, see which claims could not be verified, and follow why the agent
reached its verdict.

---

## What changed in v0.2

| Before | Now |
|---|---|
| Linear `plan -> collect -> codegen -> narrative -> PDF` | Routed research loop: plan, collect, analyse, **critique**, branch (collect more / revise plan / reanalyse / human review / insufficient evidence), narrative, **claim verification** (with correction loop), report |
| One-shot LLM-generated Python for analysis | Deterministic analytics invoked through validated, budgeted tools (the LLM may choose extra tool calls; it never computes figures) |
| Free-text numbers in the narrative were trusted | Every figure is extracted and verified against the evidence registry; failures are listed in the PDF |
| "Source: Yahoo Finance" | Evidence IDs, hashed data snapshots, exact `yfinance` calls, locators and derivation lineage |
| Formula-based 0-1 score and "confidence" | Rule-based verdict plus structured uncertainty (completeness, evidence balance, source quality, limitations) |
| No evaluation | 18-case benchmark on frozen data, independent scoring, rubric judges, human review support, version comparison |

---

## Architecture

```
            Web UI / REST API / jasper CLI / benchmark runner
                                  |
     HypothesisWorkflowClient (graph.astream + InMemorySaver checkpoints;
                               human review = interrupt(), resumed with Command)
                                  |
     LangGraphValidationOrchestrator.build_graph -> LangGraph StateGraph
       (one node per stage, conditional edges from research/agent.py:ROUTES,
        recursion_limit 40)
                                  |
   +-------------------------- ResearchAgent ----------------------------+
   |                                                                     |
   |  plan_generation -> data_collection -> hybrid_analysis -> critique  |
   |        ^                  ^                  ^              |       |
   |        |                  +-- COLLECT_MORE_DATA / REVISE_PLAN       |
   |        |                                     +-- REANALYZE          |
   |        |            CONTINUE / INSUFFICIENT_EVIDENCE / HUMAN_REVIEW |
   |        |                                     v                      |
   |        |         detailed_analysis <-> claim_verification           |
   |        |          (narrative regenerated with corrections)          |
   |        |                                     v                      |
   |        +---------------------------- report_generation -> [human_review] -> delivery
   +---------------------------------------------------------------------+
        |                |                   |                   |
   planner.py      tools.py (validated,   critique.py        claims.py
   (expectations,   budgeted; no file/     (structured        (extract + verify
   versioned plan)  shell access)          issues, decide)    against registry)
                         |
          analytics.py (deterministic) <-> evidence.py (registry, snapshots, lineage)
                         |
          YFinanceToolSet (live) | FixtureToolSet (frozen, reproducible)
```

Key modules (all under `src/hypothesis_agent/`):

| Module | Responsibility |
|---|---|
| `research/evidence.py` | `EvidenceRegistry`, `DataSnapshot` (exact call, parameters, retrieval time, SHA-256, stored payload), `EvidenceRecord` (value, unit, currency, period, locator, derivation lineage), missing-evidence and conflict tracking |
| `research/metrics.py` | Canonical metric catalogue shared by every component (fields, units, categories, synonyms, formulas) |
| `research/extraction.py` | yfinance payload to evidence records |
| `research/analytics.py` | Ratios, growth, CAGR, trends, price metrics, valuation, correlation, and `recompute()` for every derivation |
| `research/tools.py` | `ResearchToolbox`: pydantic-validated tool calls, ticker allow-list, call budget, deduplicated fetches, call log; `as_langchain_tools()` exposes the same tools to LangChain |
| `llm.py` | `NvidiaLLM` on LangChain (`ChatOpenAI` against NVIDIA NIM): pydantic structured output for plan proposals, critique, narrative and report; native function calling for tool choice |
| `research/planner.py` | Hypothesis to testable expectations and a versioned, revisable plan |
| `research/assessment.py` | Rule-based evaluation of each expectation and the verdict |
| `research/critique.py` | Structured critique issues, required actions and the branching decision |
| `research/claims.py` | Claim extraction and deterministic verification |
| `research/uncertainty.py` | Structured uncertainty assessment |
| `research/agent.py` | Research nodes and the routing table (`ROUTES`) the graph is compiled from |
| `orchestration/langgraph_pipeline.py` | `LangGraphValidationOrchestrator`: stage execution, the compiled LangGraph `StateGraph`, batch `run_research` |
| `workflows/hypothesis_workflow.py` | Async workflow client: streams the graph, records milestones, pauses and resumes human review through LangGraph checkpoints |
| `research/reproduce.py` | Re-verifies a run directory from disk |
| `reporting/pdf_report.py` | The evidence-backed PDF |
| `evaluation/` | Benchmark loading, runner, scoring, judges, human review, comparison |

### The research loop

1. **Plan.** The hypothesis is parsed into expectations such as `operating_margin increase`,
   `total_revenue_growth_yoy above 10%`, `peg_computed overvalued`, a base-effect test, or an
   *unmeasurable* concept (segment data, guidance, market share). An optional LLM planner may add
   expectations; its proposals are validated against the metric catalogue and cannot replace parsed ones.
2. **Collect.** Datasets required by the plan are fetched through `fetch_dataset`. Each raw payload
   is stored, hashed and registered as a snapshot; canonical metrics become evidence records.
   Failed retrievals become error snapshots plus missing-evidence entries.
3. **Analyse.** Planned ratios, growth rates and trends are computed through tools; the LLM may then
   request extra tool calls (bounded, validated, logged). Expectations are assessed by fixed rules.
4. **Critique.** Deterministic checks look for missing metrics, profit/cash divergence, late
   reversals, earnings-quality problems (net income positive with an operating loss), negative
   equity, source conflicts, inconsistent calculations and unmeasurable concepts. An optional LLM
   critic can add issues but cannot clear a deterministic finding. The decision policy picks
   `CONTINUE`, `COLLECT_MORE_DATA`, `REVISE_PLAN` (plan version +1, trigger recorded),
   `REANALYZE`, `HUMAN_REVIEW` or `INSUFFICIENT_EVIDENCE`, bounded by an iteration limit,
   a reanalysis limit and diminishing-returns detection. At the limit it reports an honest partial result.
5. **Narrate and verify.** The narrator receives an evidence briefing and must cite evidence IDs.
   Claims are extracted and verified; if figures are contradicted or unsupported, the narrative is
   regenerated once with the failures as corrections. Remaining failures are reported, not hidden.
6. **Report.** The PDF is rendered from recorded state; the human-review gate is applied when the
   request asks for it or the agent escalated.

The loop runs as a LangGraph `StateGraph`: each stage is a node, the critique decision, the
claim-verification retry and the review/delivery split are conditional edges, and every
invocation has a `recursion_limit` of 40 node executions on top of the research limits. The graph
state is plain JSON (request, stage context, last stage outcome). In the web app the workflow
client runs the graph with an in-memory checkpointer; `human_review` pauses with `interrupt()` and
`POST /v1/hypotheses/{id}/resume` continues the checkpointed thread with `Command(resume=decision)`.
Checkpoints live in process memory, so a paused run does not survive a server restart. The CLI and
benchmark run the same graph synchronously and stop after the report. If LangSmith tracing is
enabled through its environment variables, LangGraph traces runs including the full stage context.

### Evidence verification

Each extracted claim gets one of:

- `VERIFIED`: matches evidence (or a value recomputed from raw inputs) within rounding tolerance
  (half a unit of the last stated digit; wider for "about"/"approximately").
- `PARTIALLY_VERIFIED`: approximately right (within 5%), or right for a different or unstated
  period, or for a related source (e.g. Yahoo TTM instead of fiscal-year).
- `CONTRADICTED`: the metric and period are unambiguous (cited evidence ID, explicit year, or a
  single-valued metric) and the evidence differs.
- `UNSUPPORTED`: no evidence could confirm it. This is not proof the claim is false.
- `NOT_VERIFIABLE`: qualitative or unlinked statements.

Citations are machine-checked: an evidence ID that does not exist is flagged, and a citation pins
the comparison to that record. Verdict statements in the narrative are checked against the
rule-based verdict, which the LLM cannot change.

### Source attribution

Every snapshot records ticker, dataset, the exact call (e.g. `yfinance.Ticker('KO').cashflow`),
parameters, retrieval timestamp (capture time for replayed fixtures), SHA-256 of the canonical
payload, row coverage and the stored payload path. Raw evidence records hold a locator
(`[date=2025-12-31]["Total Revenue"]`); derived records hold the operation, formula, parameters and
input evidence IDs. Chart captions list unit, periods, evidence IDs, snapshot IDs, retrieval calls
and the calculation. `reproducibility_manifest.json`, `evidence_registry.json` and the raw
snapshots are written next to each report; `jasper reproduce <run_dir>` re-hashes payloads,
re-reads raw values and recomputes all derived values from disk.

### Structured uncertainty

Instead of a confidence number, each report states: metrics requested, collected and unavailable
(with reasons), failed retrievals and empty periods; expectations that support, contradict, are
mixed/neutral or could not be evaluated; data sources, known source limitations, freshness and
inconsistencies; and analytical limitations (no causal controls, fixed trend thresholds,
forward-looking hypotheses, iteration limits, plus every caveat raised by the assessment and critique).
The API keeps `score` and `confidence` for compatibility, redefined as the rule-based evidence
score and the data-completeness ratio. Neither is a probability.

---

## Evaluation

`benchmarks/v1/` holds 18 hypotheses (supported, contradicted, partially supported, conflicting
metrics, period-dependent, insufficient evidence) over frozen Yahoo Finance snapshots for 18
tickers, plus a paraphrase probe and a held-out set. See [benchmarks/v1/README.md](benchmarks/v1/README.md).

The runner gives the agent only the hypothesis and ticker, then scores each run **from disk**:
reproducibility check, verdict recomputed from the recorded assessments, and every narrative claim
re-verified independently (disagreements with the run's own verification are a failure category).

| Dimension | How it is measured |
|---|---|
| Verdict accuracy | Rule-based verdict vs expected (1.0 exact, 0.75 acceptable, 0.25 adjacent) |
| Numerical accuracy | Share of figures in generated text that independently verify |
| Metric relevance | Share of metrics used by assessments and claims that fall in the case's expected evidence categories |
| Logical coherence | Checks: verdict recomputes from assessments, narrative states the verdict and it matches, contradicting evidence is disclosed, no contradicted figures |
| Evidence coverage | Expected evidence categories actually investigated |
| Unsupported claims | Unsupported and contradicted claims, as a count and a rate |
| Caveat recall | Expert caveats (keyword-matched) surfaced anywhere in the audit trail |

Subjective dimensions also get rubric scores (1-5) from a deterministic rubric judge and,
optionally, a blind LLM judge (`--judge nvidia`) that never sees the expected answer. Human review
uses the same rubric through a blind CSV.

---

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -e .
```

`.env` (see `.env.example`):

```env
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=mistralai/mistral-large-4-1t-a52b
# LLM_PROVIDER=offline                       # run without any LLM (deterministic narrator)
# RESEARCH_FIXTURE_DIR=benchmarks/v1/fixtures # replay frozen data instead of live Yahoo
# RESEARCH_MAX_ITERATIONS=3
```

NVIDIA retires hosted models regularly. If calls fail with HTTP 410, choose a current model from
`https://integrate.api.nvidia.com/v1/models`.

### Run one hypothesis

```bash
# Offline and reproducible: frozen data, deterministic narrator
jasper research --ticker KO --llm offline --fixtures benchmarks/v1/fixtures \
  --hypothesis "Coca-Cola's operating margins are improving but its cash conversion is deteriorating."

# Live Yahoo Finance data with the NVIDIA LLM
jasper research --ticker AAPL --llm nvidia --hypothesis "Apple is experiencing improving profitability despite slowing revenue growth."

# Re-verify a finished run from disk
jasper reproduce data/artifacts/<workflow_id>
```

### Run the benchmark

```bash
jasper eval run --suite benchmarks/v1 --llm offline --label v0.2-offline
jasper eval run --suite benchmarks/v1 --llm offline --include-holdout --only-extra --label holdout
jasper eval run --suite benchmarks/v1 --llm nvidia --judge nvidia --label v0.2-llm
jasper eval compare eval_results/v0.2-offline/results.json eval_results/v0.2-llm/results.json --out comparison.md
jasper eval human-template eval_results/v0.2-llm/results.json review.csv
jasper eval human-import eval_results/v0.2-llm/results.json review.csv
```

`python -m hypothesis_agent ...` is equivalent to `jasper ...` if the console script is not on PATH.

### Web app

```bash
uvicorn hypothesis_agent.main:app --reload    # with PYTHONPATH=src if not installed
```

### Tests

```bash
python -m pytest
```

---

## API

| Endpoint | Method | Description |
|---|:---:|---|
| `/v1/hypotheses` | `POST` | Submit a hypothesis. |
| `/v1/hypotheses/{id}` | `GET` | Submission details and summary. |
| `/v1/hypotheses/{id}/status` | `GET` | Live status and milestones (now including `critique` and `claim_verification`). |
| `/v1/hypotheses/{id}/report` | `GET` | `ValidationSummary` with `verdict`, `claim_verification`, `uncertainty`, `research_iterations`, `human_review_reason`. |
| `/v1/hypotheses/{id}/resume` | `POST` | Human review decision (`approved`, `rejected`, `needs_changes`). |
| `/v1/hypotheses/{id}/cancel` | `POST` | Cancel a running workflow. |

See [docs/design-decisions.md](docs/design-decisions.md) for the reasoning behind the main choices
and the known limitations.

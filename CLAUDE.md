# JASPER notes for Claude

Evidence-grounded investment research agent. Architecture: `README.md`. Rationale and known
limitations: `docs/design-decisions.md`. Benchmark method and all results: `benchmarks/v1/README.md`.

## Environment (Windows)

- Use the project venv: `.venv/Scripts/python` (Python 3.12). The system `python` is a different
  version without the dependencies.
- Install or refresh after pulling: `.venv/Scripts/python -m pip install -e ".[dev]"`. Runtime
  stack: LangGraph 1.2 (orchestration), langchain-core / langchain-openai 1.8 (LLM layer), openai 3.x.
- Without the editable install, set `PYTHONPATH=src`.
- Tests: `PYTHONPATH=src .venv/Scripts/python -m pytest -q`.
- Use the Edit tool for source edits that contain regex escapes; shell heredocs have turned `\b`
  into a backspace in this repo before.

## Running

- Offline, deterministic, about 10 s for 18 cases:
  `PYTHONPATH=src .venv/Scripts/python -m hypothesis_agent eval run --suite benchmarks/v1 --llm offline --label <label>`
  Add `--include-holdout --only-extra` or `--include-paraphrases --only-extra` for the probes.
- Compare two runs: `... -m hypothesis_agent eval compare <baseline>/results.json <candidate>/results.json`.
  A refactor must leave verdicts, scores and claim counts identical.
- Live LLM (NVIDIA NIM, 5-9 minutes per case; run in the background):
  `NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b JUDGE_MODEL=minimaxai/minimax-m3.1-flash RESEARCH_MAX_LLM_TOOL_STEPS=2 PYTHONPATH=src .venv/Scripts/python -u -m hypothesis_agent eval run --suite benchmarks/v1 --llm nvidia --judge nvidia --cases <ids> --label <label>`
- NIM retires models (HTTP 410). List current ones with the OpenAI client against
  `https://integrate.api.nvidia.com/v1/models`. Never print the API key from `.env`.
- Web app: `PYTHONPATH=src .venv/Scripts/python -m uvicorn hypothesis_agent.main:app`
  (`LLM_PROVIDER=offline` and `RESEARCH_FIXTURE_DIR=benchmarks/v1/fixtures` for an offline run).
- Replay the LLM-critic escalation rule over recorded runs: `scripts/replay_critic_escalation.py`.

## Rules that must hold

- All figures come from deterministic functions over the evidence registry. The LLM never computes
  registry values or sets the verdict (`research/assessment.py:determine_verdict`).
- A claim is `CONTRADICTED` only when its metric and period are unambiguous; otherwise it is
  `UNSUPPORTED`, which is not proof the claim is false.
- Missing data is recorded (`record_missing`), never stored as a value.
- Tool calls, from code or from the LLM through LangChain, go through `ResearchToolbox.invoke`
  (validation, ticker allow-list, budget, call log).
- Routing lives in one table, `research/agent.py:ROUTES`, compiled into the LangGraph `StateGraph`.
- Expected answers live only in `benchmarks/v1/*.json`; a test asserts they never reach the agent.
- Benchmark honesty: never tune against `benchmarks/v1/holdout.json`; describe core-set offline
  accuracy as in-sample; the paraphrase probe is no longer blind.
- `eval_results/` is gitignored and holds the recorded runs the docs cite.

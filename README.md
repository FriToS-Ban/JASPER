# JASPER — Hypothesis Validation Copilot

JASPER is a powerful LangGraph-powered AI copilot designed to help equity analysts validate investment hypotheses rapidly. By pasting a natural language thesis, JASPER automatically coordinates an end-to-end investment diligence loop: structured data planning, multi-source financial data collection via Yahoo Finance (`yfinance`), sandboxed hybrid LLM analytics, detailed narrative synthesis, and a professional PDF report ready for download.

JASPER collapses multi-hour equity research sprints into an interactive, audit-ready session with human-in-the-loop oversight.

---

## 🚀 Key Features

- **🔁 LangGraph-Style Orchestration:** Deterministic workflow pipeline composed of stages with built-in telemetry, logging, and error handling.
- **🧠 Hybrid LLM Analytics (NVIDIA NIM):** Prompt-safe autonomous python code generation and sandboxed execution to compute financial metrics and plot charts.
- **📊 Robust Yahoo Finance Integration:** Custom `YFinanceToolSet` to retrieve fundamental data, historical prices, analyst recommendations, cash flow statements, balance sheets, news, and insider/institutional holders.
- **🛡️ Secure Python REPL Sandbox:** Built-in Python execution sandbox with restricted built-in operations for secure, local analysis.
- **👤 Human-in-the-Loop Review Gates:** Optional manual review states that pause the pipeline before delivery, letting analysts review findings, select a verdict (`approved`, `rejected`, `needs_changes`), and resume.
- **📄 ReportLab PDF Publishing:** Auto-compiles quantitative metrics, narrative summaries, and generated charts into a client-ready, downloadable PDF memo.
- **🖥️ FastAPI + Sleek Dark UI:** Modern single-page web app to submit hypotheses, track live milestones, view metrics, and download reports.

---

## 📐 Architecture Overview

```
           [ Web UI / Client ]
                    │
                    ▼  (REST API calls)
         [ FastAPI Application ]
                    │
                    ▼
         [ Hypothesis Service ] ───────> [ In-Memory Repo ]
                    │
                    ▼
      [ HypothesisWorkflowClient ]
                    │  (Async Local Execution)
                    ▼
    [ LangGraphValidationOrchestrator ]
       ├── 1. Plan Generation (LLM)
       ├── 2. Data Collection (YFinance Tools)
       ├── 3. Hybrid Analysis (Sandboxed Python REPL + Matplotlib)
       ├── 4. Detailed Analysis (LLM Narrative)
       ├── 5. Report Generation (ReportLab PDF compilation)
       ├── 6. Human Review (Optional Approval Gate)
       └── 7. Delivery (Publishing Report for Download)
```

---

## 📁 Repository Structure

```
Jasper/
├── src/
│   └── hypothesis_agent/
│       ├── api/                # API router and Jinja2 UI templates
│       ├── db/                 # Database connection/models (if any)
│       ├── models/             # Pydantic schemas (requests, responses, summaries)
│       ├── orchestration/      # LangGraph pipeline, Python REPL sandbox, YFinance tools
│       ├── repositories/       # Hypothesis data repositories
│       ├── services/           # Business logic layer
│       ├── storage/            # Local JSON/PDF/log storage utility
│       ├── workflows/          # Workflow activities and client controllers
│       ├── config.py           # Pydantic Settings configuration
│       ├── llm.py              # LLM clients (Nvidia NIM OpenAI-compatible)
│       └── main.py             # FastAPI App Entrypoint
├── tests/                      # Pytest unit and integration test suite
├── pyproject.toml              # Build config and project dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- **NVIDIA NIM API Key:** Used to power LLM tasks.

### 1. Installation & Environment Setup
Clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/ayushsi42/raven.git jasper
cd jasper

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell/CMD)
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

### 2. Configure Environment Variables
Create a `.env` file in the project root folder. Reference the [.env.example](file:///c:/Users/SHIBASISH/Jasper/.env.example) file for format:

```env
# Required NVIDIA NIM Configuration
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=minimaxai/minimax-m2.7   # or meta/llama-3.1-70b-instruct

# Optional Application Settings
LOG_LEVEL=INFO
API_PREFIX=/v1
ENABLE_PROMETHEUS=true
ARTIFACT_STORE_PATH=./data/artifacts
```

### 3. Start the Server & UI
Launch the FastAPI development server using Uvicorn:

```bash
# Windows / Unix
$env:PYTHONPATH="src"  # Windows PowerShell
# OR export PYTHONPATH=src (macOS/Linux)

uvicorn hypothesis_agent.main:app --reload
```

Once running, visit:
- **JASPER Web Interface:** [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Running Tests

A comprehensive suite of unit and integration tests is included. Run them locally to verify your setup:

```bash
# Set PYTHONPATH and execute pytest
$env:PYTHONPATH="src"
.venv\Scripts\python -m pytest
```

---

## 📡 API Reference

JASPER exposes a clean REST API under `/v1` for external integrations:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/v1/hypotheses` | `POST` | Submit a new investment hypothesis. |
| `/v1/hypotheses/{id}` | `GET` | Get submission details and summary. |
| `/v1/hypotheses/{id}/status` | `GET` | Retrieve live execution status and current milestone progress. |
| `/v1/hypotheses/{id}/report` | `GET` | Fetch the final validation outcome report. |
| `/v1/hypotheses/{id}/resume` | `POST` | Submit human review decision (`approved`, `rejected`, `needs_changes`). |
| `/v1/hypotheses/{id}/cancel` | `POST` | Cancel a running workflow. |

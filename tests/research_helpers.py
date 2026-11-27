"""Synthetic yfinance-shaped data and stub LLMs for research-agent tests.

All numbers are hand-specified so every expected value in the tests is known exactly.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from hypothesis_agent.llm import BaseLLM
from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.orchestration.fixture_tools import fixture_key

B = 1e9
YEARS = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]


def _statement(columns: Dict[str, List[Optional[float]]]) -> List[Dict[str, Any]]:
    rows = []
    for index, period in enumerate(YEARS):
        row: Dict[str, Any] = {"date": period}
        for field, values in columns.items():
            row[field] = values[index]
        rows.append(row)
    return list(reversed(rows))  # yfinance returns newest first


def company_payloads(*, healthy_cash: bool = False, negative_equity: bool = False) -> Dict[str, Any]:
    """Revenue grows 10%/yr, operating margin 10% -> 16%. Cash conversion falls unless healthy_cash."""

    revenue = [100 * B, 110 * B, 121 * B, 133.1 * B]
    operating_income = [10 * B, 13.2 * B, 16.94 * B, 21.296 * B]
    net_income = [8 * B, 10 * B, 12 * B, 14 * B]
    ocf = [12 * B, 14 * B, 16 * B, 18 * B] if healthy_cash else [12 * B, 13 * B, 12 * B, 9 * B]
    capex = [-3 * B, -3.5 * B, -4 * B, -4.5 * B] if healthy_cash else [-3 * B, -3.5 * B, -5 * B, -8 * B]
    fcf = [o + c for o, c in zip(ocf, capex)]
    equity = [-5 * B, -4 * B, -3 * B, -2 * B] if negative_equity else [100 * B, 105 * B, 110 * B, 115 * B]
    prices = []
    start = date(2024, 11, 1)
    for day in range(300):
        prices.append({"date": (start + timedelta(days=day)).isoformat(), "open": 50.0, "high": 51.0, "low": 49.0,
                       "close": round(40.0 + day * 0.05, 4), "volume": 1000})
    return {
        "YFINANCE_COMPANY_INFO": [{
            "symbol": "TEST", "shortName": "Test Corp", "currency": "USD", "financialCurrency": "USD",
            "currentPrice": 55.0, "trailingPE": 20.0, "trailingPegRatio": 2.5, "operatingMargins": 0.16,
            "profitMargins": 0.105, "revenueGrowth": 0.1, "marketCap": 500 * B, "mostRecentQuarter": 1767139200,
        }],
        fixture_key("YFINANCE_FINANCIALS", {"quarterly": False}): _statement({
            "Total Revenue": revenue, "Operating Income": operating_income, "Net Income": net_income,
            "Gross Profit": [40 * B, 45 * B, 50 * B, 56 * B], "Diluted EPS": [1.0, 1.25, 1.5, 1.75],
            "EBITDA": [15 * B, 18 * B, 22 * B, 27 * B], "Pretax Income": [10 * B, 12.5 * B, 15 * B, 17.5 * B],
        }),
        fixture_key("YFINANCE_CASH_FLOW", {"quarterly": False}): _statement({
            "Operating Cash Flow": ocf, "Capital Expenditure": capex, "Free Cash Flow": fcf,
            "Change In Working Capital": [-1 * B, -2 * B, -4 * B, -7 * B],
        }),
        fixture_key("YFINANCE_BALANCE_SHEET", {"quarterly": False}): _statement({
            "Total Debt": [50 * B, 52 * B, 55 * B, 60 * B], "Stockholders Equity": equity,
            "Total Assets": [200 * B, 210 * B, 220 * B, 230 * B], "Accounts Receivable": [10 * B, 12 * B, 15 * B, 20 * B],
            "Current Assets": [60 * B, 62 * B, 64 * B, 66 * B], "Current Liabilities": [40 * B, 41 * B, 42 * B, 43 * B],
        }),
        fixture_key("YFINANCE_HISTORICAL_PRICES", {"interval": "1d"}): prices,
    }


class SyntheticTool:
    def __init__(self, toolset: "SyntheticToolSet", slug: str) -> None:
        self.toolset = toolset
        self.slug = slug

    def invoke(self, payload: Dict[str, Any]) -> Any:
        self.toolset.invocations.append({"slug": self.slug, "arguments": dict(payload)})
        key = fixture_key(self.slug, payload)
        if key not in self.toolset.payloads:
            raise RuntimeError(f"Synthetic data source has no {key}")
        return copy.deepcopy(self.toolset.payloads[key])


class SyntheticToolSet:
    def __init__(self, payloads: Optional[Dict[str, Any]] = None) -> None:
        self.payloads = payloads if payloads is not None else company_payloads()
        self.invocations: List[Dict[str, Any]] = []

    def get_tool(self, name: str) -> SyntheticTool:
        return SyntheticTool(self, name)


def make_request(text: str, ticker: str = "TEST", human: bool = False) -> HypothesisRequest:
    return HypothesisRequest(user_id="test", hypothesis_text=text, entities=[ticker],
                             time_horizon=TimeHorizon(start=date(2022, 1, 1), end=date(2025, 12, 31)),
                             requires_human_review=human)


class ScriptedLLM(BaseLLM):
    """LLM stub whose research capabilities are scripted per test."""

    def __init__(self, *, narratives: Optional[List[Dict[str, Any]]] = None, critique: Optional[Dict[str, Any]] = None,
                 tool_actions: Optional[List[Dict[str, Any]]] = None, plan: Optional[Dict[str, Any]] = None) -> None:
        self.narratives = list(narratives or [])
        self.critique = critique
        self.tool_actions = list(tool_actions or [])
        self.plan = plan
        self.briefings: List[Dict[str, Any]] = []

    def generate_data_plan(self, request): return ["x"]
    def generate_analysis_plan(self, request, data_overview): return ["x"]
    def generate_detailed_analysis(self, request, metrics_overview): return "Narrative."
    def generate_analysis_code(self, **kwargs): return "print('RESULT::{}')"

    def generate_report(self, request, metrics_overview, analysis_summary, artifact_paths):
        return {"executive_summary": "", "key_findings": [], "risks": [], "next_steps": []}

    def propose_research_plan(self, request, catalogue):
        return self.plan

    def choose_next_analysis_action(self, request, briefing):
        return self.tool_actions.pop(0) if self.tool_actions else None

    def critique_analysis(self, request, briefing):
        return self.critique

    def generate_research_narrative(self, request, briefing):
        self.briefings.append(briefing)
        if len(self.narratives) > 1:
            return self.narratives.pop(0)
        return self.narratives[0] if self.narratives else {"narrative": "The hypothesis is partially supported."}

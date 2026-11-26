"""Replay frozen Yahoo Finance payloads so research runs are reproducible offline.

A fixture file holds the *exact* payloads produced by :class:`YFinanceTool` for one ticker at a
known capture time (see ``scripts/freeze_benchmark_fixtures.py``). The benchmark suite uses these
so a run today and a run next year see identical inputs, independent of what Yahoo returns then.

Fixture layout (one JSON file per ticker)::

    {
      "ticker": "AAPL",
      "captured_at": "2026-11-27T10:00:00+00:00",
      "yfinance_version": "1.3.0",
      "responses": {
        "YFINANCE_FINANCIALS|annual": [...],
        "YFINANCE_HISTORICAL_PRICES|1d": [...],        # longest captured daily history
        "YFINANCE_NEWS": {"error": "YFDataException: ..."}  # captured failures replay as errors
      }
    }
"""
from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

PRICE_PERIOD_DAYS = {
    "1mo": 31,
    "3mo": 92,
    "6mo": 183,
    "1y": 366,
    "2y": 731,
    "3y": 1096,
    "5y": 1827,
}
STATEMENT_SLUGS = {"YFINANCE_FINANCIALS", "YFINANCE_BALANCE_SHEET", "YFINANCE_CASH_FLOW"}


class FixtureDataUnavailable(RuntimeError):
    """Raised when a fixture does not contain the requested dataset (mirrors a live API gap)."""


def fixture_key(slug: str, arguments: Dict[str, Any]) -> str:
    """Return the lookup key used inside a fixture file for a tool invocation."""

    if slug in STATEMENT_SLUGS:
        return f"{slug}|{'quarterly' if arguments.get('quarterly') else 'annual'}"
    if slug == "YFINANCE_HISTORICAL_PRICES":
        return f"{slug}|{arguments.get('interval', '1d')}"
    return slug


class FixtureTool:
    """Single dataset accessor backed by a fixture file."""

    def __init__(self, toolset: "FixtureToolSet", slug: str) -> None:
        self._toolset = toolset
        self._slug = slug

    def invoke(self, payload: Dict[str, Any]) -> Any:
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("Symbol is required")
        fixture = self._toolset.load(symbol)
        responses = fixture.get("responses", {})
        key = fixture_key(self._slug, payload)
        if key not in responses:
            raise FixtureDataUnavailable(
                f"Frozen fixture for {symbol} has no '{key}' dataset (captured {fixture.get('captured_at')})."
            )
        entry = responses[key]
        if isinstance(entry, dict) and set(entry.keys()) == {"error"}:
            raise FixtureDataUnavailable(f"Data source returned an error when captured: {entry['error']}")
        data = copy.deepcopy(entry)
        if self._slug == "YFINANCE_HISTORICAL_PRICES":
            data = _slice_prices(data, str(payload.get("period", "1y")))
        return data


class FixtureToolSet:
    """ToolSet implementation that serves frozen payloads from a fixture directory."""

    def __init__(self, fixture_dir: str | Path) -> None:
        self.fixture_dir = Path(fixture_dir).resolve()
        if not self.fixture_dir.is_dir():
            raise FileNotFoundError(f"Fixture directory not found: {self.fixture_dir}")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_tool(self, name: str) -> FixtureTool:
        return FixtureTool(self, name)

    def load(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if symbol not in self._cache:
            path = self.fixture_dir / f"{symbol}.json"
            if not path.exists():
                raise FixtureDataUnavailable(f"No frozen fixture available for symbol {symbol}.")
            self._cache[symbol] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[symbol]

    def source_metadata(self, slug: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Describe where a payload came from so snapshots stay honest about replayed data."""

        symbol = str(arguments.get("symbol") or "").upper()
        fixture = self.load(symbol)
        return {
            "mode": "frozen_fixture",
            "fixture_file": str(self.fixture_dir / f"{symbol}.json"),
            "captured_at": fixture.get("captured_at"),
            "yfinance_version": fixture.get("yfinance_version"),
        }


def _slice_prices(records: Any, period: str) -> Any:
    """Trim a captured daily history to the calendar window yfinance would have returned."""

    if not isinstance(records, list) or not records or period == "max":
        return records
    last = date.fromisoformat(str(records[-1]["date"]))
    if period == "ytd":
        start = date(last.year, 1, 1)
    elif period in PRICE_PERIOD_DAYS:
        start = last - timedelta(days=PRICE_PERIOD_DAYS[period])
    else:
        raise ValueError(f"Unsupported price period for fixture replay: {period}")
    return [row for row in records if date.fromisoformat(str(row["date"])) > start]

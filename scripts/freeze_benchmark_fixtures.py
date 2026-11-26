"""Capture live Yahoo Finance payloads into frozen benchmark fixtures.

Usage::

    python scripts/freeze_benchmark_fixtures.py --out benchmarks/v1/fixtures AAPL MSFT NVDA

Each ticker is captured through the same :class:`YFinanceTool` code path the agent uses live, so
replayed payloads are byte-for-byte what the agent would have seen at capture time. Failures are
captured too (``{"error": ...}``) so the benchmark reproduces real data gaps instead of hiding them.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypothesis_agent.orchestration.fixture_tools import fixture_key  # noqa: E402
from hypothesis_agent.orchestration.yfinance_tools import YFINANCE_VERSION, YFinanceTool  # noqa: E402

CAPTURE_PLAN = [
    ("YFINANCE_COMPANY_INFO", {}),
    ("YFINANCE_FINANCIALS", {"quarterly": False}),
    ("YFINANCE_FINANCIALS", {"quarterly": True}),
    ("YFINANCE_BALANCE_SHEET", {"quarterly": False}),
    ("YFINANCE_BALANCE_SHEET", {"quarterly": True}),
    ("YFINANCE_CASH_FLOW", {"quarterly": False}),
    ("YFINANCE_CASH_FLOW", {"quarterly": True}),
    ("YFINANCE_HISTORICAL_PRICES", {"period": "5y", "interval": "1d"}),
    ("YFINANCE_DIVIDENDS", {}),
    ("YFINANCE_NEWS", {}),
]


def _round_prices(records):
    for row in records:
        for key in ("open", "high", "low", "close"):
            if isinstance(row.get(key), float):
                row[key] = round(row[key], 4)
    return records


def capture(symbol: str) -> dict:
    responses = {}
    for slug, extra in CAPTURE_PLAN:
        arguments = {"symbol": symbol, **extra}
        key = fixture_key(slug, arguments)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payload = YFinanceTool(slug).invoke(arguments)
        except Exception as exc:  # captured deliberately: replay must reproduce the gap
            responses[key] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {key}: ERROR {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if slug == "YFINANCE_HISTORICAL_PRICES":
            payload = _round_prices(payload)
        if payload in ([], {}, None):
            responses[key] = {"error": "Data source returned an empty payload."}
            print(f"  {key}: EMPTY")
            continue
        responses[key] = payload
        size = len(payload) if isinstance(payload, (list, dict)) else 1
        print(f"  {key}: ok ({size})")
    return {
        "ticker": symbol,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": YFINANCE_VERSION,
        "capture_plan": [{"slug": slug, "arguments": extra} for slug, extra in CAPTURE_PLAN],
        "responses": responses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "v1" / "fixtures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for symbol in args.tickers:
        symbol = symbol.upper()
        print(f"Capturing {symbol}")
        fixture = capture(symbol)
        (out / f"{symbol}.json").write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

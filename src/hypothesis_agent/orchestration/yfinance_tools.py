"""Yahoo Finance tools for financial data retrieval using yfinance library."""
from __future__ import annotations

import math
from typing import Any, Dict, Protocol

import yfinance as yf

YFINANCE_VERSION = getattr(yf, "__version__", "unknown")

# Fields copied verbatim from ``yfinance.Ticker(symbol).info``. Additional fields were added for
# source attribution (currency, fiscal period markers) and valuation/leverage checks.
COMPANY_INFO_FIELDS = (
    "shortName",
    "longName",
    "sector",
    "industry",
    "website",
    "longBusinessSummary",
    "currency",
    "financialCurrency",
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "trailingEps",
    "forwardEps",
    "pegRatio",
    "trailingPegRatio",
    "priceToBook",
    "profitMargins",
    "operatingMargins",
    "grossMargins",
    "ebitdaMargins",
    "revenueGrowth",
    "earningsGrowth",
    "earningsQuarterlyGrowth",
    "returnOnAssets",
    "returnOnEquity",
    "totalRevenue",
    "ebitda",
    "freeCashflow",
    "operatingCashflow",
    "totalDebt",
    "totalCash",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "beta",
    "sharesOutstanding",
    "targetHighPrice",
    "targetLowPrice",
    "targetMeanPrice",
    "recommendationKey",
    "numberOfAnalystOpinions",
    "currentPrice",
    "previousClose",
    "mostRecentQuarter",
    "lastFiscalYearEnd",
)


def describe_retrieval(slug: str, arguments: Dict[str, Any]) -> str:
    """Return the exact yfinance call a tool slug performs, for source attribution."""

    symbol = arguments.get("symbol", "?")
    quarterly = bool(arguments.get("quarterly", False))
    base = f"yfinance.Ticker({symbol!r})"
    if slug == "YFINANCE_HISTORICAL_PRICES":
        params = [f"period={arguments.get('period', '1y')!r}", f"interval={arguments.get('interval', '1d')!r}"]
        return f"{base}.history({', '.join(params)})"
    attribute = {
        "YFINANCE_COMPANY_INFO": "info",
        "YFINANCE_FINANCIALS": "quarterly_financials" if quarterly else "financials",
        "YFINANCE_BALANCE_SHEET": "quarterly_balance_sheet" if quarterly else "balance_sheet",
        "YFINANCE_CASH_FLOW": "quarterly_cashflow" if quarterly else "cashflow",
        "YFINANCE_EARNINGS": "earnings / quarterly_earnings",
        "YFINANCE_RECOMMENDATIONS": "recommendations",
        "YFINANCE_NEWS": "news",
        "YFINANCE_HOLDERS": "major_holders / institutional_holders",
        "YFINANCE_DIVIDENDS": "dividends",
        "YFINANCE_SPLITS": "splits",
        "YFINANCE_OPTIONS": "options",
    }.get(slug, slug)
    return f"{base}.{attribute}"


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ToolHandle(Protocol):
    """Protocol describing the interface for a tool that can be invoked."""

    def invoke(self, payload: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        ...


class ToolSet(Protocol):
    """Protocol describing a minimal tool registry."""

    def get_tool(self, name: str) -> ToolHandle:  # pragma: no cover - interface
        ...


# Mapping of tool slugs to handler functions
TOOL_HANDLERS: Dict[str, str] = {
    "YFINANCE_COMPANY_INFO": "_get_company_info",
    "YFINANCE_HISTORICAL_PRICES": "_get_historical_prices",
    "YFINANCE_FINANCIALS": "_get_financials",
    "YFINANCE_BALANCE_SHEET": "_get_balance_sheet",
    "YFINANCE_CASH_FLOW": "_get_cash_flow",
    "YFINANCE_EARNINGS": "_get_earnings",
    "YFINANCE_RECOMMENDATIONS": "_get_recommendations",
    "YFINANCE_NEWS": "_get_news",
    "YFINANCE_HOLDERS": "_get_holders",
    "YFINANCE_DIVIDENDS": "_get_dividends",
    "YFINANCE_SPLITS": "_get_splits",
    "YFINANCE_OPTIONS": "_get_options",
}


class YFinanceTool:
    """Execute a specific Yahoo Finance data fetch."""

    def __init__(self, slug: str) -> None:
        self._slug = slug

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance based on the tool type."""
        handler_name = TOOL_HANDLERS.get(self._slug)
        if handler_name is None:
            raise ValueError(f"Unknown tool slug: {self._slug}")

        handler = getattr(self, handler_name)
        return handler(payload)

    def _get_ticker(self, payload: Dict[str, Any]) -> yf.Ticker:
        """Get a yfinance Ticker object from the payload."""
        symbol = payload.get("symbol")
        if not symbol:
            raise ValueError("Symbol is required")
        return yf.Ticker(symbol)

    def _get_company_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get company overview information."""
        ticker = self._get_ticker(payload)
        info = ticker.info
        record: Dict[str, Any] = {"symbol": payload.get("symbol")}
        for key in COMPANY_INFO_FIELDS:
            record[key] = info.get(key)
        record["52WeekHigh"] = info.get("fiftyTwoWeekHigh")
        record["52WeekLow"] = info.get("fiftyTwoWeekLow")
        return [record]

    def _get_historical_prices(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical price data."""
        ticker = self._get_ticker(payload)
        period = payload.get("period", "1y")  # Default 1 year
        interval = payload.get("interval", "1d")  # Default daily
        
        hist = ticker.history(period=period, interval=interval)
        
        # Convert to list of records (row-oriented) for better pandas compatibility
        records = []
        for date in hist.index:
            date_str = date.strftime("%Y-%m-%d")
            volume = _finite_or_none(hist.loc[date, "Volume"]) if "Volume" in hist.columns else None
            records.append({
                "date": date_str,
                "open": _finite_or_none(hist.loc[date, "Open"]) if "Open" in hist.columns else None,
                "high": _finite_or_none(hist.loc[date, "High"]) if "High" in hist.columns else None,
                "low": _finite_or_none(hist.loc[date, "Low"]) if "Low" in hist.columns else None,
                "close": _finite_or_none(hist.loc[date, "Close"]) if "Close" in hist.columns else None,
                "volume": int(volume) if volume is not None else None,
            })

        return records

    def _get_financials(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get income statement / financials data."""
        ticker = self._get_ticker(payload)
        quarterly = payload.get("quarterly", False)
        
        if quarterly:
            df = ticker.quarterly_financials
        else:
            df = ticker.financials
        
        # Convert DataFrame to list of records
        records = []
        if df is not None and not df.empty:
            df_t = df.transpose()
            for date, row in df_t.iterrows():
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                record = {"date": date_str}
                for idx, val in row.items():
                    record[str(idx)] = float(val) if val is not None and val == val else None
                records.append(record)
        
        return records

    def _get_balance_sheet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get balance sheet data."""
        ticker = self._get_ticker(payload)
        quarterly = payload.get("quarterly", False)
        
        if quarterly:
            df = ticker.quarterly_balance_sheet
        else:
            df = ticker.balance_sheet
        
        # Convert DataFrame to list of records
        records = []
        if df is not None and not df.empty:
            df_t = df.transpose()
            for date, row in df_t.iterrows():
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                record = {"date": date_str}
                for idx, val in row.items():
                    record[str(idx)] = float(val) if val is not None and val == val else None
                records.append(record)
        
        return records

    def _get_cash_flow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get cash flow statement data."""
        ticker = self._get_ticker(payload)
        quarterly = payload.get("quarterly", False)
        
        if quarterly:
            df = ticker.quarterly_cashflow
        else:
            df = ticker.cashflow
        
        # Convert DataFrame to list of records
        records = []
        if df is not None and not df.empty:
            df_t = df.transpose()
            for date, row in df_t.iterrows():
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                record = {"date": date_str}
                for idx, val in row.items():
                    record[str(idx)] = float(val) if val is not None and val == val else None
                records.append(record)
        
        return records

    def _get_earnings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get earnings data."""
        ticker = self._get_ticker(payload)
        
        earnings = ticker.earnings
        quarterly_earnings = ticker.quarterly_earnings
        
        result = {"annual": [], "quarterly": []}
        
        if earnings is not None and not earnings.empty:
            for idx in earnings.index:
                result["annual"].append({
                    "period": str(idx),
                    "Revenue": float(earnings.loc[idx, "Revenue"]) if "Revenue" in earnings.columns else None,
                    "Earnings": float(earnings.loc[idx, "Earnings"]) if "Earnings" in earnings.columns else None,
                })
        
        if quarterly_earnings is not None and not quarterly_earnings.empty:
            for idx in quarterly_earnings.index:
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                result["quarterly"].append({
                    "date": date_str,
                    "Revenue": float(quarterly_earnings.loc[idx, "Revenue"]) if "Revenue" in quarterly_earnings.columns else None,
                    "Earnings": float(quarterly_earnings.loc[idx, "Earnings"]) if "Earnings" in quarterly_earnings.columns else None,
                })
        
        return result

    def _get_recommendations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get analyst recommendations."""
        ticker = self._get_ticker(payload)
        
        recs = ticker.recommendations
        
        result = []
        if recs is not None and not recs.empty:
            for idx in recs.index[-10:]:  # Last 10 recommendations
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                row = recs.loc[idx]
                result.append({
                    "date": date_str,
                    "firm": row.get("Firm", None),
                    "toGrade": row.get("To Grade", None),
                    "fromGrade": row.get("From Grade", None),
                    "action": row.get("Action", None),
                })
        
        return {
            "symbol": payload.get("symbol"),
            "recommendations": result,
        }

    def _get_news(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get recent news for the ticker."""
        ticker = self._get_ticker(payload)
        
        news = ticker.news
        
        result = []
        if news:
            for item in news[:15]:  # Extended to 15 news items
                # yfinance>=0.2.50 nests article fields under "content"; older versions are flat.
                content = item.get("content") if isinstance(item.get("content"), dict) else None
                if content is not None:
                    provider = content.get("provider") or {}
                    canonical = content.get("canonicalUrl") or {}
                    result.append({
                        "title": content.get("title"),
                        "publisher": provider.get("displayName") if isinstance(provider, dict) else None,
                        "link": canonical.get("url") if isinstance(canonical, dict) else None,
                        "providerPublishTime": content.get("pubDate"),
                        "type": content.get("contentType"),
                    })
                    continue
                result.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "providerPublishTime": item.get("providerPublishTime"),
                    "type": item.get("type"),
                })

        return result

    def _get_holders(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get institutional and major holders data."""
        ticker = self._get_ticker(payload)
        
        result = {
            "major_holders": {},
            "institutional_holders": [],
        }
        
        major = ticker.major_holders
        if major is not None and not major.empty:
            for idx in major.index:
                result["major_holders"][str(major.loc[idx, 1])] = str(major.loc[idx, 0])
        
        inst = ticker.institutional_holders
        if inst is not None and not inst.empty:
            for idx in inst.index[:10]:  # Top 10
                row = inst.loc[idx]
                result["institutional_holders"].append({
                    "holder": row.get("Holder"),
                    "shares": int(row.get("Shares", 0)) if row.get("Shares") else None,
                    "value": float(row.get("Value", 0)) if row.get("Value") else None,
                    "pctHeld": float(row.get("% Out", 0)) if row.get("% Out") else None,
                })
        
        return {
            "symbol": payload.get("symbol"),
            "holders": result,
        }

    def _get_dividends(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get dividend history."""
        ticker = self._get_ticker(payload)
        
        divs = ticker.dividends
        
        result = {}
        if divs is not None and len(divs) > 0:
            for date in divs.index[-20:]:  # Last 20 dividends
                date_str = date.strftime("%Y-%m-%d")
                result[date_str] = float(divs.loc[date])
        
        return {
            "symbol": payload.get("symbol"),
            "dividends": result,
        }

    def _get_splits(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get stock split history."""
        ticker = self._get_ticker(payload)
        
        splits = ticker.splits
        
        result = {}
        if splits is not None and len(splits) > 0:
            for date in splits.index:
                date_str = date.strftime("%Y-%m-%d")
                result[date_str] = float(splits.loc[date])
        
        return {
            "symbol": payload.get("symbol"),
            "splits": result,
        }

    def _get_options(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get options expiration dates."""
        ticker = self._get_ticker(payload)
        
        try:
            options = ticker.options
            return {
                "symbol": payload.get("symbol"),
                "expiration_dates": list(options) if options else [],
            }
        except Exception:
            return {
                "symbol": payload.get("symbol"),
                "expiration_dates": [],
            }


class YFinanceToolSet:
    """Tool set providing access to Yahoo Finance data tools."""

    def get_tool(self, name: str) -> YFinanceTool:
        """Get a Yahoo Finance tool by name."""
        if name not in TOOL_HANDLERS:
            raise ValueError(f"Unknown Yahoo Finance tool: {name}. Available: {list(TOOL_HANDLERS.keys())}")
        return YFinanceTool(slug=name)

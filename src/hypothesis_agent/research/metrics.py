"""Canonical financial metric catalogue.

Every number JASPER reasons about is mapped to one canonical metric key. The catalogue says which
dataset a metric comes from, which raw yfinance fields may hold it, what unit it is in, which
evidence category it belongs to, and the words a narrative might use to refer to it. Claim
verification, critique and evaluation all rely on this single vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Evidence categories used by plans, critiques and the benchmark.
CATEGORIES = (
    "growth",
    "profitability",
    "cash_flow",
    "leverage",
    "liquidity",
    "valuation",
    "price_performance",
    "working_capital",
    "scale",
)

# dataset key -> (tool slug, extra tool arguments)
DATASETS: Dict[str, Tuple[str, Dict[str, object]]] = {
    "income_statement_annual": ("YFINANCE_FINANCIALS", {"quarterly": False}),
    "income_statement_quarterly": ("YFINANCE_FINANCIALS", {"quarterly": True}),
    "balance_sheet_annual": ("YFINANCE_BALANCE_SHEET", {"quarterly": False}),
    "balance_sheet_quarterly": ("YFINANCE_BALANCE_SHEET", {"quarterly": True}),
    "cash_flow_annual": ("YFINANCE_CASH_FLOW", {"quarterly": False}),
    "cash_flow_quarterly": ("YFINANCE_CASH_FLOW", {"quarterly": True}),
    "company_info": ("YFINANCE_COMPANY_INFO", {}),
    "price_history": ("YFINANCE_HISTORICAL_PRICES", {"interval": "1d"}),
    "news": ("YFINANCE_NEWS", {}),
    "dividends": ("YFINANCE_DIVIDENDS", {}),
}


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    unit: str  # USD | ratio | USD/share | shares | x | percent | days
    category: str
    source: str  # income_statement | balance_sheet | cash_flow | company_info | price_history | derived
    fields: Tuple[str, ...] = ()
    synonyms: Tuple[str, ...] = ()
    formula: Optional[str] = None
    # Direction in which the metric is "better" for the company (used for health assessments).
    higher_is_better: Optional[bool] = None
    period_type: Optional[str] = None  # forced period type for company_info fields


def _m(*args, **kwargs) -> MetricDefinition:
    return MetricDefinition(*args, **kwargs)


_DEFINITIONS: List[MetricDefinition] = [
    # Income statement ------------------------------------------------------------------------
    _m("total_revenue", "Total revenue", "USD", "growth", "income_statement",
       ("Total Revenue", "Operating Revenue"), ("revenue", "revenues", "sales", "top line", "top-line"), higher_is_better=True),
    _m("cost_of_revenue", "Cost of revenue", "USD", "profitability", "income_statement",
       ("Cost Of Revenue", "Reconciled Cost Of Revenue"), ("cost of revenue", "cost of sales", "cogs")),
    _m("gross_profit", "Gross profit", "USD", "profitability", "income_statement",
       ("Gross Profit",), ("gross profit",), higher_is_better=True),
    _m("operating_income", "Operating income", "USD", "profitability", "income_statement",
       ("Operating Income", "Total Operating Income As Reported"), ("operating income", "operating profit"), higher_is_better=True),
    _m("operating_expense", "Operating expense", "USD", "profitability", "income_statement",
       ("Operating Expense",), ("operating expense", "operating expenses", "opex")),
    _m("research_and_development", "R&D expense", "USD", "profitability", "income_statement",
       ("Research And Development",), ("r&d", "research and development")),
    _m("ebitda", "EBITDA", "USD", "profitability", "income_statement",
       ("EBITDA", "Normalized EBITDA"), ("ebitda",), higher_is_better=True),
    _m("ebit", "EBIT", "USD", "profitability", "income_statement", ("EBIT",), ("ebit",), higher_is_better=True),
    _m("interest_expense", "Interest expense", "USD", "leverage", "income_statement",
       ("Interest Expense", "Interest Expense Non Operating"), ("interest expense",), higher_is_better=False),
    _m("pretax_income", "Pre-tax income", "USD", "profitability", "income_statement",
       ("Pretax Income",), ("pretax income", "pre-tax income"), higher_is_better=True),
    _m("net_income", "Net income", "USD", "profitability", "income_statement",
       ("Net Income", "Net Income Common Stockholders"), ("net income", "net profit", "net earnings", "net loss", "earnings"), higher_is_better=True),
    _m("diluted_eps", "Diluted EPS", "USD/share", "profitability", "income_statement",
       ("Diluted EPS",), ("diluted eps", "eps", "earnings per share"), higher_is_better=True),
    _m("basic_eps", "Basic EPS", "USD/share", "profitability", "income_statement", ("Basic EPS",), ("basic eps",)),
    _m("diluted_shares", "Diluted average shares", "shares", "scale", "income_statement",
       ("Diluted Average Shares",), ("diluted shares", "share count")),
    # Balance sheet ---------------------------------------------------------------------------
    _m("total_assets", "Total assets", "USD", "scale", "balance_sheet", ("Total Assets",), ("total assets",)),
    _m("total_liabilities", "Total liabilities", "USD", "leverage", "balance_sheet",
       ("Total Liabilities Net Minority Interest", "Total Liabilities"), ("total liabilities",), higher_is_better=False),
    _m("stockholders_equity", "Stockholders' equity", "USD", "leverage", "balance_sheet",
       ("Stockholders Equity", "Common Stock Equity"), ("shareholders' equity", "stockholders' equity", "shareholder equity", "book value")),
    _m("total_debt", "Total debt", "USD", "leverage", "balance_sheet", ("Total Debt",), ("total debt", "debt", "borrowings"), higher_is_better=False),
    _m("long_term_debt", "Long-term debt", "USD", "leverage", "balance_sheet", ("Long Term Debt",), ("long-term debt", "long term debt"), higher_is_better=False),
    _m("net_debt", "Net debt", "USD", "leverage", "balance_sheet", ("Net Debt",), ("net debt",), higher_is_better=False),
    _m("cash_and_equivalents", "Cash and equivalents", "USD", "liquidity", "balance_sheet",
       ("Cash And Cash Equivalents",), ("cash and equivalents", "cash and cash equivalents"), higher_is_better=True),
    _m("cash_and_short_term_investments", "Cash and short-term investments", "USD", "liquidity", "balance_sheet",
       ("Cash Cash Equivalents And Short Term Investments",), ("cash and short-term investments",), higher_is_better=True),
    _m("current_assets", "Current assets", "USD", "liquidity", "balance_sheet", ("Current Assets",), ("current assets",)),
    _m("current_liabilities", "Current liabilities", "USD", "liquidity", "balance_sheet", ("Current Liabilities",), ("current liabilities",)),
    _m("accounts_receivable", "Accounts receivable", "USD", "working_capital", "balance_sheet",
       ("Accounts Receivable", "Receivables"), ("accounts receivable", "receivables")),
    _m("inventory", "Inventory", "USD", "working_capital", "balance_sheet", ("Inventory",), ("inventory", "inventories")),
    _m("working_capital", "Working capital", "USD", "working_capital", "balance_sheet", ("Working Capital",), ("working capital",)),
    # Cash flow -------------------------------------------------------------------------------
    _m("operating_cash_flow", "Operating cash flow", "USD", "cash_flow", "cash_flow",
       ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"), ("operating cash flow", "cash from operations", "ocf"), higher_is_better=True),
    _m("capital_expenditure", "Capital expenditure", "USD", "cash_flow", "cash_flow",
       ("Capital Expenditure",), ("capital expenditure", "capital expenditures", "capex")),
    _m("free_cash_flow", "Free cash flow", "USD", "cash_flow", "cash_flow",
       ("Free Cash Flow",), ("free cash flow", "fcf"), higher_is_better=True),
    _m("change_in_working_capital", "Change in working capital", "USD", "working_capital", "cash_flow",
       ("Change In Working Capital",), ("change in working capital",)),
    _m("change_in_receivables", "Change in receivables", "USD", "working_capital", "cash_flow",
       ("Change In Receivables", "Changes In Account Receivables"), ("change in receivables",)),
    _m("change_in_inventory", "Change in inventory", "USD", "working_capital", "cash_flow", ("Change In Inventory",), ("change in inventory",)),
    _m("stock_based_compensation", "Stock-based compensation", "USD", "cash_flow", "cash_flow",
       ("Stock Based Compensation",), ("stock-based compensation", "stock based compensation", "sbc")),
    _m("share_repurchases", "Share repurchases", "USD", "cash_flow", "cash_flow",
       ("Repurchase Of Capital Stock",), ("buybacks", "share repurchases", "repurchases")),
    _m("dividends_paid", "Dividends paid", "USD", "cash_flow", "cash_flow", ("Cash Dividends Paid",), ("dividends paid",)),
    # Company info (point-in-time / trailing twelve months as reported by Yahoo) ---------------
    _m("market_cap", "Market capitalisation", "USD", "valuation", "company_info", ("marketCap",), ("market cap", "market capitalization", "market capitalisation"), period_type="point_in_time"),
    _m("enterprise_value", "Enterprise value", "USD", "valuation", "company_info", ("enterpriseValue",), ("enterprise value",), period_type="point_in_time"),
    _m("trailing_pe", "Trailing P/E", "x", "valuation", "company_info", ("trailingPE",), ("trailing p/e", "p/e", "pe ratio", "price-to-earnings", "price to earnings", "earnings multiple"), period_type="point_in_time"),
    _m("forward_pe", "Forward P/E", "x", "valuation", "company_info", ("forwardPE",), ("forward p/e", "forward pe", "forward earnings multiple"), period_type="point_in_time"),
    _m("peg_ratio", "PEG ratio (Yahoo, trailing)", "x", "valuation", "company_info", ("trailingPegRatio", "pegRatio"), ("peg ratio", "peg"), period_type="point_in_time"),
    _m("price_to_book", "Price to book", "x", "valuation", "company_info", ("priceToBook",), ("price-to-book", "price to book", "p/b"), period_type="point_in_time"),
    _m("ttm_operating_margin", "Operating margin (TTM, Yahoo)", "ratio", "profitability", "company_info", ("operatingMargins",), ("ttm operating margin",), higher_is_better=True, period_type="ttm"),
    _m("ttm_profit_margin", "Net profit margin (TTM, Yahoo)", "ratio", "profitability", "company_info", ("profitMargins",), ("ttm profit margin",), higher_is_better=True, period_type="ttm"),
    _m("ttm_gross_margin", "Gross margin (TTM, Yahoo)", "ratio", "profitability", "company_info", ("grossMargins",), ("ttm gross margin",), higher_is_better=True, period_type="ttm"),
    _m("ttm_revenue", "Revenue (TTM, Yahoo)", "USD", "growth", "company_info", ("totalRevenue",), ("ttm revenue", "trailing revenue"), period_type="ttm"),
    _m("quarterly_revenue_growth_yoy", "Quarterly revenue growth YoY (Yahoo)", "ratio", "growth", "company_info", ("revenueGrowth",), ("quarterly revenue growth",), higher_is_better=True, period_type="point_in_time"),
    _m("quarterly_earnings_growth_yoy", "Quarterly earnings growth YoY (Yahoo)", "ratio", "growth", "company_info", ("earningsGrowth",), ("quarterly earnings growth",), higher_is_better=True, period_type="point_in_time"),
    _m("return_on_equity", "Return on equity (TTM, Yahoo)", "ratio", "profitability", "company_info", ("returnOnEquity",), ("return on equity", "roe"), higher_is_better=True, period_type="ttm"),
    _m("return_on_assets", "Return on assets (TTM, Yahoo)", "ratio", "profitability", "company_info", ("returnOnAssets",), ("return on assets", "roa"), higher_is_better=True, period_type="ttm"),
    _m("ttm_free_cash_flow", "Free cash flow (TTM, Yahoo)", "USD", "cash_flow", "company_info", ("freeCashflow",), ("ttm free cash flow",), higher_is_better=True, period_type="ttm"),
    _m("ttm_operating_cash_flow", "Operating cash flow (TTM, Yahoo)", "USD", "cash_flow", "company_info", ("operatingCashflow",), ("ttm operating cash flow",), higher_is_better=True, period_type="ttm"),
    _m("yahoo_debt_to_equity", "Debt to equity (Yahoo, %)", "percent", "leverage", "company_info", ("debtToEquity",), (), higher_is_better=False, period_type="point_in_time"),
    _m("current_price", "Share price", "USD/share", "price_performance", "company_info", ("currentPrice",), ("share price", "stock price", "current price"), period_type="point_in_time"),
    _m("target_mean_price", "Analyst mean target price", "USD/share", "valuation", "company_info", ("targetMeanPrice",), ("price target", "target price"), period_type="point_in_time"),
    _m("trailing_eps", "Trailing EPS (Yahoo)", "USD/share", "profitability", "company_info", ("trailingEps",), ("trailing eps",), period_type="ttm"),
    _m("forward_eps", "Forward EPS (Yahoo)", "USD/share", "profitability", "company_info", ("forwardEps",), ("forward eps",), period_type="point_in_time"),
    _m("beta", "Beta", "x", "price_performance", "company_info", ("beta",), ("beta",), period_type="point_in_time"),
    # Derived ---------------------------------------------------------------------------------
    _m("gross_margin", "Gross margin", "ratio", "profitability", "derived", (), ("gross margin", "gross margins"), "Gross Profit / Total Revenue", True),
    _m("operating_margin", "Operating margin", "ratio", "profitability", "derived", (), ("operating margin", "operating margins", "op margin"), "Operating Income / Total Revenue", True),
    _m("net_margin", "Net margin", "ratio", "profitability", "derived", (), ("net margin", "net margins", "profit margin", "profit margins", "net profit margin"), "Net Income / Total Revenue", True),
    _m("fcf_margin", "Free cash flow margin", "ratio", "cash_flow", "derived", (), ("fcf margin", "free cash flow margin"), "Free Cash Flow / Total Revenue", True),
    _m("free_cash_flow_computed", "Free cash flow (computed)", "USD", "cash_flow", "derived", (), (), "Operating Cash Flow + Capital Expenditure", True),
    _m("cash_conversion", "Cash conversion (OCF / net income)", "ratio", "cash_flow", "derived", (), ("cash conversion", "cash conversion ratio"), "Operating Cash Flow / Net Income", True),
    _m("fcf_conversion", "FCF conversion (FCF / net income)", "ratio", "cash_flow", "derived", (), ("fcf conversion", "free cash flow conversion"), "Free Cash Flow / Net Income", True),
    _m("capex_intensity", "Capex intensity", "ratio", "cash_flow", "derived", (), ("capex intensity", "capex to revenue", "capital intensity"), "-Capital Expenditure / Total Revenue", None),
    _m("debt_to_equity", "Debt to equity", "ratio", "leverage", "derived", (), ("debt-to-equity", "debt to equity", "d/e", "leverage ratio"), "Total Debt / Stockholders' Equity", False),
    _m("debt_to_ebitda", "Debt to EBITDA", "x", "leverage", "derived", (), ("debt-to-ebitda", "debt to ebitda", "leverage multiple"), "Total Debt / EBITDA", False),
    _m("liabilities_to_assets", "Liabilities to assets", "ratio", "leverage", "derived", (), ("liabilities to assets",), "Total Liabilities / Total Assets", False),
    _m("current_ratio", "Current ratio", "x", "liquidity", "derived", (), ("current ratio",), "Current Assets / Current Liabilities", True),
    _m("interest_coverage", "Interest coverage", "x", "leverage", "derived", (), ("interest coverage",), "Operating Income / Interest Expense", True),
    _m("receivables_days", "Days sales outstanding", "days", "working_capital", "derived", (), ("days sales outstanding", "dso", "receivable days"), "Accounts Receivable / Total Revenue x 365", False),
    _m("return_on_equity_computed", "Return on equity (computed)", "ratio", "profitability", "derived", (), (), "Net Income / Stockholders' Equity", True),
    _m("pe_computed", "P/E (price / latest FY diluted EPS)", "x", "valuation", "derived", (), (), "Share Price / Diluted EPS", None),
    _m("peg_computed", "PEG (P/E / EPS growth %)", "x", "valuation", "derived", (), (), "P/E / (EPS growth x 100)", None),
    _m("price_return", "Price return", "ratio", "price_performance", "derived", (), ("share price return", "stock return", "price return", "stock is up", "shares rose", "shares fell"), "Close(end) / Close(start) - 1", True),
    _m("annualized_volatility", "Annualised volatility", "ratio", "price_performance", "derived", (), ("annualized volatility", "annualised volatility", "price volatility", "stock volatility", "share price volatility"), "stdev(daily log returns) x sqrt(252)", False),
    _m("max_drawdown", "Maximum drawdown", "ratio", "price_performance", "derived", (), ("drawdown", "max drawdown"), "min(Close / running max Close) - 1", False),
    _m("close_price", "Closing price", "USD/share", "price_performance", "price_history", ("close",), ("closing price",)),
]

METRICS: Dict[str, MetricDefinition] = {definition.key: definition for definition in _DEFINITIONS}

# Ratio metrics computed from two statement metrics: key -> (numerator, denominator, scale, sign)
RATIO_RECIPES: Dict[str, Tuple[str, str, float, float]] = {
    "gross_margin": ("gross_profit", "total_revenue", 1.0, 1.0),
    "operating_margin": ("operating_income", "total_revenue", 1.0, 1.0),
    "net_margin": ("net_income", "total_revenue", 1.0, 1.0),
    "fcf_margin": ("free_cash_flow", "total_revenue", 1.0, 1.0),
    "cash_conversion": ("operating_cash_flow", "net_income", 1.0, 1.0),
    "fcf_conversion": ("free_cash_flow", "net_income", 1.0, 1.0),
    "capex_intensity": ("capital_expenditure", "total_revenue", 1.0, -1.0),
    "debt_to_equity": ("total_debt", "stockholders_equity", 1.0, 1.0),
    "debt_to_ebitda": ("total_debt", "ebitda", 1.0, 1.0),
    "liabilities_to_assets": ("total_liabilities", "total_assets", 1.0, 1.0),
    "current_ratio": ("current_assets", "current_liabilities", 1.0, 1.0),
    "interest_coverage": ("operating_income", "interest_expense", 1.0, 1.0),
    "receivables_days": ("accounts_receivable", "total_revenue", 365.0, 1.0),
    "return_on_equity_computed": ("net_income", "stockholders_equity", 1.0, 1.0),
}

SOURCE_DATASET_PREFIX = {
    "income_statement": "income_statement",
    "balance_sheet": "balance_sheet",
    "cash_flow": "cash_flow",
}


def get_metric(key: str) -> MetricDefinition:
    """Return the definition for a metric key, including growth/trend variants of base metrics."""

    if key in METRICS:
        return METRICS[key]
    for suffix, label_suffix, unit in (
        ("_growth_yoy", "growth (YoY)", "ratio"),
        ("_growth_qoq", "growth (QoQ)", "ratio"),
        ("_cagr", "CAGR", "ratio"),
        ("_change", "change", None),
        ("_trend", "trend", None),
    ):
        if key.endswith(suffix):
            base_key = key[: -len(suffix)]
            # Variants can stack, e.g. total_revenue_growth_yoy_trend (trend of a growth rate).
            base = METRICS.get(base_key) or (get_metric(base_key) if suffix in {"_trend", "_change"} and is_known_metric(base_key) else None)
            if base is None:
                break
            resolved_unit = unit or ("ratio_points" if base.unit == "ratio" else base.unit)
            if suffix == "_trend":
                resolved_unit = "ratio_points" if base.unit == "ratio" else "ratio"
            return MetricDefinition(
                key=key,
                label=f"{base.label} {label_suffix}",
                unit=resolved_unit,
                category=base.category if suffix != "_growth_yoy" or base.category != "scale" else "growth",
                source="derived",
                synonyms=tuple(f"{syn} growth" for syn in base.synonyms) if "growth" in suffix else (),
                higher_is_better=base.higher_is_better,
            )
    raise KeyError(f"Unknown metric '{key}'")


def is_known_metric(key: str) -> bool:
    try:
        get_metric(key)
    except KeyError:
        return False
    return True


def metric_category(key: str) -> str:
    try:
        definition = get_metric(key)
    except KeyError:
        return "unknown"
    stripped = key[: -len("_trend")] if key.endswith("_trend") else key[: -len("_change")] if key.endswith("_change") else key
    if stripped.endswith(("_growth_yoy", "_growth_qoq", "_cagr")):
        return "growth"
    return definition.category


def dataset_for_metric(key: str, frequency: str = "annual") -> Optional[str]:
    """Return the dataset that must be collected for a (raw) metric."""

    definition = METRICS.get(key)
    if definition is None:
        return None
    if definition.source == "company_info":
        return "company_info"
    if definition.source == "price_history":
        return "price_history"
    if definition.source in SOURCE_DATASET_PREFIX:
        return f"{SOURCE_DATASET_PREFIX[definition.source]}_{frequency}"
    return None


def raw_dependencies(key: str) -> List[str]:
    """Return the raw metrics a (possibly derived) metric ultimately depends on."""

    for suffix in ("_growth_yoy", "_growth_qoq", "_cagr", "_change", "_trend"):
        if key.endswith(suffix) and is_known_metric(key[: -len(suffix)]):
            return raw_dependencies(key[: -len(suffix)])
    if key in RATIO_RECIPES:
        numerator, denominator, _, _ = RATIO_RECIPES[key]
        deps = raw_dependencies(numerator) + raw_dependencies(denominator)
        return list(dict.fromkeys(deps))
    if key == "free_cash_flow_computed":
        return ["operating_cash_flow", "capital_expenditure"]
    if key == "pe_computed":
        return ["current_price", "diluted_eps"]
    if key == "peg_computed":
        return ["trailing_pe", "diluted_eps"]
    if key in {"price_return", "annualized_volatility", "max_drawdown"}:
        return ["close_price"]
    if key in METRICS and METRICS[key].source != "derived":
        return [key]
    return []

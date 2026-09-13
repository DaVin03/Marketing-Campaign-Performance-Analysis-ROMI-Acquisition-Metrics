"""
KPI formulas
Each function takes aggregated (summed) columns, not row-by-row values --
see romi_bias.py for why that distinction matters.
"""

import numpy as np
import pandas as pd


def _safe_div(numerator, denominator):
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan) if isinstance(result, pd.Series) else result


def romi(revenue, spend):
    return _safe_div(revenue - spend, spend)


def ctr(clicks, impressions):
    return _safe_div(clicks, impressions)


def conv_1_visitor_to_lead(leads, clicks):
    return _safe_div(leads, clicks)


def conv_2_lead_to_sale(orders, leads):
    return _safe_div(orders, leads)


def aov(revenue, orders):
    return _safe_div(revenue, orders)


def cpc(spend, clicks):
    return _safe_div(spend, clicks)


def cpl(spend, leads):
    return _safe_div(spend, leads)


def cac(spend, orders):
    return _safe_div(spend, orders)


def gross_profit(revenue, spend):
    return revenue - spend


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate KPI block for a given (already grouped/summed) slice."""
    revenue = df["revenue"].sum()
    spend = df["mark_spent"].sum()
    clicks = df["clicks"].sum()
    impressions = df["impressions"].sum()
    leads = df["leads"].sum()
    orders = df["orders"].sum()

    return {
        "romi": romi(revenue, spend),
        "ctr": ctr(clicks, impressions),
        "conv_1": conv_1_visitor_to_lead(leads, clicks),
        "conv_2": conv_2_lead_to_sale(orders, leads),
        "aov": aov(revenue, orders),
        "cpc": cpc(spend, clicks),
        "cpl": cpl(spend, leads),
        "cac": cac(spend, orders),
        "gross_profit": gross_profit(revenue, spend),
    }

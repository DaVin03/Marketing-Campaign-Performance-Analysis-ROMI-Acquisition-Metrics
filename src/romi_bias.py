"""
Step 2: The ROMI Bias

Common pitfall: compute ROMI per day, then average those daily ROMIs.
This is wrong because it weights every day equally regardless of how
much was actually spent that day -- a day with 50 rupees spent and a
lucky 300% ROMI counts as much as a day with 50,000 rupees spent at 8%.

Correct approach: sum revenue and sum spend first (per campaign, or
overall), then compute ROMI once on the totals.
"""

import pandas as pd

from src import metrics


def naive_avg_of_daily_romi(df: pd.DataFrame, group_col: str | None = None) -> pd.Series | float:
    daily_romi = metrics.romi(df["revenue"], df["mark_spent"])
    if group_col is None:
        return daily_romi.mean()
    return daily_romi.groupby(df[group_col]).mean()


def correct_aggregated_romi(df: pd.DataFrame, group_col: str | None = None) -> pd.Series | float:
    if group_col is None:
        return metrics.romi(df["revenue"].sum(), df["mark_spent"].sum())

    grouped = df.groupby(group_col).agg(revenue=("revenue", "sum"), mark_spent=("mark_spent", "sum"))
    return metrics.romi(grouped["revenue"], grouped["mark_spent"])


def compare(df: pd.DataFrame, group_col: str = "campaign_name") -> pd.DataFrame:
    """Side-by-side table showing how far off the naive average is."""
    naive = naive_avg_of_daily_romi(df, group_col)
    correct = correct_aggregated_romi(df, group_col)

    out = pd.DataFrame({"naive_avg_romi": naive, "correct_romi": correct})
    out["bias"] = out["naive_avg_romi"] - out["correct_romi"]
    return out.sort_values("bias", ascending=False)

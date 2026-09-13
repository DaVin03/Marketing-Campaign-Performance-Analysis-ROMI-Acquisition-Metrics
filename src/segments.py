"""
Step 4 + 5: Normalized Time Analysis & Segment Evaluation

Note: the README's Data Dictionary doesn't list a "Tier" / geo column.
Looking at the raw data, tier (Tier 1 / Tier 2) is actually baked into
campaign_name (e.g. "facebook_tier1", "facebOOK_tier2" -- casing is
inconsistent in the source data), not a separate field. Extracted here
with a case-insensitive regex rather than fixed in the README, since
that mismatch is exactly the kind of thing you find once you actually
open the data instead of trusting the doc.
"""

import re

import pandas as pd

from src import metrics


def extract_tier(campaign_name: str) -> str:
    match = re.search(r"tier\s*([12])", campaign_name, flags=re.IGNORECASE)
    return f"tier{match.group(1)}" if match else "unknown"


def add_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["campaign_name"].apply(extract_tier)
    return df


def weekday_vs_weekend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend"] = df["c_date"].dt.dayofweek >= 5
    df["day_bucket"] = df["is_weekend"].map({True: "weekend", False: "weekday"})

    rows = {}
    for bucket, group in df.groupby("day_bucket"):
        rows[bucket] = metrics.summarize(group)
    return pd.DataFrame(rows).T


def by_channel(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for category, group in df.groupby("category"):
        rows[category] = metrics.summarize(group)
    return pd.DataFrame(rows).T.sort_values("romi", ascending=False)


def by_geo_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = add_tier_column(df)
    rows = {}
    for tier, group in df.groupby("tier"):
        rows[tier] = metrics.summarize(group)
    return pd.DataFrame(rows).T.sort_values("romi", ascending=False)

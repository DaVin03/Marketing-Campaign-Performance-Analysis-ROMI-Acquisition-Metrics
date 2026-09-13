"""
Step 3: Outlier Handling

Two separate buckets, handled differently:
- Technical errors: physically impossible rows (clicks > impressions,
  leads > clicks, orders > leads -- i.e. a "conversion rate" over 100%).
  These get removed, they're data-entry/tracking bugs, not signal.
- Top performers: extreme revenue days. These are NOT discarded --
  they get pulled into their own subset so we can study what made
  them work, instead of treating them as noise.
"""

import pandas as pd


def flag_technical_errors(df: pd.DataFrame) -> pd.Series:
    return (
        (df["clicks"] > df["impressions"])
        | (df["leads"] > df["clicks"])
        | (df["orders"] > df["leads"])
    )


def remove_technical_errors(df: pd.DataFrame) -> pd.DataFrame:
    bad = flag_technical_errors(df)
    return df.loc[~bad].copy()


def isolate_top_performers(df: pd.DataFrame, revenue_quantile: float = 0.99) -> pd.DataFrame:
    threshold = df["revenue"].quantile(revenue_quantile)
    return df.loc[df["revenue"] >= threshold].copy()

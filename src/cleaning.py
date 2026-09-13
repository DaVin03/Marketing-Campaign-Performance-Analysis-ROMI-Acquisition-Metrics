"""
Step 1: Data Cleaning & Feature Engineering

Loads the raw daily campaign log and computes the daily operational
metrics (CPC, CTR). Rows with marketing spend but zero clicks (or zero
impressions) would otherwise produce division-by-zero -> those become
NaN rather than inf, so they don't silently poison later aggregates.
"""

import numpy as np
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["c_date"])
    return df


def add_daily_operational_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ctr"] = df["clicks"] / df["impressions"]
    df["cpc"] = df["mark_spent"] / df["clicks"]

    df["ctr"] = df["ctr"].replace([np.inf, -np.inf], np.nan)
    df["cpc"] = df["cpc"].replace([np.inf, -np.inf], np.nan)

    return df


def clean(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = add_daily_operational_metrics(df)
    return df

"""
Orchestrator: runs the full analytical flow described in README.md
(Methodology) and prints the answers to the 6 Business Questions.

Usage: python main.py
"""

import pandas as pd

from src import cleaning, metrics, outliers, plots, romi_bias, segments

DATA_PATH = "data/Marketing.csv"
OUTPUT_DIR = "output"


def main():
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    # --- 1. Data Cleaning & Feature Engineering ---
    df = cleaning.clean(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # --- 3. Outlier Handling (done early, everything downstream uses clean data) ---
    df_clean = outliers.remove_technical_errors(df)
    dropped = len(df) - len(df_clean)
    print(f"Dropped {dropped} rows as technical errors (impossible click/lead/order counts)")

    top_performers = outliers.isolate_top_performers(df_clean)
    top_performers.to_csv(f"{OUTPUT_DIR}/top_performers.csv", index=False)
    print(f"Isolated {len(top_performers)} top-performer rows (>= p99 revenue) -> {OUTPUT_DIR}/top_performers.csv")

    # --- Business Question 1: overall ROMI + per-campaign ROMI ---
    overall_romi = romi_bias.correct_aggregated_romi(df_clean)
    print(f"\n[Q1] Overall ROMI (correct, aggregated): {overall_romi:.4f}")

    romi_comparison = romi_bias.compare(df_clean, group_col="campaign_name")
    romi_comparison.to_csv(f"{OUTPUT_DIR}/romi_bias_comparison.csv")
    print("[Q1] Per-campaign ROMI (naive avg-of-daily vs correct aggregated) -> "
          f"{OUTPUT_DIR}/romi_bias_comparison.csv")
    print(romi_comparison.head())
    plots.romi_bias(romi_comparison)

    # --- Business Question 2: dates with highest spend / highest revenue ---
    by_date = df_clean.groupby("c_date")[["mark_spent", "revenue"]].sum()
    top_spend_date = by_date["mark_spent"].idxmax()
    top_revenue_date = by_date["revenue"].idxmax()
    print(f"\n[Q2] Highest spend date: {top_spend_date.date()} ({by_date['mark_spent'].max():,.2f})")
    print(f"[Q2] Highest revenue date: {top_revenue_date.date()} ({by_date['revenue'].max():,.2f})")
    plots.spend_vs_revenue_over_time(df_clean)

    # --- Business Question 3: conversion rate extremes + AOV ---
    daily_conv = pd.DataFrame({
        "conv_1": metrics.conv_1_visitor_to_lead(df_clean["leads"], df_clean["clicks"]),
        "conv_2": metrics.conv_2_lead_to_sale(df_clean["orders"], df_clean["leads"]),
    })
    daily_conv["c_date"] = df_clean["c_date"].values
    print(f"\n[Q3] Conv. 1 (visitor->lead) range: {daily_conv['conv_1'].min():.4f} - {daily_conv['conv_1'].max():.4f}")
    print(f"[Q3] Conv. 2 (lead->sale) range: {daily_conv['conv_2'].min():.4f} - {daily_conv['conv_2'].max():.4f}")
    overall_aov = metrics.aov(df_clean["revenue"].sum(), df_clean["orders"].sum())
    print(f"[Q3] Overall AOV: {overall_aov:,.2f}")
    plots.conversion_rates_and_aov(df_clean, metrics)

    # --- Business Question 4: weekday vs weekend, normalized ---
    time_analysis = segments.weekday_vs_weekend(df_clean)
    time_analysis.to_csv(f"{OUTPUT_DIR}/weekday_vs_weekend.csv")
    print(f"\n[Q4] Weekday vs weekend (revenue, CAC, conversions) -> {OUTPUT_DIR}/weekday_vs_weekend.csv")
    print(time_analysis[["romi", "cac", "conv_1", "conv_2", "gross_profit"]])
    plots.weekday_vs_weekend(time_analysis)

    # --- Business Question 5: best channels ---
    channel_analysis = segments.by_channel(df_clean)
    channel_analysis.to_csv(f"{OUTPUT_DIR}/by_channel.csv")
    print(f"\n[Q5] Channel performance (sorted by ROMI) -> {OUTPUT_DIR}/by_channel.csv")
    print(channel_analysis[["romi", "cac", "gross_profit"]])
    plots.by_channel(channel_analysis)

    # --- Business Question 6: Tier 1 vs Tier 2 ---
    geo_analysis = segments.by_geo_tier(df_clean)
    geo_analysis.to_csv(f"{OUTPUT_DIR}/by_geo_tier.csv")
    print(f"\n[Q6] Geo tier performance (sorted by ROMI) -> {OUTPUT_DIR}/by_geo_tier.csv")
    print(geo_analysis[["romi", "cac", "gross_profit"]])
    plots.by_geo_tier(geo_analysis)

    print(f"\nPlots saved to {plots.PLOT_DIR}/")


if __name__ == "__main__":
    main()

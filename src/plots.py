"""
Charts for the portfolio 
"""

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = "output/plots"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{name}.png", dpi=150)
    plt.close(fig)


def romi_bias(comparison: pd.DataFrame):
    """Q1: naive avg-of-daily ROMI vs correct aggregated ROMI, per campaign."""
    fig, ax = plt.subplots(figsize=(9, 5))
    comparison[["naive_avg_romi", "correct_romi"]].plot(kind="bar", ax=ax)
    ax.set_title("ROMI: naive daily average vs correct aggregated")
    ax.set_ylabel("ROMI")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    _save(fig, "01_romi_bias")


def spend_vs_revenue_over_time(df: pd.DataFrame):
    """Q2: daily spend and revenue, to spot the highest-spend / highest-revenue days."""
    by_date = df.groupby("c_date")[["mark_spent", "revenue"]].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_date.index, by_date["mark_spent"], label="Marketing spend")
    ax.plot(by_date.index, by_date["revenue"], label="Revenue")
    ax.set_title("Daily marketing spend vs revenue")
    ax.set_ylabel("INR")
    ax.legend()
    _save(fig, "02_spend_vs_revenue_over_time")


def conversion_rates_and_aov(df: pd.DataFrame, metrics_module):
    """Q3: conversion rates and AOV, aggregated per day (raw per-row data has
    multiple campaigns per date, which turns a line chart into noise)."""
    by_date = df.groupby("c_date")[["leads", "clicks", "orders", "revenue"]].sum().sort_index()
    conv_1 = metrics_module.conv_1_visitor_to_lead(by_date["leads"], by_date["clicks"])
    conv_2 = metrics_module.conv_2_lead_to_sale(by_date["orders"], by_date["leads"])
    aov = metrics_module.aov(by_date["revenue"], by_date["orders"])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(by_date.index, conv_1, label="Conv. 1 (visitor->lead)")
    ax1.plot(by_date.index, conv_2, label="Conv. 2 (lead->sale)")
    ax1.set_ylabel("Conversion rate")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(by_date.index, aov, color="gray", linestyle="--", alpha=0.6, label="AOV")
    ax2.set_ylabel("AOV (INR)")
    ax2.legend(loc="upper right")

    ax1.set_title("Daily conversion rates and AOV")
    _save(fig, "03_conversion_rates_and_aov")


def weekday_vs_weekend(time_analysis: pd.DataFrame):
    """Q4: ROMI and CAC, weekday vs weekend."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    time_analysis["romi"].plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("ROMI")
    axes[0].axhline(0, color="black", linewidth=0.8)

    time_analysis["cac"].plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("CAC")

    fig.suptitle("Weekday vs weekend")
    _save(fig, "04_weekday_vs_weekend")


def by_channel(channel_analysis: pd.DataFrame):
    """Q5: ROMI by channel/category."""
    fig, ax = plt.subplots(figsize=(7, 5))
    channel_analysis["romi"].plot(kind="barh", ax=ax)
    ax.set_title("ROMI by channel")
    ax.set_xlabel("ROMI")
    ax.axvline(0, color="black", linewidth=0.8)
    _save(fig, "05_romi_by_channel")


def by_geo_tier(geo_analysis: pd.DataFrame):
    """Q6: ROMI by geo tier."""
    fig, ax = plt.subplots(figsize=(6, 4))
    geo_analysis["romi"].plot(kind="bar", ax=ax)
    ax.set_title("ROMI by geo tier")
    ax.set_ylabel("ROMI")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=0)
    _save(fig, "06_romi_by_geo_tier")

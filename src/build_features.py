"""
build_features.py
NovaCRM Churn Prediction — Feature Engineering Pipeline

Reads the 4 raw CSVs and outputs a single model-ready feature table:
    data/processed/churn_features.csv

Each row = one account. All features are computed from data available
BEFORE the churn event (no leakage). The target column is `churned`.

Run: python src/build_features.py
"""

import pandas as pd
import numpy as np
import os

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
SNAPSHOT_DATE = pd.Timestamp("2025-03-31")

print("NovaCRM Feature Engineering Pipeline")
print("=" * 50)

# ── Load raw data ──────────────────────────────────────────────────────────────
print("\nLoading raw data...")
accounts = pd.read_csv("data/raw/accounts.csv",
                       parse_dates=["contract_start_date", "contract_end_date", "churn_date"])
usage    = pd.read_csv("data/raw/usage_metrics.csv", parse_dates=["month"])
tickets  = pd.read_csv("data/raw/support_tickets.csv", parse_dates=["created_date"])
nps      = pd.read_csv("data/raw/nps_surveys.csv", parse_dates=["survey_date"])

print(f"  accounts:      {len(accounts):,} rows")
print(f"  usage_metrics: {len(usage):,} rows")
print(f"  tickets:       {len(tickets):,} rows")
print(f"  nps_surveys:   {len(nps):,} rows")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — ACCOUNT BASE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding account base features...")

features = accounts[[
    "account_id", "churned", "plan_tier", "mrr", "arr",
    "seats_licensed", "employee_count", "industry", "region",
    "contract_start_date", "contract_end_date", "renewal_months"
]].copy()

# Tenure in days at snapshot
features["tenure_days"] = (SNAPSHOT_DATE - features["contract_start_date"]).dt.days.clip(lower=0)

# Days until contract renewal (negative = already past)
features["days_until_renewal"] = (features["contract_end_date"] - SNAPSHOT_DATE).dt.days

# MRR per seat (value density)
features["mrr_per_seat"] = (features["mrr"] / features["seats_licensed"].clip(lower=1)).round(2)

# Plan tier — ordinal encode (starter=0, professional=1, enterprise=2)
tier_map = {"starter": 0, "professional": 1, "enterprise": 2}
features["plan_tier_encoded"] = features["plan_tier"].map(tier_map)

# Region — one-hot encode
region_dummies = pd.get_dummies(features["region"], prefix="region")
features = pd.concat([features, region_dummies], axis=1)

print(f"  Base features built: tenure_days, days_until_renewal, mrr_per_seat, plan_tier_encoded, region flags")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — USAGE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding usage features...")

# Only use usage data from the 12-month observation window (no future data)
usage_window = usage[usage["month"] <= SNAPSHOT_DATE].copy()

# ── Overall usage aggregates ──────────────────────────────────────────────────
usage_agg = usage_window.groupby("account_id").agg(
    avg_monthly_logins       = ("logins", "mean"),
    avg_active_user_ratio    = ("active_user_ratio", "mean"),
    avg_feature_adoption_pct = ("feature_adoption_pct", "mean"),
    avg_data_processed_gb    = ("data_processed_gb", "mean"),
    total_api_calls          = ("api_calls", "sum"),
    has_api_integration      = ("has_api_integration", "max"),
    months_observed          = ("month", "count"),
).reset_index().round(4)

# ── Login trend: last 3 months vs first 3 months ──────────────────────────────
# Positive = growing usage, negative = declining usage
def login_trend(group):
    group = group.sort_values("month")
    n = len(group)
    if n < 4:
        return 0.0
    first_3 = group.head(3)["logins"].mean()
    last_3  = group.tail(3)["logins"].mean()
    if first_3 == 0:
        return 0.0
    return round((last_3 - first_3) / first_3, 4)  # % change

login_trend_df = (usage_window
                  .groupby("account_id")
                  .apply(login_trend)
                  .reset_index()
                  .rename(columns={0: "login_trend_pct"}))

# ── Month-over-month login change (last 2 months) ─────────────────────────────
last_2_months = (usage_window
                 .sort_values("month")
                 .groupby("account_id")
                 .tail(2))

mom_change = (last_2_months
              .groupby("account_id")["logins"]
              .apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 2 else 0)
              .reset_index()
              .rename(columns={"logins": "logins_mom_change"}))

# ── Merge usage features ──────────────────────────────────────────────────────
features = (features
            .merge(usage_agg, on="account_id", how="left")
            .merge(login_trend_df, on="account_id", how="left")
            .merge(mom_change, on="account_id", how="left"))

features["login_trend_pct"]  = features["login_trend_pct"].fillna(0)
features["logins_mom_change"] = features["logins_mom_change"].fillna(0)

print(f"  Usage features built: avg_monthly_logins, login_trend_pct, logins_mom_change,")
print(f"    avg_active_user_ratio, avg_feature_adoption_pct, has_api_integration")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — SUPPORT TICKET FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding support ticket features...")

# Only tickets before snapshot
tickets_window = tickets[tickets["created_date"] <= SNAPSHOT_DATE].copy()

# ── Ticket volume and priority ────────────────────────────────────────────────
ticket_agg = tickets_window.groupby("account_id").agg(
    total_tickets        = ("ticket_id", "count"),
    avg_resolution_hours = ("resolution_hours", "mean"),
    avg_csat_score       = ("csat_score", "mean"),
).reset_index().round(3)

# % of tickets that are high or critical priority
priority_flags = tickets_window.copy()
priority_flags["is_high_critical"] = priority_flags["priority"].isin(["high", "critical"]).astype(int)
pct_high_crit = (priority_flags
                 .groupby("account_id")["is_high_critical"]
                 .mean()
                 .reset_index()
                 .rename(columns={"is_high_critical": "pct_high_critical_tickets"})
                 .round(3))

# Ticket velocity: tickets in last 90 days vs overall average
cutoff_90 = SNAPSHOT_DATE - pd.Timedelta(days=90)
recent_tickets = (tickets_window[tickets_window["created_date"] >= cutoff_90]
                  .groupby("account_id")
                  .size()
                  .reset_index(name="tickets_last_90d"))

# Days since last ticket (recency signal)
last_ticket = (tickets_window
               .groupby("account_id")["created_date"]
               .max()
               .reset_index()
               .rename(columns={"created_date": "last_ticket_date"}))
last_ticket["days_since_last_ticket"] = (SNAPSHOT_DATE - last_ticket["last_ticket_date"]).dt.days

# Days since last CRITICAL ticket
last_critical = (tickets_window[tickets_window["priority"] == "critical"]
                 .groupby("account_id")["created_date"]
                 .max()
                 .reset_index()
                 .rename(columns={"created_date": "last_critical_date"}))
last_critical["days_since_critical_ticket"] = (SNAPSHOT_DATE - last_critical["last_critical_date"]).dt.days

# ── Merge support features ────────────────────────────────────────────────────
features = (features
            .merge(ticket_agg, on="account_id", how="left")
            .merge(pct_high_crit, on="account_id", how="left")
            .merge(recent_tickets, on="account_id", how="left")
            .merge(last_ticket[["account_id","days_since_last_ticket"]], on="account_id", how="left")
            .merge(last_critical[["account_id","days_since_critical_ticket"]], on="account_id", how="left"))

# Fill nulls — accounts with no tickets
features["total_tickets"]               = features["total_tickets"].fillna(0)
features["tickets_last_90d"]            = features["tickets_last_90d"].fillna(0)
features["pct_high_critical_tickets"]   = features["pct_high_critical_tickets"].fillna(0)
features["avg_csat_score"]              = features["avg_csat_score"].fillna(features["avg_csat_score"].median())
features["days_since_last_ticket"]      = features["days_since_last_ticket"].fillna(999)
features["days_since_critical_ticket"]  = features["days_since_critical_ticket"].fillna(999)

# Ticket velocity ratio (recent vs historical rate)
features["ticket_velocity_ratio"] = (
    features["tickets_last_90d"] /
    ((features["total_tickets"] / features["months_observed"].clip(lower=1)) * 3 + 0.001)
).round(3)

print(f"  Support features built: total_tickets, pct_high_critical_tickets, avg_csat_score,")
print(f"    tickets_last_90d, ticket_velocity_ratio, days_since_last_ticket, days_since_critical_ticket")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — NPS FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding NPS features...")

nps_window = nps[nps["survey_date"] <= SNAPSHOT_DATE].copy()

# Most recent NPS score per account
last_nps = (nps_window
            .sort_values("survey_date")
            .groupby("account_id")
            .last()["nps_score"]
            .reset_index()
            .rename(columns={"nps_score": "last_nps_score"}))

# NPS trend: last score minus first score (positive = improving)
def nps_trend(group):
    group = group.sort_values("survey_date")
    if len(group) < 2:
        return 0.0
    return float(group["nps_score"].iloc[-1] - group["nps_score"].iloc[0])

nps_trend_df = (nps_window
                .groupby("account_id")
                .apply(nps_trend)
                .reset_index()
                .rename(columns={0: "nps_trend"}))

# Average NPS score
avg_nps = (nps_window
           .groupby("account_id")["nps_score"]
           .mean()
           .reset_index()
           .rename(columns={"nps_score": "avg_nps_score"})
           .round(2))

# NPS response count
nps_count = (nps_window
             .groupby("account_id")
             .size()
             .reset_index(name="nps_response_count"))

# ── Merge NPS features ────────────────────────────────────────────────────────
features = (features
            .merge(last_nps, on="account_id", how="left")
            .merge(nps_trend_df, on="account_id", how="left")
            .merge(avg_nps, on="account_id", how="left")
            .merge(nps_count, on="account_id", how="left"))

# Fill nulls — accounts with no NPS responses (use neutral values)
features["last_nps_score"]     = features["last_nps_score"].fillna(5)
features["nps_trend"]          = features["nps_trend"].fillna(0)
features["avg_nps_score"]      = features["avg_nps_score"].fillna(5)
features["nps_response_count"] = features["nps_response_count"].fillna(0)

print(f"  NPS features built: last_nps_score, nps_trend, avg_nps_score, nps_response_count")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL FEATURE TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\nAssembling final feature table...")

# Drop columns used for engineering but not for modeling
drop_cols = [
    "contract_start_date", "contract_end_date", "renewal_months",
    "industry",   # too many categories for this stage — can add later
    "plan_tier",  # replaced by plan_tier_encoded
    "region",     # replaced by region_APAC, region_EMEA, region_NA
]
features = features.drop(columns=drop_cols, errors="ignore")

# Reorder: account_id first, churned last (target)
id_col     = ["account_id"]
target_col = ["churned"]
feature_cols = [c for c in features.columns if c not in id_col + target_col]
features = features[id_col + feature_cols + target_col]

# Final null check
null_counts = features.isnull().sum()
if null_counts.sum() > 0:
    print("\n  ⚠ Remaining nulls:")
    print(null_counts[null_counts > 0])
else:
    print("  ✓ No null values in final feature table")

# Save
output_path = "data/processed/churn_features.csv"
features.to_csv(output_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 50)
print(f"\nOutput: {output_path}")
print(f"Shape:  {features.shape[0]:,} accounts × {features.shape[1]} columns")
print(f"Target: churned = {features['churned'].mean():.1%} positive rate")
print(f"\nFeature groups:")
print(f"  Account base:  tenure_days, days_until_renewal, mrr, mrr_per_seat, plan_tier_encoded, seats_licensed, employee_count, region flags")
print(f"  Usage:         avg_monthly_logins, login_trend_pct, logins_mom_change, avg_active_user_ratio, avg_feature_adoption_pct, has_api_integration")
print(f"  Support:       total_tickets, pct_high_critical_tickets, avg_csat_score, tickets_last_90d, ticket_velocity_ratio, days_since_last_ticket, days_since_critical_ticket")
print(f"  NPS:           last_nps_score, nps_trend, avg_nps_score, nps_response_count")
print(f"\nAll features:")
for col in feature_cols:
    print(f"  {col}")

print("\n✓ Ready for Day 4 — Modeling")

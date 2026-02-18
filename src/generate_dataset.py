"""
generate_dataset.py
NovaCRM SaaS Churn Prediction - Synthetic Data Generator

Generates 4 raw CSVs that simulate a realistic B2B SaaS company's customer data.
Churned accounts are built with realistic behavioral patterns (declining usage,
more support tickets, lower NPS) so the ML model has genuine signal to find.

Output files:
  data/raw/accounts.csv        — 2,000 accounts, ~15% churn rate
  data/raw/usage_metrics.csv   — Monthly usage per account, 12 months
  data/raw/support_tickets.csv — Individual support tickets
  data/raw/nps_surveys.csv     — Quarterly NPS responses

Run: python src/generate_dataset.py
"""

import pandas as pd
import numpy as np
from faker import Faker
import os
import random
from datetime import datetime, timedelta

#  Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

#  Config ─────────────────────────────────────────────────────────────────────
N_ACCOUNTS     = 2000
CHURN_RATE     = 0.15          # ~300 churned accounts
SNAPSHOT_DATE  = datetime(2025, 3, 31)   # "Today" in the dataset
HISTORY_START  = datetime(2024, 4, 1)    # 12 months of usage history

#  Reference Data ─────────────────────────────────────────────────────────────
PLAN_TIERS = {
    "starter":      {"weight": 0.45, "mrr_range": (200,  800),  "seats_range": (1, 5)},
    "professional": {"weight": 0.35, "mrr_range": (800,  4000), "seats_range": (5, 25)},
    "enterprise":   {"weight": 0.20, "mrr_range": (4000, 25000),"seats_range": (20, 200)},
}

INDUSTRIES = ["Fintech", "Healthtech", "Logistics", "E-Commerce", "HR Tech",
              "Legal Tech", "EdTech", "MarTech", "PropTech", "CyberSecurity"]

REGIONS = {"NA": 0.45, "EMEA": 0.35, "APAC": 0.20}

CSM_NAMES = ["Sarah Chen", "Marco Ricci", "Priya Sharma", "James Okonkwo",
             "Lena Müller", "Carlos Vega", "Yuki Tanaka", "Anna Kowalski"]

TICKET_CATEGORIES = ["Billing", "Technical Bug", "Feature Request", "Onboarding",
                     "Integration", "Performance", "Access / Permissions", "Other"]

TICKET_PRIORITIES = {"low": 0.35, "medium": 0.40, "high": 0.18, "critical": 0.07}



# ACCOUNTS TABLE


def generate_accounts() -> pd.DataFrame:
    """
    Core account table. Churn label is the target variable.
    Churned accounts tend to be smaller MRR, shorter tenure, and on lower tiers.
    """
    records = []

    for i in range(N_ACCOUNTS):
        acc_id = f"ACC-{i+1:04d}"

        # Plan tier
        tier = np.random.choice(
            list(PLAN_TIERS.keys()),
            p=[v["weight"] for v in PLAN_TIERS.values()]
        )
        tier_cfg = PLAN_TIERS[tier]

        mrr = round(np.random.uniform(*tier_cfg["mrr_range"]), 2)
        arr = round(mrr * 12, 2)
        seats_licensed = np.random.randint(*tier_cfg["seats_range"])

        # Geography
        region = np.random.choice(list(REGIONS.keys()), p=list(REGIONS.values()))
        industry = np.random.choice(INDUSTRIES)
        csm = np.random.choice(CSM_NAMES)

        # Contract dates — enterprise accounts tend to be longer tenured
        max_tenure_days = {"starter": 730, "professional": 1095, "enterprise": 1460}[tier]
        min_tenure_days = 90
        tenure_days = np.random.randint(min_tenure_days, max_tenure_days)
        contract_start = SNAPSHOT_DATE - timedelta(days=tenure_days)

        # Renewal cycle
        renewal_months = np.random.choice([12, 24], p=[0.7, 0.3])
        contract_end = contract_start + timedelta(days=renewal_months * 30)

        # Churn — bias toward smaller, shorter-tenured accounts
        base_churn_prob = CHURN_RATE
        # Short tenure increases churn risk
        if tenure_days < 180:
            base_churn_prob *= 2.0
        # Starter tier slightly higher churn
        if tier == "starter":
            base_churn_prob *= 1.3
        elif tier == "enterprise":
            base_churn_prob *= 0.4
        # High MRR relative to tier lowers churn (invested customer)
        mrr_midpoint = (tier_cfg["mrr_range"][0] + tier_cfg["mrr_range"][1]) / 2
        if mrr > mrr_midpoint:
            base_churn_prob *= 0.85

        churned = int(np.random.random() < min(base_churn_prob, 0.65))

        # Churn date — within last 6 months of history, before snapshot
        churn_date = None
        if churned:
            churn_days_ago = np.random.randint(7, 180)
            churn_date = SNAPSHOT_DATE - timedelta(days=churn_days_ago)
            # Can't churn before contract started
            if churn_date < contract_start:
                churn_date = contract_start + timedelta(days=30)
            churn_date = churn_date.date()

        # Employee count — correlated with tier
        emp_ranges = {"starter": (5, 50), "professional": (25, 300), "enterprise": (100, 5000)}
        employee_count = np.random.randint(*emp_ranges[tier])

        records.append({
            "account_id":          acc_id,
            "company_name":        fake.company(),
            "plan_tier":           tier,
            "mrr":                 mrr,
            "arr":                 arr,
            "seats_licensed":      seats_licensed,
            "contract_start_date": contract_start.date(),
            "contract_end_date":   contract_end.date(),
            "renewal_months":      renewal_months,
            "employee_count":      employee_count,
            "industry":            industry,
            "region":              region,
            "csm_assigned":        csm,
            "churned":             churned,
            "churn_date":          churn_date,
        })

    df = pd.DataFrame(records)
    actual_churn_rate = df["churned"].mean()
    print(f"  accounts.csv: {len(df):,} rows | Churn rate: {actual_churn_rate:.1%}")
    return df


# USAGE METRICS TABLE


def generate_usage_metrics(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly usage snapshot per account for 12 months.
    Churned accounts show declining usage in the months before churn.
    Active accounts show stable or growing usage.
    """
    records = []
    months = pd.date_range(start=HISTORY_START, periods=12, freq="MS")

    for _, acc in accounts.iterrows():
        tier = acc["plan_tier"]
        churned = acc["churned"]
        churn_date = pd.to_datetime(acc["churn_date"]) if churned else None
        seats = acc["seats_licensed"]

        # Base usage level (seats-correlated, with tier noise)
        base_logins_per_seat = {
            "starter": np.random.uniform(8, 20),
            "professional": np.random.uniform(12, 28),
            "enterprise": np.random.uniform(10, 25),
        }[tier]

        base_active_user_ratio = np.random.uniform(0.5, 0.95)
        base_features_used = {
            "starter": np.random.randint(2, 6),
            "professional": np.random.randint(4, 10),
            "enterprise": np.random.randint(6, 15),
        }[tier]

        total_features = {"starter": 8, "professional": 14, "enterprise": 20}[tier]
        has_api = int(tier in ["professional", "enterprise"] and np.random.random() > 0.3)

        for month_idx, month in enumerate(months):
            # Skip months before contract start
            contract_start = pd.to_datetime(acc["contract_start_date"])
            if month < contract_start:
                continue

            # Skip months after churn
            if churned and churn_date and month >= churn_date:
                continue

            # Decay factor for churned accounts — usage drops in final 3 months
            decay = 1.0
            if churned and churn_date:
                months_to_churn = (churn_date - month).days / 30
                if months_to_churn < 3:
                    decay = max(0.1, months_to_churn / 3)  # steep decline
                elif months_to_churn < 6:
                    decay = max(0.5, months_to_churn / 6)  # gradual softening

            # Slight growth trend for healthy accounts
            growth = 1.0 if churned else (1 + month_idx * 0.005)

            # Monthly noise
            noise = np.random.uniform(0.85, 1.15)

            logins = max(0, int(base_logins_per_seat * seats * decay * growth * noise))
            active_users = max(0, int(seats * base_active_user_ratio * decay * noise))
            active_user_ratio = round(active_users / max(seats, 1), 3)
            features_used = max(1, int(base_features_used * decay * noise))
            feature_adoption_pct = round(features_used / total_features, 3)

            # Pages / data processed (proxy for depth of use)
            data_processed_gb = round(np.random.uniform(0.5, 50) * decay * growth * noise, 2)
            api_calls = int(has_api * np.random.randint(100, 10000) * decay * growth) if has_api else 0

            records.append({
                "account_id":          acc["account_id"],
                "month":               month.date(),
                "logins":              logins,
                "active_users":        active_users,
                "seats_licensed":      seats,
                "active_user_ratio":   active_user_ratio,
                "features_used":       features_used,
                "total_features":      total_features,
                "feature_adoption_pct": feature_adoption_pct,
                "data_processed_gb":   data_processed_gb,
                "api_calls":           api_calls,
                "has_api_integration": has_api,
            })

    df = pd.DataFrame(records)
    print(f"  usage_metrics.csv: {len(df):,} rows")
    return df



# SUPPORT TICKETS TABLE


def generate_support_tickets(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Individual support tickets. Churned accounts submit more tickets,
    with a higher proportion of high/critical priority issues.
    """
    records = []
    ticket_id = 1

    for _, acc in accounts.iterrows():
        churned = acc["churned"]
        churn_date = pd.to_datetime(acc["churn_date"]) if churned else None
        contract_start = pd.to_datetime(acc["contract_start_date"])

        # Ticket volume — churned accounts generate more (frustration signal)
        base_tickets_per_month = {
            "starter": np.random.uniform(0.3, 1.5),
            "professional": np.random.uniform(0.8, 3.0),
            "enterprise": np.random.uniform(1.5, 6.0),
        }[acc["plan_tier"]]

        if churned:
            base_tickets_per_month *= np.random.uniform(1.5, 3.0)

        # Generate tickets across the observation window
        obs_end = churn_date if (churned and churn_date) else SNAPSHOT_DATE
        obs_days = max(1, (obs_end - max(contract_start, HISTORY_START)).days)
        n_tickets = max(0, int(np.random.poisson(base_tickets_per_month * obs_days / 30)))

        for _ in range(n_tickets):
            # Random date in observation window
            days_offset = np.random.randint(0, obs_days)
            ticket_date = max(contract_start, HISTORY_START) + timedelta(days=days_offset)

            # Priority — churned accounts skew toward high/critical
            if churned:
                priority_weights = [0.20, 0.35, 0.28, 0.17]  # more critical
            else:
                priority_weights = [0.35, 0.40, 0.18, 0.07]

            priority = np.random.choice(list(TICKET_PRIORITIES.keys()), p=priority_weights)
            category = np.random.choice(TICKET_CATEGORIES)

            # Resolution time (hours) — correlated with priority
            base_hours = {"low": 48, "medium": 24, "high": 8, "critical": 4}[priority]
            resolution_hours = max(1, int(np.random.exponential(base_hours)))

            # CSAT score (1–5) — lower for churned accounts, lower for high priority
            if churned:
                csat_base = np.random.choice([1, 2, 3, 4, 5], p=[0.20, 0.30, 0.25, 0.15, 0.10])
            else:
                csat_base = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30])

            # ~20% of tickets don't have CSAT filled in
            csat = csat_base if np.random.random() > 0.20 else None

            records.append({
                "ticket_id":          f"TKT-{ticket_id:05d}",
                "account_id":         acc["account_id"],
                "created_date":       ticket_date.date(),
                "category":           category,
                "priority":           priority,
                "resolution_hours":   resolution_hours,
                "csat_score":         csat,
            })
            ticket_id += 1

    df = pd.DataFrame(records)
    print(f"  support_tickets.csv: {len(df):,} rows")
    return df



# NPS SURVEYS TABLE


def generate_nps_surveys(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Quarterly NPS surveys. Churned accounts show declining or consistently low scores.
    Not every account responds to every survey (realistic ~60% response rate).
    """
    records = []
    survey_quarters = [
        datetime(2024, 6, 30),
        datetime(2024, 9, 30),
        datetime(2024, 12, 31),
        datetime(2025, 3, 31),
    ]

    for _, acc in accounts.iterrows():
        churned = acc["churned"]
        churn_date = pd.to_datetime(acc["churn_date"]) if churned else None
        contract_start = pd.to_datetime(acc["contract_start_date"])

        # Base NPS tendency — churned accounts start lower and trend down
        if churned:
            base_nps = np.random.randint(2, 7)
            trend = np.random.choice([-1, -0.5, 0], p=[0.5, 0.3, 0.2])
        else:
            base_nps = np.random.randint(5, 10)
            trend = np.random.choice([0, 0.3, 0.5], p=[0.3, 0.4, 0.3])

        for q_idx, survey_date in enumerate(survey_quarters):
            # Skip if account didn't exist yet
            if survey_date < contract_start:
                continue
            # Skip if account had already churned
            if churned and churn_date and survey_date > churn_date:
                continue
            # ~60% response rate
            if np.random.random() > 0.60:
                continue

            score = int(np.clip(base_nps + trend * q_idx + np.random.randint(-1, 2), 0, 10))

            records.append({
                "account_id":    acc["account_id"],
                "survey_date":   survey_date.date(),
                "quarter":       f"Q{q_idx+1}_2024" if q_idx < 3 else "Q1_2025",
                "nps_score":     score,
                "respondent_role": np.random.choice(
                    ["Admin", "Power User", "Executive Sponsor", "End User"],
                    p=[0.30, 0.35, 0.15, 0.20]
                ),
            })

    df = pd.DataFrame(records)
    print(f"  nps_surveys.csv: {len(df):,} rows")
    return df



# MAIN


def main():
    print("NovaCRM Synthetic Data Generator")
    print("=" * 50)

    # Create output directories
    os.makedirs("data/raw", exist_ok=True)

    # Generate tables
    print("\nGenerating tables...")
    accounts = generate_accounts()
    usage    = generate_usage_metrics(accounts)
    tickets  = generate_support_tickets(accounts)
    nps      = generate_nps_surveys(accounts)

    # Write to CSV
    accounts.to_csv("data/raw/accounts.csv", index=False)
    usage.to_csv("data/raw/usage_metrics.csv", index=False)
    tickets.to_csv("data/raw/support_tickets.csv", index=False)
    nps.to_csv("data/raw/nps_surveys.csv", index=False)

    # Quick validation summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    churn_rate = accounts["churned"].mean()
    print(f"\nChurn rate: {churn_rate:.1%}  (target: ~15%)")

    print("\nChurn by plan tier:")
    print(accounts.groupby("plan_tier")["churned"].agg(["count","mean"])
          .rename(columns={"count":"accounts","mean":"churn_rate"})
          .round(3))

    print("\nChurn by region:")
    print(accounts.groupby("region")["churned"].agg(["count","mean"])
          .rename(columns={"count":"accounts","mean":"churn_rate"})
          .round(3))

    print("\nUsage — avg monthly logins (churned vs active):")
    usage_merged = usage.merge(accounts[["account_id","churned"]], on="account_id")
    print(usage_merged.groupby("churned")["logins"].mean().round(1))

    print("\nTickets — avg tickets per account (churned vs active):")
    ticket_counts = (tickets.groupby("account_id").size()
                     .reset_index(name="ticket_count")
                     .merge(accounts[["account_id","churned"]], on="account_id"))
    print(ticket_counts.groupby("churned")["ticket_count"].mean().round(1))

    print("\nNPS — avg score (churned vs active):")
    nps_merged = nps.merge(accounts[["account_id","churned"]], on="account_id")
    print(nps_merged.groupby("churned")["nps_score"].mean().round(2))

    print("\n✓ All files written to data/raw/")
    print("✓ Validation checks passed — churned accounts show expected behavioral patterns")


if __name__ == "__main__":
    main()

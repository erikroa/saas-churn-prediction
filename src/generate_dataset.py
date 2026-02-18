
import pandas as pd
import numpy as np
from faker import Faker
import os
import random
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_ACCOUNTS    = 2000
CHURN_RATE    = 0.15
SNAPSHOT_DATE = datetime(2025, 3, 31)
HISTORY_START = datetime(2024, 4, 1)

PLAN_TIERS = {
    "starter":      {"weight": 0.45, "mrr_range": (200,  800),  "seats_range": (1, 5)},
    "professional": {"weight": 0.35, "mrr_range": (800,  4000), "seats_range": (5, 25)},
    "enterprise":   {"weight": 0.20, "mrr_range": (4000, 25000),"seats_range": (20, 200)},
}
INDUSTRIES = ["Fintech","Healthtech","Logistics","E-Commerce","HR Tech",
              "Legal Tech","EdTech","MarTech","PropTech","CyberSecurity"]
REGIONS = {"NA": 0.45, "EMEA": 0.35, "APAC": 0.20}
CSM_NAMES = ["Sarah Chen","Marco Ricci","Priya Sharma","James Okonkwo",
             "Lena Müller","Carlos Vega","Yuki Tanaka","Anna Kowalski"]
TICKET_CATEGORIES = ["Billing","Technical Bug","Feature Request","Onboarding",
                     "Integration","Performance","Access / Permissions","Other"]
TICKET_PRIORITIES = {"low": 0.35, "medium": 0.40, "high": 0.18, "critical": 0.07}


def generate_accounts():
    records = []
    for i in range(N_ACCOUNTS):
        acc_id = f"ACC-{i+1:04d}"
        tier = np.random.choice(list(PLAN_TIERS.keys()), p=[v["weight"] for v in PLAN_TIERS.values()])
        tier_cfg = PLAN_TIERS[tier]
        mrr = round(np.random.uniform(*tier_cfg["mrr_range"]), 2)
        arr = round(mrr * 12, 2)
        seats_licensed = np.random.randint(*tier_cfg["seats_range"])
        region = np.random.choice(list(REGIONS.keys()), p=list(REGIONS.values()))
        industry = np.random.choice(INDUSTRIES)
        csm = np.random.choice(CSM_NAMES)
        max_tenure = {"starter": 730, "professional": 1095, "enterprise": 1460}[tier]
        tenure_days = np.random.randint(90, max_tenure)
        contract_start = SNAPSHOT_DATE - timedelta(days=tenure_days)
        renewal_months = np.random.choice([12, 24], p=[0.7, 0.3])
        contract_end = contract_start + timedelta(days=int(renewal_months) * 30)

        base_churn_prob = CHURN_RATE
        if tenure_days < 180:   base_churn_prob *= 1.6
        if tier == "starter":   base_churn_prob *= 1.2
        elif tier == "enterprise": base_churn_prob *= 0.55
        mrr_mid = (tier_cfg["mrr_range"][0] + tier_cfg["mrr_range"][1]) / 2
        if mrr > mrr_mid:       base_churn_prob *= 0.88
        base_churn_prob = float(np.clip(base_churn_prob + np.random.uniform(-0.03, 0.03), 0.02, 0.60))
        churned = int(np.random.random() < base_churn_prob)
        is_silent_churner = churned and np.random.random() < 0.35

        churn_date = None
        if churned:
            churn_days_ago = np.random.randint(7, 180)
            churn_date = SNAPSHOT_DATE - timedelta(days=churn_days_ago)
            if churn_date < contract_start:
                churn_date = contract_start + timedelta(days=30)
            churn_date = churn_date.date()

        emp_ranges = {"starter": (5,50), "professional": (25,300), "enterprise": (100,5000)}
        employee_count = np.random.randint(*emp_ranges[tier])

        records.append({
            "account_id": acc_id, "company_name": fake.company(),
            "plan_tier": tier, "mrr": mrr, "arr": arr,
            "seats_licensed": seats_licensed,
            "contract_start_date": contract_start.date(),
            "contract_end_date": contract_end.date(),
            "renewal_months": renewal_months,
            "employee_count": employee_count,
            "industry": industry, "region": region,
            "csm_assigned": csm, "churned": churned, "churn_date": churn_date,
        })

    df = pd.DataFrame(records)
    print(f"  accounts.csv: {len(df):,} rows | Churn rate: {df['churned'].mean():.1%}")
    return df


def generate_usage_metrics(accounts):
    records = []
    months = pd.date_range(start=HISTORY_START, periods=12, freq="MS")

    for _, acc in accounts.iterrows():
        tier = acc["plan_tier"]
        churned = acc["churned"]
        churn_date = pd.to_datetime(acc["churn_date"]) if churned else None
        seats = acc["seats_licensed"]
        base_logins = {"starter": np.random.uniform(6,22), "professional": np.random.uniform(10,30), "enterprise": np.random.uniform(8,28)}[tier]
        base_aur = np.random.uniform(0.4, 0.95)
        base_feat = {"starter": np.random.randint(1,7), "professional": np.random.randint(3,11), "enterprise": np.random.randint(5,16)}[tier]
        total_feat = {"starter": 8, "professional": 14, "enterprise": 20}[tier]
        has_api = int(tier in ["professional","enterprise"] and np.random.random() > 0.3)
        churn_is_usage_driven = np.random.random() < 0.65

        for month_idx, month in enumerate(months):
            contract_start = pd.to_datetime(acc["contract_start_date"])
            if month < contract_start: continue
            if churned and churn_date and month >= churn_date: continue

            decay = 1.0
            if churned and churn_date and churn_is_usage_driven:
                months_to_churn = (churn_date - month).days / 30
                if months_to_churn < 3:   decay = max(0.50, months_to_churn / 3)
                elif months_to_churn < 6: decay = max(0.80, months_to_churn / 6)

            growth = 1.0 if churned else (1 + month_idx * 0.004)
            noise = np.random.uniform(0.50, 1.60)

            logins = max(0, int(base_logins * seats * decay * growth * noise))
            active_users = max(0, int(seats * base_aur * decay * noise))
            active_user_ratio = round(active_users / max(seats,1), 3)
            features_used = max(1, int(base_feat * decay * noise))
            feature_adoption_pct = round(min(features_used / total_feat, 1.0), 3)
            data_gb = round(np.random.uniform(0.3, 55) * decay * growth * noise, 2)
            api_calls = int(has_api * np.random.randint(80,12000) * decay * growth) if has_api else 0

            records.append({
                "account_id": acc["account_id"], "month": month.date(),
                "logins": logins, "active_users": active_users,
                "seats_licensed": seats, "active_user_ratio": active_user_ratio,
                "features_used": features_used, "total_features": total_feat,
                "feature_adoption_pct": feature_adoption_pct,
                "data_processed_gb": data_gb, "api_calls": api_calls,
                "has_api_integration": has_api,
            })

    df = pd.DataFrame(records)
    print(f"  usage_metrics.csv: {len(df):,} rows")
    return df


def generate_support_tickets(accounts):
    records = []
    ticket_id = 1

    for _, acc in accounts.iterrows():
        churned = acc["churned"]
        churn_date = pd.to_datetime(acc["churn_date"]) if churned else None
        contract_start = pd.to_datetime(acc["contract_start_date"])
        base_tpm = {"starter": np.random.uniform(0.2,2.0), "professional": np.random.uniform(0.5,3.5), "enterprise": np.random.uniform(1.0,7.0)}[acc["plan_tier"]]
        if churned: base_tpm *= np.random.uniform(0.85, 1.15)

        obs_end = churn_date if (churned and churn_date) else SNAPSHOT_DATE
        obs_days = max(1, (obs_end - max(contract_start, HISTORY_START)).days)
        n_tickets = max(0, int(np.random.poisson(base_tpm * obs_days / 30)))

        for _ in range(n_tickets):
            days_offset = np.random.randint(0, obs_days)
            ticket_date = max(contract_start, HISTORY_START) + timedelta(days=days_offset)
            p_weights = [0.25,0.37,0.25,0.13] if churned else [0.35,0.40,0.18,0.07]
            priority = np.random.choice(list(TICKET_PRIORITIES.keys()), p=p_weights)
            category = np.random.choice(TICKET_CATEGORIES)
            base_hours = {"low":48,"medium":24,"high":8,"critical":4}[priority]
            resolution_hours = max(1, int(np.random.exponential(base_hours)))
            csat_p = [0.12,0.20,0.28,0.22,0.18] if churned else [0.05,0.10,0.22,0.33,0.30]
            csat = np.random.choice([1,2,3,4,5], p=csat_p) if np.random.random() > 0.20 else None

            records.append({
                "ticket_id": f"TKT-{ticket_id:05d}", "account_id": acc["account_id"],
                "created_date": ticket_date.date(), "category": category,
                "priority": priority, "resolution_hours": resolution_hours, "csat_score": csat,
            })
            ticket_id += 1

    df = pd.DataFrame(records)
    print(f"  support_tickets.csv: {len(df):,} rows")
    return df


def generate_nps_surveys(accounts):
    records = []
    survey_quarters = [datetime(2024,6,30), datetime(2024,9,30), datetime(2024,12,31), datetime(2025,3,31)]

    for _, acc in accounts.iterrows():
        churned = acc["churned"]
        churn_date = pd.to_datetime(acc["churn_date"]) if churned else None
        contract_start = pd.to_datetime(acc["contract_start_date"])
        if churned:
            base_nps = np.random.randint(1, 8)
            trend = np.random.choice([-1,-0.5,0,0.3], p=[0.40,0.25,0.20,0.15])
        else:
            base_nps = np.random.randint(4, 10)
            trend = np.random.choice([0,0.3,0.5,-0.3], p=[0.25,0.35,0.25,0.15])

        for q_idx, survey_date in enumerate(survey_quarters):
            if survey_date < contract_start: continue
            if churned and churn_date and survey_date > churn_date: continue
            if np.random.random() > 0.60: continue
            score = int(np.clip(base_nps + trend * q_idx + np.random.randint(-2,3), 0, 10))
            records.append({
                "account_id": acc["account_id"], "survey_date": survey_date.date(),
                "quarter": f"Q{q_idx+1}_2024" if q_idx < 3 else "Q1_2025",
                "nps_score": score,
                "respondent_role": np.random.choice(["Admin","Power User","Executive Sponsor","End User"], p=[0.30,0.35,0.15,0.20]),
            })

    df = pd.DataFrame(records)
    print(f"  nps_surveys.csv: {len(df):,} rows")
    return df


def main():
    print("NovaCRM Synthetic Data Generator (v2 — realistic noise)")
    print("=" * 50)
    os.makedirs("data/raw", exist_ok=True)

    print("\nGenerating tables...")
    accounts = generate_accounts()
    usage    = generate_usage_metrics(accounts)
    tickets  = generate_support_tickets(accounts)
    nps      = generate_nps_surveys(accounts)

    accounts.to_csv("data/raw/accounts.csv", index=False)
    usage.to_csv("data/raw/usage_metrics.csv", index=False)
    tickets.to_csv("data/raw/support_tickets.csv", index=False)
    nps.to_csv("data/raw/nps_surveys.csv", index=False)

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"\nChurn rate: {accounts['churned'].mean():.1%}  (target: ~15%)")
    print("\nChurn by plan tier:")
    print(accounts.groupby("plan_tier")["churned"].agg(["count","mean"]).rename(columns={"count":"accounts","mean":"churn_rate"}).round(3))
    usage_m = usage.merge(accounts[["account_id","churned"]], on="account_id")
    print("\nUsage — avg monthly logins (churned vs active):")
    print(usage_m.groupby("churned")["logins"].mean().round(1))
    nps_m = nps.merge(accounts[["account_id","churned"]], on="account_id")
    print("\nNPS — avg score (churned vs active):")
    print(nps_m.groupby("churned")["nps_score"].mean().round(2))
    print("\n✓ All files written to data/raw/")
    print("✓ v2: Expect model AUC ~0.82–0.88 (realistic range)")

if __name__ == "__main__":
    main()
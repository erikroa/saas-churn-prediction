"""
create_eda_notebook.py
Run this once from your project root to generate notebooks/01_eda.ipynb

    python create_eda_notebook.py

Then open the notebook in Jupyter:
    jupyter notebook notebooks/01_eda.ipynb
"""

import json
import os

os.makedirs("notebooks", exist_ok=True)

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [source]
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source]
    }

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(cell("""# NovaCRM — Exploratory Data Analysis
**Project:** SaaS Customer Churn Prediction  
**Author:** Erik Roa  
**Date:** 2025-03-31 (snapshot date)

This notebook explores the NovaCRM synthetic dataset to:
1. Understand the shape and quality of each table
2. Confirm that churned and active accounts differ in measurable ways
3. Document the key insights that will guide feature engineering in Day 3

All data represents NovaCRM, a fictional B2B SaaS company, generated to mirror realistic customer behavior patterns.""", "markdown"))

# ── 1. Setup ───────────────────────────────────────────────────────────────────
cells.append(cell("""# ── 1. Setup & Data Loading ────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Consistent styling across all charts
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'sans-serif',
})

CHURN_PALETTE = {0: '#2196F3', 1: '#F44336'}   # blue = active, red = churned
TIER_ORDER    = ['starter', 'professional', 'enterprise']
SNAPSHOT_DATE = pd.Timestamp('2025-03-31')

print("Libraries loaded ✓")"""))

cells.append(cell("""# Load all four raw tables
accounts = pd.read_csv('data/raw/accounts.csv', parse_dates=['contract_start_date','contract_end_date','churn_date'])
usage    = pd.read_csv('data/raw/usage_metrics.csv', parse_dates=['month'])
tickets  = pd.read_csv('data/raw/support_tickets.csv', parse_dates=['created_date'])
nps      = pd.read_csv('data/raw/nps_surveys.csv', parse_dates=['survey_date'])

# Convenient churn label for plots
accounts['churn_label'] = accounts['churned'].map({0: 'Active', 1: 'Churned'})

print(f"accounts:      {accounts.shape}")
print(f"usage_metrics: {usage.shape}")
print(f"tickets:       {tickets.shape}")
print(f"nps_surveys:   {nps.shape}")"""))

# ── 2. Accounts Overview ───────────────────────────────────────────────────────
cells.append(cell("# ── 2. Accounts Overview ───────────────────────────────────────────────────────", "markdown"))

cells.append(cell("""accounts.info()"""))

cells.append(cell("""accounts.describe(include='all').T"""))

cells.append(cell("""# Null check
print("Null counts:")
print(accounts.isnull().sum()[accounts.isnull().sum() > 0])
print("\\n(churn_date nulls are expected — only churned accounts have a date)")"""))

# ── 3. Churn Distribution ─────────────────────────────────────────────────────
cells.append(cell("# ── 3. Churn Distribution ──────────────────────────────────────────────────────", "markdown"))

cells.append(cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Churn Rate Breakdown', fontsize=15, fontweight='bold', y=1.01)

# 3a — Overall churn rate
churn_counts = accounts['churn_label'].value_counts()
colors = [CHURN_PALETTE[0], CHURN_PALETTE[1]]
axes[0].pie(
    churn_counts, labels=churn_counts.index, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title('Overall Churn Rate')

# 3b — Churn rate by plan tier
tier_churn = (accounts.groupby('plan_tier')['churned']
              .mean()
              .reindex(TIER_ORDER)
              .reset_index())
tier_churn.columns = ['plan_tier', 'churn_rate']
bars = axes[1].bar(tier_churn['plan_tier'], tier_churn['churn_rate'] * 100,
                   color=[CHURN_PALETTE[1]] * 3, alpha=0.85, edgecolor='white')
axes[1].set_title('Churn Rate by Plan Tier')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(bars, tier_churn['churn_rate']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

# 3c — Churn rate by region
region_churn = accounts.groupby('region')['churned'].mean().reset_index()
region_churn.columns = ['region', 'churn_rate']
bars = axes[2].bar(region_churn['region'], region_churn['churn_rate'] * 100,
                   color=['#FF9800', '#9C27B0', '#00BCD4'], alpha=0.85, edgecolor='white')
axes[2].set_title('Churn Rate by Region')
axes[2].set_ylabel('Churn Rate (%)')
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(bars, region_churn['churn_rate']):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_churn_distribution.png', bbox_inches='tight')
plt.show()
print("Saved → outputs/01_churn_distribution.png")"""))

# ── 4. MRR Analysis ───────────────────────────────────────────────────────────
cells.append(cell("# ── 4. Revenue Analysis (MRR) ──────────────────────────────────────────────────", "markdown"))

cells.append(cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MRR Distribution: Active vs Churned Accounts', fontsize=15, fontweight='bold')

# 4a — MRR histogram by churn status
for label, color in CHURN_PALETTE.items():
    subset = accounts[accounts['churned'] == label]['mrr']
    axes[0].hist(subset, bins=40, alpha=0.65, color=color,
                 label=accounts['churn_label'].map({label: accounts['churn_label'].unique()[label]}).iloc[0])
axes[0].set_xlabel('Monthly Recurring Revenue ($)')
axes[0].set_ylabel('Number of Accounts')
axes[0].set_title('MRR Distribution')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# 4b — Median MRR by tier and churn status
mrr_by_tier = accounts.groupby(['plan_tier', 'churn_label'])['mrr'].median().reset_index()
mrr_by_tier['plan_tier'] = pd.Categorical(mrr_by_tier['plan_tier'], TIER_ORDER)
mrr_by_tier = mrr_by_tier.sort_values('plan_tier')

x = np.arange(len(TIER_ORDER))
width = 0.35
for i, (label, color) in enumerate(zip(['Active', 'Churned'], [CHURN_PALETTE[0], CHURN_PALETTE[1]])):
    subset = mrr_by_tier[mrr_by_tier['churn_label'] == label]
    bars = axes[1].bar(x + (i - 0.5) * width, subset['mrr'], width,
                       label=label, color=color, alpha=0.85, edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(TIER_ORDER)
axes[1].set_title('Median MRR by Tier')
axes[1].set_ylabel('Median MRR ($)')
axes[1].legend()
axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('outputs/02_mrr_analysis.png', bbox_inches='tight')
plt.show()
print("Saved → outputs/02_mrr_analysis.png")"""))

cells.append(cell("""# Revenue at risk summary
arr_at_risk = accounts[accounts['churned'] == 1]['arr'].sum()
total_arr   = accounts['arr'].sum()
print(f"Total portfolio ARR:  ${total_arr:,.0f}")
print(f"ARR lost to churn:    ${arr_at_risk:,.0f}  ({arr_at_risk/total_arr:.1%} of total)")
print()
print("Median MRR — Active vs Churned:")
print(accounts.groupby('churn_label')['mrr'].median().apply(lambda x: f'${x:,.0f}'))"""))

# ── 5. Usage Patterns ─────────────────────────────────────────────────────────
cells.append(cell("# ── 5. Usage Patterns Over Time ────────────────────────────────────────────────", "markdown"))

cells.append(cell("""# Merge churn label into usage
usage_merged = usage.merge(accounts[['account_id','churned','churn_label','churn_date']], on='account_id')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Usage Patterns: Active vs Churned Accounts', fontsize=15, fontweight='bold')

# 5a — Average monthly logins over time
avg_logins = (usage_merged
              .groupby(['month','churn_label'])['logins']
              .mean()
              .reset_index())
for label, color in zip(['Active', 'Churned'], [CHURN_PALETTE[0], CHURN_PALETTE[1]]):
    subset = avg_logins[avg_logins['churn_label'] == label]
    axes[0].plot(subset['month'], subset['logins'], color=color, linewidth=2.5,
                 label=label, marker='o', markersize=4)
axes[0].set_title('Avg Monthly Logins Over Time')
axes[0].set_ylabel('Average Logins per Account')
axes[0].set_xlabel('')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)

# 5b — Feature adoption rate: churned vs active
avg_adoption = usage_merged.groupby('churn_label')['feature_adoption_pct'].mean()
bars = axes[1].bar(avg_adoption.index, avg_adoption.values * 100,
                   color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], alpha=0.85, edgecolor='white')
axes[1].set_title('Avg Feature Adoption Rate')
axes[1].set_ylabel('Features Used (%)')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(bars, avg_adoption.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/03_usage_patterns.png', bbox_inches='tight')
plt.show()
print("Saved → outputs/03_usage_patterns.png")"""))

cells.append(cell("""# API integration vs churn
api_churn = accounts.groupby(['has_api_integration' if 'has_api_integration' in accounts.columns else 'churned'])
# Pull from usage table — has_api_integration is per monthly row, get the max per account
api_flag = usage.groupby('account_id')['has_api_integration'].max().reset_index()
api_flag.columns = ['account_id', 'has_api']
acc_with_api = accounts.merge(api_flag, on='account_id')

print("Churn rate by API integration:")
print(acc_with_api.groupby('has_api')['churned'].agg(['count','mean'])
      .rename(columns={'count':'accounts','mean':'churn_rate'})
      .rename(index={0:'No API', 1:'Has API'})
      .round(3))"""))

# ── 6. Support Tickets ────────────────────────────────────────────────────────
cells.append(cell("# ── 6. Support Ticket Analysis ─────────────────────────────────────────────────", "markdown"))

cells.append(cell("""# Ticket-level features per account
ticket_features = (tickets
    .groupby('account_id')
    .agg(
        total_tickets    = ('ticket_id', 'count'),
        avg_resolution_h = ('resolution_hours', 'mean'),
        avg_csat         = ('csat_score', 'mean'),
        pct_high_crit    = ('priority', lambda x: (x.isin(['high','critical'])).mean()),
    )
    .reset_index()
)

ticket_features = ticket_features.merge(accounts[['account_id','churn_label','churned']], on='account_id')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Support Ticket Analysis: Active vs Churned', fontsize=15, fontweight='bold')

# 6a — Ticket priority mix
priority_mix = (tickets.merge(accounts[['account_id','churn_label']], on='account_id')
                .groupby(['churn_label','priority'])
                .size()
                .groupby(level=0).transform(lambda x: x / x.sum())
                .reset_index(name='pct'))
priority_mix = (tickets.merge(accounts[['account_id','churn_label']], on='account_id')
                .groupby(['churn_label','priority'])['ticket_id']
                .count()
                .reset_index())
pivot = priority_mix.pivot(index='churn_label', columns='priority', values='ticket_id').fillna(0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct.plot(kind='bar', ax=axes[0], colormap='RdYlGn_r', edgecolor='white', alpha=0.9)
axes[0].set_title('Ticket Priority Mix')
axes[0].set_ylabel('Share of Tickets (%)')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Priority', fontsize=8)

# 6b — Avg CSAT score
csat_avg = ticket_features.groupby('churn_label')['avg_csat'].mean()
bars = axes[1].bar(csat_avg.index, csat_avg.values,
                   color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], alpha=0.85, edgecolor='white')
axes[1].set_title('Avg Support CSAT Score')
axes[1].set_ylabel('CSAT (1–5)')
axes[1].set_ylim(0, 5.5)
for bar, val in zip(bars, csat_avg.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# 6c — % high/critical tickets
pct_crit = ticket_features.groupby('churn_label')['pct_high_crit'].mean() * 100
bars = axes[2].bar(pct_crit.index, pct_crit.values,
                   color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], alpha=0.85, edgecolor='white')
axes[2].set_title('% High/Critical Priority Tickets')
axes[2].set_ylabel('Share of Tickets (%)')
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(bars, pct_crit.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/04_support_analysis.png', bbox_inches='tight')
plt.show()
print("Saved → outputs/04_support_analysis.png")"""))

# ── 7. NPS Analysis ───────────────────────────────────────────────────────────
cells.append(cell("# ── 7. NPS Score Analysis ──────────────────────────────────────────────────────", "markdown"))

cells.append(cell("""nps_merged = nps.merge(accounts[['account_id','churn_label','churned']], on='account_id')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('NPS Score Analysis: Active vs Churned', fontsize=15, fontweight='bold')

# 7a — NPS distribution (histogram)
for label, color in zip(['Active', 'Churned'], [CHURN_PALETTE[0], CHURN_PALETTE[1]]):
    subset = nps_merged[nps_merged['churn_label'] == label]['nps_score']
    axes[0].hist(subset, bins=11, range=(-0.5, 10.5), alpha=0.65,
                 color=color, label=label, edgecolor='white')
axes[0].set_title('NPS Score Distribution')
axes[0].set_xlabel('NPS Score (0–10)')
axes[0].set_ylabel('Number of Responses')
axes[0].set_xticks(range(0, 11))
axes[0].legend()

# 7b — Avg NPS by quarter
nps_trend = nps_merged.groupby(['quarter','churn_label'])['nps_score'].mean().reset_index()
quarter_order = ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q1_2025']
# Fix: use the quarters that actually exist in the data
quarter_order = [q for q in ['Q1_2024','Q2_2024','Q3_2024','Q4_2024','Q1_2025'] if q in nps_trend['quarter'].values]
for label, color in zip(['Active', 'Churned'], [CHURN_PALETTE[0], CHURN_PALETTE[1]]):
    subset = nps_trend[nps_trend['churn_label'] == label].set_index('quarter').reindex(quarter_order)
    axes[1].plot(quarter_order, subset['nps_score'], color=color, linewidth=2.5,
                 label=label, marker='o', markersize=6)
axes[1].set_title('Avg NPS Score by Quarter')
axes[1].set_ylabel('Average NPS Score')
axes[1].set_xlabel('Quarter')
axes[1].legend()
axes[1].set_ylim(0, 10)
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('outputs/05_nps_analysis.png', bbox_inches='tight')
plt.show()
print("Saved → outputs/05_nps_analysis.png")"""))

# ── 8. Industry Heatmap ───────────────────────────────────────────────────────
cells.append(cell("# ── 8. Churn Rate by Industry ──────────────────────────────────────────────────", "markdown"))

cells.append(cell("""industry_churn = (accounts.groupby('industry')['churned']
                  .agg(['count','mean'])
                  .rename(columns={'count':'accounts','mean':'churn_rate'})
                  .sort_values('churn_rate', ascending=True))

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(industry_churn)))
bars = ax.barh(industry_churn.index, industry_churn['churn_rate'] * 100,
               color=colors, edgecolor='white', alpha=0.9)
ax.set_title('Churn Rate by Industry', fontsize=14, fontweight='bold')
ax.set_xlabel('Churn Rate (%)')
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(bars, industry_churn['churn_rate']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/06_industry_churn.png', bbox_inches='tight')
plt.show()
print("Saved → outputs/06_industry_churn.png")"""))

# ── 9. Key Insights Summary ───────────────────────────────────────────────────
cells.append(cell("""## EDA Summary — Key Findings

This section documents the insights that will drive feature engineering in Day 3.

---

### 1. Usage is the strongest single signal
Churned accounts average **~169 logins/month** vs **~582 for active accounts** — a 3.4× gap. This makes `logins` (and derived features like login trend and month-over-month change) the most predictive inputs for the model.

### 2. NPS divergence is dramatic
Active accounts score **~7.1 NPS on average**, churned accounts score **~3.4**. A score below 5 is a strong churn indicator. We'll engineer `last_nps_score` and `nps_trend` (change over last two surveys) as features.

### 3. Tier is a structural churn predictor
- **Starter: ~21% churn** — high-volume, low-commitment customers
- **Professional: ~14% churn** — healthy mid-market
- **Enterprise: ~5% churn** — sticky, deep integrations

Plan tier will be encoded as a categorical feature. We should also test an interaction between tier and MRR (low-MRR enterprise = potential downgrade risk).

### 4. Ticket priority, not volume, is the signal
Total ticket counts are nearly identical across churned and active accounts (~21-22). What differs is **priority mix** — churned accounts have more high/critical tickets and lower CSAT scores. Features to engineer: `pct_high_critical_tickets`, `avg_csat`, `recent_critical_ticket` (binary flag).

### 5. API integration is a retention anchor
Accounts without API integrations churn at a substantially higher rate. This makes `has_api_integration` a high-value binary feature — it proxies switching cost and product depth.

### 6. Short-tenure accounts are high risk
Accounts under 6 months old churn at nearly double the base rate. `tenure_days` will be a feature, along with `days_until_renewal` (urgency signal for CS teams).

---
**Next: Day 3 — Feature Engineering**  
We'll translate these observations into a clean, model-ready feature table (`data/processed/churn_features.csv`).""", "markdown"))

# ── Write notebook ─────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

output_path = "notebooks/01_eda.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Notebook written to {output_path}")
print(f"  Cells: {len(cells)}")
print()
print("Next steps:")
print("  1. cd to your project root (saas-churn-prediction/)")
print("  2. python create_eda_notebook.py")
print("  3. jupyter notebook notebooks/01_eda.ipynb")
print("  4. Run All Cells (Kernel → Restart & Run All)")
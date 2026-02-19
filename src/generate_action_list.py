import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

os.makedirs("outputs", exist_ok=True)
SEED = 42

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "sans-serif",
})

print("NovaCRM — CS Action List & Business Deliverables")
print("=" * 50)


# LOAD MODEL + DATA
print("\n[1] Loading XGBoost model and data...")

with open("models/xgboost.pkl", "rb") as f:
    artifact = pickle.load(f)
model        = artifact["model"]
scaler       = artifact["scaler"]
feature_names = artifact["features"]

features_df = pd.read_csv("data/processed/churn_features.csv")

accounts = pd.read_csv("data/raw/accounts.csv")

# Impute nulls 
X = features_df.drop(columns=["account_id", "churned"])
usage_cols = ["avg_monthly_logins", "avg_active_user_ratio", "avg_feature_adoption_pct",
              "avg_data_processed_gb", "total_api_calls", "has_api_integration",
              "ticket_velocity_ratio"]
X[usage_cols] = X[usage_cols].fillna(0)
X["avg_resolution_hours"] = X["avg_resolution_hours"].fillna(X["avg_resolution_hours"].median())
X_scaled = scaler.transform(X)

print(f"    Model loaded ✓ | Scoring {len(features_df):,} accounts")



# SCORE ALL ACCOUNTS
print("\n[2] Generating churn probability scores...")

churn_proba = model.predict_proba(X_scaled)[:, 1]

scores = pd.DataFrame({
    "account_id":   features_df["account_id"],
    "churn_proba":  churn_proba.round(4),
    "churned":      features_df["churned"],  
})

# Merge with account details
scores = scores.merge(
    accounts[["account_id", "company_name", "plan_tier", "mrr", "arr",
              "csm_assigned", "region", "industry", "contract_end_date"]],
    on="account_id"
)

# Risk tiers based on churn probability
def assign_risk_tier(p):
    if p >= 0.75:   return "Critical"
    elif p >= 0.50: return "High"
    elif p >= 0.25: return "Moderate"
    else:           return "Low"

scores["risk_tier"] = scores["churn_proba"].apply(assign_risk_tier)

# ARR at risk = churn probability × ARR (expected value)
scores["arr_at_risk"] = (scores["churn_proba"] * scores["arr"]).round(0)

tier_order = ["Critical", "High", "Moderate", "Low"]
tier_colors = {"Critical": "#F44336", "High": "#FF9800", "Moderate": "#FFC107", "Low": "#4CAF50"}

print(f"\n    Risk tier distribution:")
tier_summary = scores.groupby("risk_tier").agg(
    accounts   = ("account_id", "count"),
    total_arr  = ("arr", "sum"),
    arr_at_risk = ("arr_at_risk", "sum"),
    avg_churn_proba = ("churn_proba", "mean")
).reindex(tier_order).round(0)
print(tier_summary.to_string())


# CS ACTION LIST
print("\n[3] Building CS Action List...")

action_accounts = scores[scores["risk_tier"].isin(["Critical", "High"])].copy()

action_accounts = action_accounts.sort_values("arr_at_risk", ascending=False)

def recommended_action(row):
    if row["risk_tier"] == "Critical":
        if row["plan_tier"] == "enterprise":
            return "Executive escalation + QBR within 7 days"
        else:
            return "CSM intervention call within 48 hours"
    else:  # High
        if row["plan_tier"] == "enterprise":
            return "CSM check-in + usage review this week"
        else:
            return "Automated re-engagement sequence + CSM follow-up"

action_accounts["recommended_action"] = action_accounts.apply(recommended_action, axis=1)

# Final action list columns
action_list = action_accounts[[
    "account_id", "company_name", "plan_tier", "csm_assigned",
    "region", "mrr", "arr", "churn_proba", "risk_tier",
    "arr_at_risk", "contract_end_date", "recommended_action"
]].reset_index(drop=True)

action_list.index += 1  # 1-based rank
action_list.index.name = "rank"

output_path = "outputs/cs_action_list.csv"
action_list.to_csv(output_path)

print(f"\n    CS Action List: {len(action_list)} accounts")
print(f"    Total ARR at risk: ${action_list['arr_at_risk'].sum():,.0f}")
print(f"\n    Top 10 accounts by ARR at risk:")
print(action_list[["company_name","plan_tier","csm_assigned","churn_proba","risk_tier","arr_at_risk"]].head(10).to_string())
print(f"\n    Saved → {output_path}")


# VISUALIZATION 
print("\n[4] Generating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("NovaCRM Portfolio Risk — Q1 2025", fontsize=15, fontweight="bold")

# ARR at risk by tier
arr_by_tier = scores.groupby("risk_tier")["arr_at_risk"].sum().reindex(tier_order)
colors = [tier_colors[t] for t in tier_order]
bars = axes[0].bar(arr_by_tier.index, arr_by_tier.values / 1e6,
                   color=colors, alpha=0.9, edgecolor="white", linewidth=1.5)
axes[0].set_title("Expected ARR at Risk by Tier")
axes[0].set_ylabel("ARR at Risk ($M)")
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
for bar, val in zip(bars, arr_by_tier.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"${val/1e6:.2f}M", ha="center", va="bottom",
                fontweight="bold", fontsize=10)

# Account count and ARR by tier
tier_stats = scores.groupby("risk_tier").agg(
    n_accounts=("account_id", "count"),
    total_arr=("arr", "sum"),
    arr_at_risk=("arr_at_risk", "sum")
).reindex(tier_order).reset_index()

x_pos = range(len(tier_order))
scatter_colors = [tier_colors[t] for t in tier_order]
bubble_sizes = (tier_stats["arr_at_risk"] / tier_stats["arr_at_risk"].max() * 3000).values

scatter = axes[1].scatter(x_pos, tier_stats["n_accounts"],
                          s=bubble_sizes, c=scatter_colors, alpha=0.8, edgecolors="white",
                          linewidth=2)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(tier_order)
axes[1].set_title("Accounts by Risk Tier\n(bubble size = ARR at risk)")
axes[1].set_ylabel("Number of Accounts")
for i, (_, row) in enumerate(tier_stats.iterrows()):
    axes[1].annotate(f"{int(row['n_accounts'])} accounts\n${row['arr_at_risk']/1e3:.0f}K at risk",
                    (i, row["n_accounts"]), textcoords="offset points",
                    xytext=(0, 20), ha="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("outputs/16_arr_at_risk.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/16_arr_at_risk.png")


# THRESHOLD ANALYSIS

y_true  = features_df["churned"]
y_proba = churn_proba

precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Threshold Analysis — Choosing the Operating Point", fontsize=14, fontweight="bold")

# Precision & Recall vs Threshold
axes[0].plot(thresholds, precision_vals[:-1], color="#2196F3", linewidth=2.5, label="Precision")
axes[0].plot(thresholds, recall_vals[:-1], color="#F44336", linewidth=2.5, label="Recall")
axes[0].plot(thresholds, f1_vals[:-1], color="#9C27B0", linewidth=2, linestyle="--", label="F1")
axes[0].axvline(x=0.50, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Default threshold (0.50)")
axes[0].axvline(x=0.35, color="#FF9800", linestyle=":", linewidth=1.5, alpha=0.7, label="CS-optimized threshold (0.35)")
axes[0].set_xlabel("Decision Threshold")
axes[0].set_ylabel("Score")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.05)

# Accounts flagged vs ARR coverage at each threshold
threshold_range = np.arange(0.1, 0.9, 0.05)
flagged_counts = []
arr_coverage   = []
total_churner_arr = scores[scores["churned"] == 1]["arr"].sum()

for t in threshold_range:
    flagged = scores[scores["churn_proba"] >= t]
    flagged_counts.append(len(flagged))
    churner_arr_caught = flagged[flagged["churned"] == 1]["arr"].sum()
    arr_coverage.append(churner_arr_caught / total_churner_arr if total_churner_arr > 0 else 0)

ax2 = axes[1].twinx()
axes[1].plot(threshold_range, flagged_counts, color="#2196F3", linewidth=2.5, label="Accounts flagged")
ax2.plot(threshold_range, [x * 100 for x in arr_coverage], color="#F44336",
         linewidth=2.5, linestyle="--", label="% Churner ARR covered")
axes[1].axvline(x=0.50, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
axes[1].axvline(x=0.35, color="#FF9800", linestyle=":", linewidth=1.5, alpha=0.7)
axes[1].set_xlabel("Decision Threshold")
axes[1].set_ylabel("Accounts Flagged", color="#2196F3")
ax2.set_ylabel("Churner ARR Covered (%)", color="#F44336")
axes[1].set_title("Business Impact vs Threshold")
axes[1].set_xlim(0.1, 0.9)

lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/17_threshold_analysis.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/17_threshold_analysis.png")


# MODEL COMPARISON TABLE
try:
    comparison_df = pd.read_csv("outputs/model_comparison.csv")

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    display_cols = ["model", "roc_auc", "f1", "precision", "recall", "cv_roc_auc_mean", "cv_roc_auc_std"]
    display_df = comparison_df[display_cols].copy()
    display_df.columns = ["Model", "ROC-AUC", "F1", "Precision", "Recall", "CV AUC (mean)", "CV AUC (std)"]

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best model row
    best_idx = display_df["F1"].astype(float).idxmax() + 1
    row_colors = ["#E8F5E9", "#FFFFFF", "#FFF3E0"]
    for i in range(1, len(display_df) + 1):
        color = "#E8F5E9" if i == best_idx else ("#FFF8E1" if i % 2 == 0 else "#FAFAFA")
        for j in range(len(display_df.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("Model Comparison — NovaCRM Churn Prediction",
                fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("outputs/18_model_comparison_table.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("    Saved → outputs/18_model_comparison_table.png")
except Exception as e:
    print(f"    Skipped comparison table: {e}")


# FINAL SUMMARY
total_arr         = scores["arr"].sum()
critical_accounts = len(scores[scores["risk_tier"] == "Critical"])
high_accounts     = len(scores[scores["risk_tier"] == "High"])
total_at_risk_arr = scores[scores["risk_tier"].isin(["Critical","High"])]["arr_at_risk"].sum()
critical_arr      = scores[scores["risk_tier"] == "Critical"]["arr_at_risk"].sum()

print(f"""
{'='*50}
README SUMMARY STATS
{'='*50}

Portfolio:
  Total accounts:    {len(scores):,}
  Total ARR:         ${total_arr/1e6:.2f}M

Risk overview:
  Critical risk:     {critical_accounts} accounts  (${critical_arr/1e3:.0f}K ARR at risk)
  High risk:         {high_accounts} accounts
  Total ARR at risk: ${total_at_risk_arr/1e3:.0f}K (Critical + High combined)

CS Action List:
  {len(action_list)} accounts flagged for immediate intervention
  ${action_list['arr_at_risk'].sum()/1e3:.0f}K ARR prioritized for CS outreach

Model (XGBoost):
  ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}
  Top driver: days_since_last_ticket (accounts that go quiet on support)

""")

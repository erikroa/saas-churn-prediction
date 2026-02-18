
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

# Setup 
os.makedirs("models", exist_ok=True)
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

print("NovaCRM — Logistic Regression Baseline")
print("=" * 50)



# LOAD 

print("\n[1] Loading feature table...")
df = pd.read_csv("data/processed/churn_features.csv")
print(f"    Shape: {df.shape[0]:,} accounts × {df.shape[1]} columns")
print(f"    Churn rate: {df['churned'].mean():.1%}")

# Drop non-feature columns
drop_cols = ["account_id"]
X = df.drop(columns=drop_cols + ["churned"])
y = df["churned"]

feature_names = X.columns.tolist()
print(f"    Features: {len(feature_names)}")
usage_cols = ["avg_monthly_logins", "avg_active_user_ratio", "avg_feature_adoption_pct",
              "avg_data_processed_gb", "total_api_calls", "has_api_integration", "ticket_velocity_ratio"]
X[usage_cols] = X[usage_cols].fillna(0)

# Resolution hours (15 accounts)
X["avg_resolution_hours"] = X["avg_resolution_hours"].fillna(X["avg_resolution_hours"].median())

print(f"    Nulls after imputation: {X.isnull().sum().sum()}")


# TRAIN / TEST 

print("\n[2] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y          # preserves churn rate in both splits
)
print(f"    Train: {len(X_train):,} accounts ({y_train.mean():.1%} churn)")
print(f"    Test:  {len(X_test):,} accounts  ({y_test.mean():.1%} churn)")



# FEATURE SCALING

print("\n[3] Scaling features (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("    ✓ Scaler fit on training data only (no leakage)")


# TRAIN MODEL
print("\n[4] Training Logistic Regression...")
model = LogisticRegression(
    class_weight="balanced",   
    max_iter=1000,
    random_state=SEED,
    C=1.0                      
)
model.fit(X_train_scaled, y_train)
print("    ✓ Model trained")


# CROSS-VALIDATION

print("\n[5] Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = cross_validate(
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    X_train_scaled, y_train,
    cv=cv,
    scoring=["roc_auc", "f1", "precision", "recall"],
    return_train_score=False
)

print(f"\n    Cross-Validation Results (5-fold, training set only):")
print(f"    {'Metric':<12} {'Mean':>8} {'Std':>8}")
print(f"    {'-'*30}")
for metric in ["roc_auc", "f1", "precision", "recall"]:
    scores = cv_results[f"test_{metric}"]
    print(f"    {metric:<12} {scores.mean():>8.3f} {scores.std():>8.3f}")



# EVALUATE

print("\n[6] Evaluating on held-out test set...")
y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_pre = average_precision_score(y_test, y_pred_proba)
f1      = f1_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)

print(f"\n    Test Set Performance:")
print(f"    {'Metric':<25} {'Score':>8}")
print(f"    {'-'*35}")
print(f"    {'ROC-AUC':<25} {roc_auc:>8.3f}")
print(f"    {'Average Precision':<25} {avg_pre:>8.3f}")
print(f"    {'F1 Score':<25} {f1:>8.3f}")
print(f"    {'Precision':<25} {prec:>8.3f}")
print(f"    {'Recall':<25} {rec:>8.3f}")

print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))



# 7. VISUALIZATIONS

print("\n[7] Generating visualizations...")

# 7a: Matrix 
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted Active", "Predicted Churn"],
    yticklabels=["Actual Active", "Actual Churn"],
    ax=ax, linewidths=0.5, cbar=False,
    annot_kws={"size": 14, "weight": "bold"}
)
ax.set_title("Logistic Regression — Confusion Matrix\n(Test Set)", fontweight="bold")

# Annotate with business context
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.15,
        f"True Negatives: {tn}  |  False Positives: {fp}  |  False Negatives: {fn}  |  True Positives: {tp}",
        transform=ax.transAxes, ha="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("outputs/07_logistic_confusion.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/07_logistic_confusion.png")

# 7b: ROC Curve 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Logistic Regression — Model Performance", fontsize=14, fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#2196F3", linewidth=2.5,
             label=f"Logistic Regression (AUC = {roc_auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random baseline")
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.02)

# Precision-Recall curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(recall_vals, precision_vals, color="#F44336", linewidth=2.5,
             label=f"Logistic Regression (AP = {avg_pre:.3f})")
axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", linewidth=1,
                alpha=0.5, label=f"Baseline (random) = {y_test.mean():.2f}")
axes[1].fill_between(recall_vals, precision_vals, alpha=0.1, color="#F44336")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(loc="upper right")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig("outputs/08_logistic_roc.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/08_logistic_roc.png")

# 7c: Feature
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=True)

# Show top 15 most influential in each direction
top_n = 15
top_pos = coef_df.tail(top_n)
top_neg = coef_df.head(top_n)
plot_df = pd.concat([top_neg, top_pos]).drop_duplicates()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#F44336" if c > 0 else "#2196F3" for c in plot_df["coefficient"]]
bars = ax.barh(plot_df["feature"], plot_df["coefficient"],
               color=colors, alpha=0.85, edgecolor="white")
ax.axvline(x=0, color="black", linewidth=0.8, alpha=0.5)
ax.set_title("Logistic Regression — Feature Coefficients\n(Red = increases churn risk  |  Blue = reduces churn risk)",
             fontweight="bold")
ax.set_xlabel("Coefficient (standardized)")

plt.tight_layout()
plt.savefig("outputs/09_logistic_coefficients.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/09_logistic_coefficients.png")


# SAVE MODEL 

print("\n[8] Saving model artifacts...")

with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": feature_names}, f)
print("    Saved → models/logistic_regression.pkl")

# Save performance summary as CSV
perf = pd.DataFrame([{
    "model":             "Logistic Regression",
    "roc_auc":           round(roc_auc, 3),
    "average_precision": round(avg_pre, 3),
    "f1":                round(f1, 3),
    "precision":         round(prec, 3),
    "recall":            round(rec, 3),
    "cv_roc_auc_mean":   round(cv_results["test_roc_auc"].mean(), 3),
    "cv_roc_auc_std":    round(cv_results["test_roc_auc"].std(), 3),
}])
perf.to_csv("outputs/model_comparison.csv", index=False)
print("    Saved → outputs/model_comparison.csv")


# BUSINESS INTERPRETATION

print("\n" + "=" * 50)
print("BUSINESS INTERPRETATION")
print("=" * 50)

print(f"""
Model: Logistic Regression (balanced class weights)
Test set: {len(y_test):,} accounts

Key metrics:
  ROC-AUC {roc_auc:.3f} — the model ranks churners above non-churners
  {roc_auc:.0%} of the time (1.0 = perfect, 0.5 = random)

  Recall  {rec:.3f} — of all accounts that actually churned,
  the model correctly identified {rec:.0%} of them

  Precision {prec:.3f} — when the model flags an account as at-risk,
  it is correct {prec:.0%} of the time

Top churn risk drivers (positive coefficients):""")

top_churn_drivers = coef_df.tail(5).iloc[::-1]
for _, row in top_churn_drivers.iterrows():
    print(f"  + {row['feature']:<35} coef: {row['coefficient']:+.3f}")

print("\nTop churn protection factors (negative coefficients):")
top_protectors = coef_df.head(5)
for _, row in top_protectors.iterrows():
    print(f"  - {row['feature']:<35} coef: {row['coefficient']:+.3f}")

print(f"""
Baseline comparison:
  A naive model predicting "no churn" for everyone would achieve:
  - Accuracy: {1 - y_test.mean():.1%} (misleading — catches 0 churners)
  - Recall:   0.0% (useless for CS teams)

  This model achieves {rec:.0%} recall, meaning CS teams using this
  model would have visibility into {rec:.0%} of at-risk accounts.

""")
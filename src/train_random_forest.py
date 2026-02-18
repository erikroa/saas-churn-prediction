

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

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

print("NovaCRM — Random Forest Model")
print("=" * 50)



# LOAD & PREPARE DATA
print("\n[1] Loading feature table...")
df = pd.read_csv("data/processed/churn_features.csv")
print(f"    Shape: {df.shape[0]:,} accounts × {df.shape[1]} columns")

X = df.drop(columns=["account_id", "churned"])
y = df["churned"]
feature_names = X.columns.tolist()

# Impute nulls
usage_cols = ["avg_monthly_logins", "avg_active_user_ratio", "avg_feature_adoption_pct",
              "avg_data_processed_gb", "total_api_calls", "has_api_integration",
              "ticket_velocity_ratio"]
X[usage_cols] = X[usage_cols].fillna(0)
X["avg_resolution_hours"] = X["avg_resolution_hours"].fillna(X["avg_resolution_hours"].median())
print(f"    Features: {len(feature_names)} | Nulls: {X.isnull().sum().sum()}")



# TRAIN / TEST SPLIT

print("\n[2] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

# Note: Random Forest doesn't require scaling, but we scale anyway
# so models are directly comparable and the pkl is consistent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# HYPERPARAMETER TUNING (GridSearchCV)
print("\n[3] Hyperparameter tuning (GridSearchCV — this may take ~60 seconds)...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 10, 20],
    "min_samples_leaf": [1, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=SEED),
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

print(f"    Best params:   {grid_search.best_params_}")
print(f"    Best CV AUC:   {grid_search.best_score_:.3f}")
model = grid_search.best_estimator_


# CROSS-VALIDATION WITH BEST MODEL
print("\n[4] Cross-validation with best model...")
cv_results = cross_validate(
    model, X_train_scaled, y_train,
    cv=cv,
    scoring=["roc_auc", "f1", "precision", "recall"],
    return_train_score=False
)

print(f"\n    Cross-Validation Results (5-fold):")
print(f"    {'Metric':<12} {'Mean':>8} {'Std':>8}")
print(f"    {'-'*30}")
for metric in ["roc_auc", "f1", "precision", "recall"]:
    scores = cv_results[f"test_{metric}"]
    print(f"    {metric:<12} {scores.mean():>8.3f} {scores.std():>8.3f}")


# EVALUATE ON TEST SET
print("\n[5] Evaluating on held-out test set...")
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


# VISUALIZATIONS
print("\n[6] Generating visualizations...")

# 6a: Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Predicted Active", "Predicted Churn"],
            yticklabels=["Actual Active", "Actual Churn"],
            ax=ax, linewidths=0.5, cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("Random Forest — Confusion Matrix\n(Test Set)", fontweight="bold")
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.15,
        f"TN: {tn}  |  FP: {fp}  |  FN: {fn}  |  TP: {tp}",
        transform=ax.transAxes, ha="center", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig("outputs/10_rf_confusion.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/10_rf_confusion.png")

# 6b: ROC + PR curves 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Random Forest — Model Performance", fontsize=14, fontweight="bold")

# Load logistic results for comparison
try:
    lr_comparison = pd.read_csv("outputs/model_comparison.csv")
    lr_auc = lr_comparison.loc[lr_comparison["model"] == "Logistic Regression", "roc_auc"].values[0]
    show_comparison = True
except:
    show_comparison = False

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#4CAF50", linewidth=2.5,
             label=f"Random Forest (AUC = {roc_auc:.3f})")
if show_comparison:
    axes[0].axhline(y=0, color="white")  # spacer
    axes[0].plot([], [], color="#2196F3", linewidth=2, linestyle="--",
                 label=f"Logistic Regression (AUC = {lr_auc:.3f})")
axes[0].plot([0,1],[0,1],"k--", linewidth=1, alpha=0.4, label="Random baseline")
axes[0].fill_between(fpr, tpr, alpha=0.08, color="#4CAF50")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")

prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(rec_vals, prec_vals, color="#4CAF50", linewidth=2.5,
             label=f"Random Forest (AP = {avg_pre:.3f})")
axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", linewidth=1,
                alpha=0.4, label=f"Baseline = {y_test.mean():.2f}")
axes[1].fill_between(rec_vals, prec_vals, alpha=0.08, color="#4CAF50")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("outputs/11_rf_roc.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/11_rf_roc.png")

# 6c: Feature Importances
importance_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importance_df["feature"], importance_df["importance"],
               color="#4CAF50", alpha=0.85, edgecolor="white")
ax.set_title("Random Forest — Top 15 Feature Importances\n(Mean decrease in impurity)",
             fontweight="bold")
ax.set_xlabel("Feature Importance")
for bar, val in zip(bars, importance_df["importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/12_rf_feature_importance.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/12_rf_feature_importance.png")

# SAVE MODEL

print("\n[7] Saving artifacts...")
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": feature_names}, f)
print("    Saved → models/random_forest.pkl")

# Update model comparison CSV
new_row = pd.DataFrame([{
    "model":             "Random Forest",
    "roc_auc":           round(roc_auc, 3),
    "average_precision": round(avg_pre, 3),
    "f1":                round(f1, 3),
    "precision":         round(prec, 3),
    "recall":            round(rec, 3),
    "cv_roc_auc_mean":   round(cv_results["test_roc_auc"].mean(), 3),
    "cv_roc_auc_std":    round(cv_results["test_roc_auc"].std(), 3),
}])
try:
    existing = pd.read_csv("outputs/model_comparison.csv")
    existing = existing[existing["model"] != "Random Forest"]
    comparison = pd.concat([existing, new_row], ignore_index=True)
except:
    comparison = new_row
comparison.to_csv("outputs/model_comparison.csv", index=False)

print("\n    Model Comparison Table:")
print(comparison[["model","roc_auc","f1","precision","recall","cv_roc_auc_mean"]].to_string(index=False))


# BUSINESS INTERPRETATION

print("\n" + "=" * 50)
print("BUSINESS INTERPRETATION")
print("=" * 50)

top5 = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(5)

print(f"""
Model: Random Forest (best params: {grid_search.best_params_})
Test set: {len(y_test):,} accounts

Key metrics:
  ROC-AUC   {roc_auc:.3f}
  F1 Score  {f1:.3f}
  Recall    {rec:.3f}  ({rec:.0%} of churners caught)
  Precision {prec:.3f}  ({prec:.0%} of flagged accounts truly at risk)

Top 5 features by importance:""")
for _, row in top5.iterrows():
    print(f"  {row['feature']:<35} {row['importance']:.3f}")

print(f"""
vs Logistic Regression baseline:
  Random Forest {'outperforms' if roc_auc > lr_auc else 'similar to'} Logistic Regression
  on ROC-AUC ({roc_auc:.3f} vs {lr_auc:.3f})

  The gap {'confirms' if roc_auc > lr_auc else 'suggests'} that churn prediction benefits from
  {'capturing non-linear feature interactions.' if roc_auc > lr_auc else 'linear relationships being dominant in this dataset.'}

""")

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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier

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

print("NovaCRM — XGBoost Model")
print("=" * 50)


# LOAD 
print("\n[1] Loading feature table...")
df = pd.read_csv("data/processed/churn_features.csv")

X = df.drop(columns=["account_id", "churned"])
y = df["churned"]
feature_names = X.columns.tolist()

usage_cols = ["avg_monthly_logins", "avg_active_user_ratio", "avg_feature_adoption_pct",
              "avg_data_processed_gb", "total_api_calls", "has_api_integration",
              "ticket_velocity_ratio"]
X[usage_cols] = X[usage_cols].fillna(0)
X["avg_resolution_hours"] = X["avg_resolution_hours"].fillna(X["avg_resolution_hours"].median())

print(f"    Features: {len(feature_names)} | Nulls: {X.isnull().sum().sum()}")

# Class imbalance ratio
neg = (y == 0).sum()
pos = (y == 1).sum()
scale_pos_weight = round(neg / pos, 2)
print(f"    Class ratio (neg/pos): {scale_pos_weight} → used as scale_pos_weight")


# TRAIN / TEST 
print("\n[2] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")


# HYPERPARAMETER TUNING
print("\n[3] Hyperparameter tuning (GridSearchCV - ~90 seconds)...")

param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(
    XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    ),
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

print(f"    Best params: {grid_search.best_params_}")
print(f"    Best CV AUC: {grid_search.best_score_:.3f}")
model = grid_search.best_estimator_


# CROSS-VALIDATION
print("\n[4] Cross-validation with best model...")
cv_results = cross_validate(
    model, X_train_scaled, y_train,
    cv=cv,
    scoring=["roc_auc", "f1", "precision", "recall"],
)

print(f"\n    Cross-Validation Results (5-fold):")
print(f"    {'Metric':<12} {'Mean':>8} {'Std':>8}")
print(f"    {'-'*30}")
for metric in ["roc_auc", "f1", "precision", "recall"]:
    scores = cv_results[f"test_{metric}"]
    print(f"    {metric:<12} {scores.mean():>8.3f} {scores.std():>8.3f}")


# EVALUATE 
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

# Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Predicted Active", "Predicted Churn"],
            yticklabels=["Actual Active", "Actual Churn"],
            ax=ax, linewidths=0.5, cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("XGBoost — Confusion Matrix\n(Test Set)", fontweight="bold")
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.15, f"TN: {tn}  |  FP: {fp}  |  FN: {fn}  |  TP: {tp}",
        transform=ax.transAxes, ha="center", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig("outputs/13_xgb_confusion.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/13_xgb_confusion.png")

#Model ROC Comparison 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("All Models — Performance Comparison", fontsize=14, fontweight="bold")

model_styles = {
    "Logistic Regression": {"color": "#2196F3", "ls": "--"},
    "Random Forest":       {"color": "#4CAF50", "ls": "-."},
    "XGBoost":             {"color": "#FF9800", "ls": "-"},
}

# XGBoost ROC 
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr_xgb, tpr_xgb, color="#FF9800", linewidth=2.5,
             label=f"XGBoost (AUC = {roc_auc:.3f})")

# Load comparison table 
try:
    comparison_df = pd.read_csv("outputs/model_comparison.csv")
    for _, row in comparison_df.iterrows():
        if row["model"] in model_styles and row["model"] != "XGBoost":
            style = model_styles[row["model"]]
            axes[0].plot([], [], color=style["color"], linestyle=style["ls"],
                         linewidth=2, label=f"{row['model']} (AUC = {row['roc_auc']:.3f})")
except:
    pass

axes[0].plot([0,1],[0,1], "k--", linewidth=1, alpha=0.4, label="Random baseline")
axes[0].fill_between(fpr_xgb, tpr_xgb, alpha=0.08, color="#FF9800")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — All Models")
axes[0].legend(loc="lower right", fontsize=9)

# Precision-Recall
prec_vals, rec_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(rec_vals, prec_vals, color="#FF9800", linewidth=2.5,
             label=f"XGBoost (AP = {avg_pre:.3f})")
axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", linewidth=1,
                alpha=0.4, label=f"Baseline = {y_test.mean():.2f}")
axes[1].fill_between(rec_vals, prec_vals, alpha=0.08, color="#FF9800")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — XGBoost")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("outputs/14_xgb_roc_comparison.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/14_xgb_roc_comparison.png")

#Feature Importance 
importance_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importance_df["feature"], importance_df["importance"],
               color="#FF9800", alpha=0.85, edgecolor="white")
ax.set_title("XGBoost — Top 15 Feature Importances",fontweight="bold")
ax.set_xlabel("Feature Importance (gain)")
for bar, val in zip(bars, importance_df["importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/15_xgb_feature_importance.png", bbox_inches="tight")
plt.close()
print("    Saved → outputs/15_xgb_feature_importance.png")


# SAVE + UPDATE
print("\n[7] Saving artifacts...")
with open("models/xgboost.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": feature_names}, f)
print("    Saved → models/xgboost.pkl")

new_row = pd.DataFrame([{
    "model":             "XGBoost",
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
    existing = existing[existing["model"] != "XGBoost"]
    comparison = pd.concat([existing, new_row], ignore_index=True)
except:
    comparison = new_row
comparison.to_csv("outputs/model_comparison.csv", index=False)

print("\n    Final Model Comparison")
print(comparison[["model","roc_auc","f1","precision","recall","cv_roc_auc_mean"]].to_string(index=False))


# WINNER SELECTION + BUSINESS 
best_model = comparison.loc[comparison["f1"].idxmax(), "model"]
print(f"\n    ✓ Best model by F1: {best_model}")

top5 = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(5)

print(f"""
{'='*50}
BUSINESS INTERPRETATION
{'='*50}

Model: XGBoost (params: {grid_search.best_params_})
Best overall model: {best_model} (by F1 score)

XGBoost key metrics:
  ROC-AUC   {roc_auc:.3f}
  F1        {f1:.3f}
  Recall    {rec:.3f}  ({rec:.0%} of churners identified)
  Precision {prec:.3f}  ({prec:.0%} accuracy when flagging at-risk)

Top churn predictors:""")
for _, row in top5.iterrows():
    print(f"  {row['feature']:<35} {row['importance']:.3f}")

print(f"""
Model selection rationale:
  Use {best_model} for the CS Action List.
  It achieves the best balance of catching churners (recall)
  while keeping false alarms manageable (precision).
""")

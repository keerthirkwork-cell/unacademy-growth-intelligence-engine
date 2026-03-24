"""
Layer 3 — Churn Prediction Model (Random Forest)
Predicts: Will this paid user churn in the next 30 days?
Output: churn_probability score per user + feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils import resample
import pickle, warnings
warnings.filterwarnings("ignore")

BRAND  = "#1A56A0"; ACCENT = "#F97316"; GREEN = "#16A34A"
RED    = "#DC2626";  GREY   = "#6B7280"; LIGHT = "#EAF1FB"; DARK = "#111827"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white"})

df = pd.read_csv("/home/claude/growth_engine/data/learner_data.csv")
paid = df[df["tier"] != "Free"].copy()

# ── FEATURES ─────────────────────────────────────────────────────────────────
FEATURES = [
    "lessons_last_30d", "mock_tests_last_30d", "days_since_last_login",
    "avg_session_min", "streak_days", "login_freq_per_week",
    "months_subscribed", "lessons_total", "mock_tests_total",
    "doubt_sessions", "notes_saved", "videos_rewatched",
    "days_active", "total_time_hrs"
]
FEATURE_LABELS = {
    "lessons_last_30d":       "Lessons (last 30d)",
    "mock_tests_last_30d":    "Mock Tests (last 30d)",
    "days_since_last_login":  "Days Since Last Login",
    "avg_session_min":        "Avg Session Duration (min)",
    "streak_days":            "Streak Days",
    "login_freq_per_week":    "Login Frequency / Week",
    "months_subscribed":      "Months Subscribed",
    "lessons_total":          "Total Lessons Watched",
    "mock_tests_total":       "Total Mock Tests",
    "doubt_sessions":         "Doubt Sessions",
    "notes_saved":            "Notes Saved",
    "videos_rewatched":       "Videos Rewatched",
    "days_active":            "Total Active Days",
    "total_time_hrs":         "Total Time on Platform (hrs)",
}

X = paid[FEATURES].fillna(0)
y = paid["churned"]

# ── HANDLE CLASS IMBALANCE ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Upsample minority class in training set
X_tr_df = X_train.copy(); X_tr_df["churned"] = y_train.values
majority = X_tr_df[X_tr_df["churned"]==0]
minority = X_tr_df[X_tr_df["churned"]==1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced = pd.concat([majority, minority_up])
X_train_b = balanced[FEATURES]
y_train_b  = balanced["churned"]

# ── TRAIN RANDOM FOREST ───────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_b, y_train_b)

y_pred      = rf.predict(X_test)
y_prob      = rf.predict_proba(X_test)[:,1]
auc_score   = roc_auc_score(y_test, y_prob)
cv_scores   = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")

print(f"✅ Model trained")
print(f"   AUC-ROC:        {auc_score:.3f}")
print(f"   CV AUC (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ── SAVE MODEL + PREDICTIONS ──────────────────────────────────────────────────
with open("/home/claude/growth_engine/models/churn_model.pkl","wb") as f:
    pickle.dump(rf, f)

# Predict on ALL paid users
paid["churn_probability"] = rf.predict_proba(paid[FEATURES].fillna(0))[:,1]
paid["churn_probability"] = paid["churn_probability"].round(3)
paid.to_csv("/home/claude/growth_engine/data/paid_with_predictions.csv", index=False)

# Merge predictions back to main df
if "churn_probability" in df.columns:
    df = df.drop(columns=["churn_probability"])
df = df.merge(paid[["learner_id","churn_probability"]], on="learner_id", how="left")
df["churn_probability"] = df["churn_probability"].fillna(0)
df.to_csv("/home/claude/growth_engine/data/learner_data.csv", index=False)
print("✅ Predictions saved")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Layer 3 — Churn Prediction Model (Random Forest)", fontsize=16, fontweight="bold", color=DARK)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1. Feature Importance
ax1 = fig.add_subplot(gs[0, :2])
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
labels = [FEATURE_LABELS[f] for f in importances.index]
colors = [RED if "login" in f or "last" in f else BRAND for f in importances.index]
bars = ax1.barh(labels, importances.values, color=colors, edgecolor="white", height=0.65)
ax1.set_title("Feature Importance — What Drives Churn?", fontsize=12, fontweight="bold")
ax1.set_xlabel("Importance Score")
top_feat = FEATURE_LABELS[importances.index[-1]]
ax1.text(0.98, 0.02, f"Top signal: {top_feat}", transform=ax1.transAxes,
         ha="right", fontsize=9, color=RED, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=RED, alpha=0.9))

red_patch   = mpatches.Patch(color=RED,   label="Churn risk signals")
blue_patch  = mpatches.Patch(color=BRAND, label="Retention signals")
ax1.legend(handles=[red_patch, blue_patch], fontsize=9)

# 2. ROC Curve
ax2 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color=BRAND, lw=2.5, label=f"RF Model (AUC = {auc_score:.3f})")
ax2.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
ax2.fill_between(fpr, tpr, alpha=0.08, color=BRAND)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_facecolor(LIGHT)

# 3. Churn probability distribution
ax3 = fig.add_subplot(gs[1, 0])
churned_p     = paid[paid["churned"]==1]["churn_probability"]
not_churned_p = paid[paid["churned"]==0]["churn_probability"]
ax3.hist(not_churned_p, bins=30, alpha=0.65, color=GREEN, label="Active",  edgecolor="white")
ax3.hist(churned_p,     bins=30, alpha=0.65, color=RED,   label="Churned", edgecolor="white")
ax3.axvline(0.5, color=DARK, linestyle="--", lw=1.5, label="Decision threshold (0.5)")
ax3.set_title("Predicted Churn Probability\nDistribution", fontsize=12, fontweight="bold")
ax3.set_xlabel("Churn Probability"); ax3.set_ylabel("Users")
ax3.legend(fontsize=9)
ax3.set_facecolor(LIGHT)

# 4. Risk tier breakdown
ax4 = fig.add_subplot(gs[1, 1])
paid2 = paid.copy()
paid2["risk_tier"] = pd.cut(paid2["churn_probability"],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=["🟢 Low (<30%)", "🟡 Medium (30–50%)", "🟠 High (50–70%)", "🔴 Critical (>70%)"])
risk_counts = paid2["risk_tier"].value_counts().sort_index()
risk_colors = [GREEN, "#FBBF24", ACCENT, RED]
bars4 = ax4.bar(range(len(risk_counts)), risk_counts.values,
                color=risk_colors, edgecolor="white", width=0.6)
ax4.set_xticks(range(len(risk_counts)))
ax4.set_xticklabels([r.split(" ")[0]+" "+r.split("(")[1].rstrip(")") for r in risk_counts.index], fontsize=8)
for bar, val in zip(bars4, risk_counts.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
             f"{val:,}", ha="center", fontsize=10, fontweight="bold")
ax4.set_title("Users by Churn Risk Tier", fontsize=12, fontweight="bold")
ax4.set_ylabel("Number of Paid Users")
ax4.set_facecolor(LIGHT)

# Monthly revenue at risk per tier
critical = paid2[paid2["risk_tier"]=="🔴 Critical (>70%)"]["monthly_spend_inr"].sum()
high     = paid2[paid2["risk_tier"]=="🟠 High (50–70%)"]["monthly_spend_inr"].sum()
ax4.text(0.5, 0.92,
         f"Critical + High risk: ₹{(critical+high)/1e5:.1f}L/month at risk",
         transform=ax4.transAxes, ha="center", fontsize=9, color=RED,
         fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=RED, alpha=0.8))

# 5. Confusion matrix
ax5 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Active","Churned"])
disp.plot(ax=ax5, colorbar=False, cmap="Blues")
ax5.set_title(f"Confusion Matrix\n(AUC = {auc_score:.3f}, CV = {cv_scores.mean():.3f})",
              fontsize=12, fontweight="bold")

plt.savefig("/home/claude/growth_engine/charts/layer3_churn_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Layer 3 chart saved")

# Print high-risk summary
print(f"\n── High-Risk Users (churn prob > 0.7) ──")
critical_users = paid[paid["churn_probability"] > 0.7]
print(f"Count:             {len(critical_users):,}")
print(f"Revenue at risk:   ₹{critical_users['monthly_spend_inr'].sum():,.0f}/month")
print(f"Top exam category: {critical_users['exam_category'].value_counts().index[0]}")

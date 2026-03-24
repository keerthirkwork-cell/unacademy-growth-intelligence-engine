"""
Layer 2 — Learner Behavioural Segmentation (KMeans)
Segments: Serious Aspirants / High Intent / At-Risk / Casual Browsers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

BRAND   = "#1A56A0"
ACCENT  = "#F97316"
GREEN   = "#16A34A"
RED     = "#DC2626"
PURPLE  = "#7C3AED"
GREY    = "#6B7280"
LIGHT   = "#EAF1FB"
DARK    = "#111827"

plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white"})

df = pd.read_csv("/home/claude/growth_engine/data/learner_data.csv")

# ── FEATURES FOR CLUSTERING ───────────────────────────────────────────────────
# Use ALL users (free + paid) — clustering is about behaviour, not tier
features = [
    "lessons_total", "mock_tests_total", "doubt_sessions",
    "days_active", "days_since_last_login", "avg_session_min",
    "streak_days", "login_freq_per_week", "notes_saved", "videos_rewatched"
]

X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── ELBOW METHOD (saved but not shown in main charts) ─────────────────────────
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# ── FIT KMEANS (4 clusters = meaningful business segments) ────────────────────
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)

# ── LABEL CLUSTERS by profile ────────────────────────────────────────────────
cluster_means = df.groupby("cluster")[features + ["churned","monthly_spend_inr"]].mean()

# Assign meaningful names based on engagement + recency
# High lessons + low recency gap + high mock = Serious
# High lessons + high conv = High Intent
# High recency gap + low engagement = At-Risk
# Low everything = Casual

# Score each cluster: high lessons + high mock + low recency_gap = best
cluster_means["eng_score"] = (
    cluster_means["lessons_total"].rank() +
    cluster_means["mock_tests_total"].rank() +
    cluster_means["streak_days"].rank() -
    cluster_means["days_since_last_login"].rank()
)
ranked = cluster_means["eng_score"].rank(ascending=False).astype(int)
label_map = {
    ranked[ranked==1].index[0]: "🧠 Serious Aspirants",
    ranked[ranked==2].index[0]: "🚀 High Intent",
    ranked[ranked==3].index[0]: "⚠️ At-Risk",
    ranked[ranked==4].index[0]: "😴 Casual Browsers",
}
df["segment"] = df["cluster"].map(label_map)

SEGMENT_COLORS = {
    "🧠 Serious Aspirants": BRAND,
    "🚀 High Intent":       GREEN,
    "⚠️ At-Risk":           RED,
    "😴 Casual Browsers":   GREY,
}

# ── PCA for 2D viz ────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
df["pca1"] = X_2d[:, 0]
df["pca2"] = X_2d[:, 1]

# ── SAVE ENRICHED DATA ────────────────────────────────────────────────────────
df.to_csv("/home/claude/growth_engine/data/learner_data.csv", index=False)
print("✅ Segments added to dataset")

# ── CHARTS ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Layer 2 — Learner Behavioural Segmentation (KMeans)", fontsize=16, fontweight="bold", color=DARK)

segments = df["segment"].value_counts()

# Top-left: PCA scatter
ax = axes[0][0]
for seg, color in SEGMENT_COLORS.items():
    mask = df["segment"] == seg
    ax.scatter(df.loc[mask,"pca1"], df.loc[mask,"pca2"],
               c=color, s=6, alpha=0.35, label=seg)
ax.set_title("Behavioural Clusters (PCA 2D)", fontsize=12, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(fontsize=8, markerscale=3)

# Top-right: Segment size
ax = axes[0][1]
colors = [SEGMENT_COLORS[s] for s in segments.index]
bars = ax.bar(range(len(segments)), segments.values, color=colors, edgecolor="white", width=0.6)
ax.set_xticks(range(len(segments)))
ax.set_xticklabels([s.split(" ",1)[1] for s in segments.index], fontsize=10)
for bar, val in zip(bars, segments.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
            f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
ax.set_title("Segment Size Distribution", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Learners")
ax.set_facecolor(LIGHT)

# Bottom-left: Conversion rate by segment
ax = axes[1][0]
conv = df.groupby("segment").apply(lambda x: (x["tier"]!="Free").mean() * 100).reset_index()
conv.columns = ["segment","conversion_rate"]
conv = conv.sort_values("conversion_rate", ascending=False)
colors2 = [SEGMENT_COLORS[s] for s in conv["segment"]]
bars2 = ax.bar(range(len(conv)), conv["conversion_rate"], color=colors2, edgecolor="white", width=0.6)
ax.set_xticks(range(len(conv)))
ax.set_xticklabels([s.split(" ",1)[1] for s in conv["segment"]], fontsize=9)
for bar, val in zip(bars2, conv["conversion_rate"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Paid Conversion Rate by Segment", fontsize=12, fontweight="bold")
ax.set_ylabel("Conversion Rate (%)")
ax.set_facecolor(LIGHT)

# Key insight annotation
top2_segs  = conv.head(2)
top2_names = " vs ".join([s.split(" ",1)[1] for s in top2_segs["segment"]])
top1_conv  = top2_segs.iloc[0]["conversion_rate"]
top2_conv  = top2_segs.iloc[1]["conversion_rate"]
ax.text(0.5, 0.92,
        f"Top segment converts at {top1_conv:.0f}% vs {top2_conv:.0f}% — {top1_conv/max(top2_conv,1):.1f}× gap",
        transform=ax.transAxes, ha="center", fontsize=9, color=BRAND,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, edgecolor=BRAND, alpha=0.8))

# Bottom-right: Revenue concentration
ax = axes[1][1]
rev = df.groupby("segment")["ltv_inr"].sum().reset_index()
rev["pct"] = rev["ltv_inr"] / rev["ltv_inr"].sum() * 100
rev = rev.sort_values("ltv_inr", ascending=False)
wedge_colors = [SEGMENT_COLORS[s] for s in rev["segment"]]
wedges, texts, autotexts = ax.pie(
    rev["ltv_inr"], labels=[s.split(" ",1)[1] for s in rev["segment"]],
    colors=wedge_colors, autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor":"white","linewidth":2})
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold")
ax.set_title("Revenue Share by Segment\n(LTV contribution)", fontsize=12, fontweight="bold")

top2_rev_pct = rev.head(2)["pct"].sum()
ax.text(0, -1.5,
        f"Top 2 segments drive {top2_rev_pct:.0f}% of total revenue",
        ha="center", fontsize=9, fontweight="bold", color=BRAND)

plt.tight_layout()
plt.savefig("/home/claude/growth_engine/charts/layer2_segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Layer 2 chart saved")

# Print segment summary
print("\n── Segment Summary ──")
summary = df.groupby("segment").agg(
    count=("learner_id","count"),
    conversion_rate=("tier", lambda x: (x!="Free").mean()),
    avg_ltv=("ltv_inr","mean"),
    churn_rate=("churned","mean"),
    avg_lessons=("lessons_total","mean"),
).round(2)
print(summary.to_string())

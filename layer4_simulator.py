"""
Layer 4 — Revenue Impact Simulator
"What does each 1% improvement actually mean in ₹?"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

BRAND="#1A56A0"; ACCENT="#F97316"; GREEN="#16A34A"; RED="#DC2626"
PURPLE="#7C3AED"; GREY="#6B7280"; LIGHT="#EAF1FB"; DARK="#111827"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white"})

df   = pd.read_csv("/home/claude/growth_engine/data/learner_data.csv")
paid = df[df["tier"]!="Free"].copy()

free_count    = (df["tier"]=="Free").sum()
total         = len(df)
plus_price    = 1562
iconic_price  = 2396
plus_count    = (df["tier"]=="Plus").sum()
iconic_count  = (df["tier"]=="Iconic").sum()
paid_count    = len(paid)
current_rev_mo = paid["monthly_spend_inr"].sum()
churn_rev_risk = paid[paid["churned"]==1]["monthly_spend_inr"].sum()
high_risk_rev  = paid[paid["churn_probability"]>=0.5]["monthly_spend_inr"].sum()

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Layer 4 — Revenue Impact Simulator\n\"What does each 1% improvement mean in ₹?\"",
             fontsize=16, fontweight="bold", color=DARK)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

def inr(x):
    if x>=1e7: return f"₹{x/1e7:.1f}Cr"
    if x>=1e5: return f"₹{x/1e5:.1f}L"
    return f"₹{x/1e3:.0f}K"

# ── Sim 1: Free → Plus conversion uplift ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0,:2])
uplifts    = np.arange(0.5, 11, 0.5)
mo_revs    = [free_count * u/100 * plus_price for u in uplifts]
ann_revs   = [r*12 for r in mo_revs]
colors_bar = [GREEN if u<=3 else ACCENT if u<=6 else RED for u in uplifts]
bars = ax1.bar(uplifts, ann_revs, color=colors_bar, width=0.4, edgecolor="white")

ax1.set_title("Scenario 1 — Free-to-Plus Conversion Uplift → Annual Revenue Impact",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Conversion Rate Improvement (%)")
ax1.set_ylabel("Additional Annual Revenue")
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: inr(x)))

# Annotate key milestones
for u, rev in zip(uplifts, ann_revs):
    if u in [1, 3, 5, 10]:
        ax1.text(u, rev+2e5, inr(rev), ha="center", fontsize=8.5, fontweight="bold", color=DARK)

patches = [mpatches.Patch(color=GREEN,  label="Achievable (≤3%)"),
           mpatches.Patch(color=ACCENT, label="Stretch (3–6%)"),
           mpatches.Patch(color=RED,    label="Ambitious (>6%)")]
ax1.legend(handles=patches, fontsize=9)
ax1.set_facecolor(LIGHT)

# ── Sim 2: Churn reduction impact ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0,2])
churn_reductions = [5, 10, 15, 20, 25, 30]
saved_revs = [churn_rev_risk * r/100 * 12 for r in churn_reductions]
bars2 = ax2.bar([f"{r}%" for r in churn_reductions], saved_revs,
                color=[BRAND]*len(churn_reductions), edgecolor="white", alpha=0.85, width=0.6)
for bar, rev in zip(bars2, saved_revs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1e4,
             inr(rev), ha="center", fontsize=9, fontweight="bold")
bars2[1].set_color(ACCENT); bars2[1].set_alpha(1)
ax2.set_title("Scenario 2 — Churn Reduction\n→ Annual Revenue Saved", fontsize=11, fontweight="bold")
ax2.set_xlabel("Churn Rate Reduced By")
ax2.set_ylabel("Annual Revenue Saved")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: inr(x)))
ax2.set_facecolor(LIGHT)

# ── Sim 3: Plus → Iconic upgrade ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1,:2])
upgrade_pcts = np.arange(1, 16)
upgrade_delta = iconic_price - plus_price  # ₹834/month incremental
upgrade_revs  = [plus_count * u/100 * upgrade_delta * 12 for u in upgrade_pcts]
ax3.plot(upgrade_pcts, upgrade_revs, color=PURPLE, lw=2.5, marker="o", markersize=6)
ax3.fill_between(upgrade_pcts, upgrade_revs, alpha=0.12, color=PURPLE)
for u, rev in zip(upgrade_pcts, upgrade_revs):
    if u in [3, 7, 10, 15]:
        ax3.annotate(inr(rev), (u, rev), textcoords="offset points",
                     xytext=(5,8), fontsize=9, fontweight="bold", color=PURPLE)
ax3.set_title("Scenario 3 — Plus → Iconic Upgrade\n(₹834/month incremental per user)",
              fontsize=12, fontweight="bold")
ax3.set_xlabel("% of Plus Users Upgraded to Iconic")
ax3.set_ylabel("Additional Annual Revenue")
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x,_: inr(x)))
ax3.axvline(5, color=PURPLE, linestyle="--", alpha=0.5, lw=1)
ax3.text(5.2, max(upgrade_revs)*0.3, "5% upgrade\ntarget", fontsize=9, color=PURPLE)
ax3.set_facecolor(LIGHT)

# ── Sim 4: Combined scenario waterfall ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1,2])
scenarios_combined = {
    "Current\nRevenue":    current_rev_mo * 12,
    "+3% Free\nConversion": free_count * 0.03 * plus_price * 12,
    "10% Churn\nReduction": churn_rev_risk * 0.10 * 12,
    "5% Plus→\nIconic":    plus_count * 0.05 * upgrade_delta * 12,
    "Combined\nRevenue":   (current_rev_mo*12) + (free_count*0.03*plus_price*12) +
                            (churn_rev_risk*0.10*12) + (plus_count*0.05*upgrade_delta*12)
}
labels4 = list(scenarios_combined.keys())
values4 = list(scenarios_combined.values())
bar_colors4 = [BRAND, GREEN, ACCENT, PURPLE, "#15803D"]
bars4 = ax4.bar(range(len(labels4)), values4, color=bar_colors4, edgecolor="white", width=0.65)
for bar, val in zip(bars4, values4):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1e5,
             inr(val), ha="center", fontsize=8.5, fontweight="bold")
ax4.set_xticks(range(len(labels4)))
ax4.set_xticklabels(labels4, fontsize=8)
ax4.set_title("Combined Scenario\nRevenue Waterfall (Annual)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Annual Revenue (₹)")
ax4.yaxis.set_major_formatter(FuncFormatter(lambda x,_: inr(x)))
ax4.set_facecolor(LIGHT)
uplift_total = values4[-1] - values4[0]
ax4.text(0.5, 0.93, f"Total uplift potential: {inr(uplift_total)}/year",
         transform=ax4.transAxes, ha="center", fontsize=9,
         fontweight="bold", color=GREEN,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor=GREEN, alpha=0.9))

# ── Sim 5: Engagement → LTV heatmap ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2,:])
lessons_buckets  = ["0–5", "6–15", "16–30", "31–60", "61+"]
months_buckets   = ["1–3 mo", "4–6 mo", "7–12 mo", "13–18 mo", "18+ mo"]
# Average LTV by engagement × tenure
paid2 = paid.copy()
paid2["lesson_bucket"] = pd.cut(paid2["lessons_total"], bins=[0,5,15,30,60,9999],
                                labels=lessons_buckets)
paid2["months_bucket"] = pd.cut(paid2["months_subscribed"], bins=[0,3,6,12,18,9999],
                                labels=months_buckets)
heatmap_data = paid2.groupby(["months_bucket","lesson_bucket"], observed=True)["ltv_inr"].mean().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(index=months_buckets, columns=lessons_buckets, fill_value=0)

im = ax5.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")
ax5.set_xticks(range(len(lessons_buckets))); ax5.set_xticklabels(lessons_buckets, fontsize=10)
ax5.set_yticks(range(len(months_buckets))); ax5.set_yticklabels(months_buckets, fontsize=10)
ax5.set_title("LTV Heatmap: Lessons Watched × Tenure — Where High-Value Users Cluster",
              fontsize=12, fontweight="bold")
ax5.set_xlabel("Total Lessons Watched", fontsize=11)
ax5.set_ylabel("Months Subscribed", fontsize=11)
plt.colorbar(im, ax=ax5, label="Avg LTV (₹)", shrink=0.8)
for i in range(len(months_buckets)):
    for j in range(len(lessons_buckets)):
        val = heatmap_data.values[i,j]
        if val > 0:
            ax5.text(j, i, inr(val), ha="center", va="center",
                     fontsize=9, fontweight="bold",
                     color="white" if val > heatmap_data.values.max()*0.6 else DARK)

plt.savefig("/home/claude/growth_engine/charts/layer4_revenue_simulator.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Layer 4 saved")
print(f"   Combined annual uplift potential: {inr(uplift_total)}")

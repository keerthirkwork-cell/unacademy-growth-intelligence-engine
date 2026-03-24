"""
Layer 1 — Free-to-Paid Conversion Funnel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

BRAND="#1A56A0"; ACCENT="#F97316"; GREEN="#16A34A"; RED="#DC2626"
GREY="#6B7280"; LIGHT="#EAF1FB"; DARK="#111827"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white"})

df = pd.read_csv("/home/claude/growth_engine/data/learner_data.csv")

FUNNEL_ORDER = ["Signed Up","First Lesson","3+ Lessons",
                "Mock Test Taken","Subscription Page Visited","Subscribed"]

df_f = df.copy()
df_f["funnel_stage"] = df_f.apply(
    lambda r: "Subscribed" if r["tier"] != "Free" else r["funnel_stage"], axis=1)

stage_counts = {}
for i, stage in enumerate(FUNNEL_ORDER):
    stage_counts[stage] = df_f[df_f["funnel_stage"].isin(FUNNEL_ORDER[i:])].shape[0]

counts = list(stage_counts.values())
pcts   = [c/counts[0]*100 for c in counts]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Layer 1 — Free-to-Paid Conversion Funnel", fontsize=16, fontweight="bold", color=DARK)

# Left: funnel
ax = axes[0]
bar_colors = ["#DBEAFE","#BFDBFE","#93C5FD","#60A5FA","#3B82F6",BRAND]
bars = ax.barh(FUNNEL_ORDER[::-1], counts[::-1], color=bar_colors[::-1],
               edgecolor="white", height=0.62)
for bar, cnt, pct in zip(bars, counts[::-1], pcts[::-1]):
    ax.text(bar.get_width()+100, bar.get_y()+bar.get_height()/2,
            f"{cnt:,}  ({pct:.1f}%)", va="center", fontsize=10, color=DARK, fontweight="bold")

# Drop-off arrows
for i in range(len(counts)-1):
    drop_pct = (counts[i]-counts[i+1])/counts[i]*100
    if drop_pct > 3:
        y_pos = len(FUNNEL_ORDER)-1-i - 0.5
        ax.annotate(f"↓ {drop_pct:.0f}% drop",
                    xy=(counts[i+1], y_pos),
                    fontsize=8.5, color=RED, style="italic",
                    xytext=(counts[i+1]+200, y_pos))

ax.set_xlim(0, counts[0]*1.38)
ax.set_xlabel("Number of Learners", fontsize=11)
ax.set_title("Where 67% of Learners Drop Before Paying", fontsize=12, fontweight="bold")
ax.set_facecolor(LIGHT)

# Key callout box
biggest_drop_idx = np.argmax([counts[i]-counts[i+1] for i in range(len(counts)-1)])
biggest_drop_stage = FUNNEL_ORDER[biggest_drop_idx]
biggest_drop_pct = (counts[biggest_drop_idx]-counts[biggest_drop_idx+1])/counts[biggest_drop_idx]*100
ax.text(0.38, 0.08,
        f"💡 Biggest leak:\n\"{biggest_drop_stage}\"\n({biggest_drop_pct:.0f}% drop)",
        transform=ax.transAxes, fontsize=9, color=BRAND, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=BRAND, alpha=0.95))

# Right: Revenue opportunity simulation
ax2 = axes[1]
free_count   = (df["tier"]=="Free").sum()
plus_price   = 1562
scenarios    = [1, 2, 3, 5, 8, 10]
monthly_revs = [int(free_count * s/100) * plus_price for s in scenarios]
annual_revs  = [r*12 for r in monthly_revs]

bar_cols = [LIGHT, "#BFDBFE", ACCENT, "#F97316", GREEN, "#15803D"]
bars2 = ax2.bar([f"+{s}%" for s in scenarios], annual_revs,
                color=bar_cols, edgecolor="white", width=0.62)
for bar, rev in zip(bars2, annual_revs):
    label = f"₹{rev/1e7:.1f}Cr" if rev>=1e7 else f"₹{rev/1e5:.1f}L"
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5e5,
             label, ha="center", fontsize=9, fontweight="bold", color=DARK)

# Highlight 3% bar
bars2[2].set_edgecolor(ACCENT); bars2[2].set_linewidth(2.5)
ax2.annotate("3% lift = ₹56.5Cr/year\n(most achievable target)",
             xy=(2, annual_revs[2]),
             xytext=(3.5, annual_revs[2]*0.75),
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
             fontsize=9, color=ACCENT, fontweight="bold")

ax2.set_title("Annual Revenue Unlocked by\nImproving Free-to-Paid Conversion", fontsize=12, fontweight="bold")
ax2.set_xlabel("Conversion Rate Improvement")
ax2.set_ylabel("Additional Annual Revenue (₹)")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"₹{x/1e7:.0f}Cr" if x>=1e7 else f"₹{x/1e5:.0f}L"))
ax2.set_facecolor(LIGHT)

plt.tight_layout()
plt.savefig("/home/claude/growth_engine/charts/layer1_funnel.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Layer 1 saved")
print(f"   Free users:           {free_count:,}")
print(f"   Biggest drop stage:   {biggest_drop_stage} ({biggest_drop_pct:.0f}%)")
print(f"   3% conv lift annual:  ₹{annual_revs[2]/1e7:.1f}Cr")

"""
Layer 5 — Action Engine
"What should Unacademy DO tomorrow morning?"
Turns predictions into a prioritised intervention playbook.
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
YELLOW="#F59E0B"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white"})

df   = pd.read_csv("/home/claude/growth_engine/data/learner_data.csv")
paid = df[df["tier"]!="Free"].copy()
free = df[df["tier"]=="Free"].copy()

# ── ACTION RULES ──────────────────────────────────────────────────────────────
# Each rule = a real intervention Unacademy could run tomorrow

def assign_action(row):
    # Paid users
    if row["tier"] != "Free":
        prob = row.get("churn_probability", 0)
        if prob >= 0.70:
            return ("🔴 CRITICAL", "Send personalised 20% discount NOW",
                    "Renewal at serious risk — price incentive + educator message",
                    "churn_prevention", row["monthly_spend_inr"])
        elif prob >= 0.50:
            return ("🟠 HIGH RISK", "Send re-engagement push notification",
                    "Drop in session activity — remind them of progress made",
                    "churn_prevention", row["monthly_spend_inr"] * 0.7)
        elif prob >= 0.30:
            return ("🟡 WATCH", "Schedule check-in from educator",
                    "Early warning signals — proactive educator outreach",
                    "churn_prevention", row["monthly_spend_inr"] * 0.3)
        elif row["tier"] == "Plus" and row.get("mock_tests_total",0) >= 5 and prob < 0.20:
            return ("🚀 UPGRADE", "Show Iconic upgrade offer with social proof",
                    "High engagement Plus user — ready for premium",
                    "upsell", (2396-1562))
        else:
            return ("✅ HEALTHY", "Nurture with content recommendations",
                    "Engaged, low risk — keep them happy",
                    "nurture", 0)
    # Free users
    else:
        lessons = row.get("lessons_total", 0)
        mocks   = row.get("mock_tests_total", 0)
        days_inactive = row.get("days_since_last_login", 99)
        funnel  = row.get("funnel_stage", "Signed Up")

        if mocks >= 1 and funnel in ["Mock Test Taken","Subscription Page Visited"]:
            return ("⚡ HOT LEAD", "Show Plus offer immediately after mock test result",
                    "Mock test completion = highest intent signal. Strike now.",
                    "conversion", 1562)
        elif lessons >= 10 and days_inactive <= 7:
            return ("🎯 HIGH INTENT", "Trigger 7-day free Plus trial",
                    "Active learner with high lesson count — trial removes friction",
                    "conversion", 1562)
        elif lessons >= 5 and days_inactive <= 14:
            return ("📊 NURTURE", "Send personalised performance gap email",
                    "Show them what paid learners achieve vs free learners",
                    "conversion", 781)  # expected 50% conv
        elif days_inactive > 30:
            return ("😴 DORMANT", "Win-back email with exam date urgency",
                    "Long inactive — exam deadline urgency is the trigger",
                    "reactivation", 0)
        else:
            return ("👀 OBSERVING", "Continue onboarding sequence",
                    "Still in early journey — educate before selling",
                    "nurture", 0)

# Apply action engine
result = df.apply(assign_action, axis=1)
df["action_priority"]  = result.apply(lambda x: x[0])
df["action_label"]     = result.apply(lambda x: x[1])
df["action_reason"]    = result.apply(lambda x: x[2])
df["action_type"]      = result.apply(lambda x: x[3])
df["action_rev_upside"]= result.apply(lambda x: x[4])

df.to_csv("/home/claude/growth_engine/data/learner_data.csv", index=False)
print("✅ Action engine applied")

# ── SUMMARY TABLE ────────────────────────────────────────────────────────────
summary = df.groupby("action_priority").agg(
    users=("learner_id","count"),
    total_rev_upside=("action_rev_upside","sum"),
).reset_index().sort_values("total_rev_upside", ascending=False)
print("\n── Action Playbook Summary ──")
print(summary.to_string(index=False))

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("Layer 5 — Action Engine: What Should Unacademy Do Tomorrow?",
             fontsize=16, fontweight="bold", color=DARK)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

PRIORITY_COLORS = {
    "🔴 CRITICAL":   RED,
    "🟠 HIGH RISK":  ACCENT,
    "🟡 WATCH":      YELLOW,
    "🚀 UPGRADE":    PURPLE,
    "✅ HEALTHY":    GREEN,
    "⚡ HOT LEAD":   "#DC2626",
    "🎯 HIGH INTENT":BRAND,
    "📊 NURTURE":    "#60A5FA",
    "😴 DORMANT":    GREY,
    "👀 OBSERVING":  "#9CA3AF",
}

def inr(x):
    if x>=1e7: return f"₹{x/1e7:.1f}Cr"
    if x>=1e5: return f"₹{x/1e5:.1f}L"
    if x>=1e3: return f"₹{x/1e3:.0f}K"
    return f"₹{x:.0f}"

# 1. User count by action
ax1 = fig.add_subplot(gs[0,:2])
act_counts = df["action_priority"].value_counts()
colors1 = [PRIORITY_COLORS.get(p, GREY) for p in act_counts.index]
bars1 = ax1.barh(act_counts.index[::-1], act_counts.values[::-1],
                 color=colors1[::-1], edgecolor="white", height=0.65)
for bar, val in zip(bars1, act_counts.values[::-1]):
    ax1.text(bar.get_width()+30, bar.get_y()+bar.get_height()/2,
             f"{val:,} users", va="center", fontsize=10, fontweight="bold")
ax1.set_title("Learners by Recommended Action (Today's Intervention Queue)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Number of Learners")
ax1.set_xlim(0, act_counts.max()*1.25)
ax1.set_facecolor(LIGHT)

# 2. Revenue upside by action type
ax2 = fig.add_subplot(gs[0,2])
rev_by_type = df.groupby("action_type")["action_rev_upside"].sum().sort_values(ascending=False)
type_colors = {"conversion":GREEN,"churn_prevention":RED,"upsell":PURPLE,
               "reactivation":ACCENT,"nurture":GREY}
colors2 = [type_colors.get(t, GREY) for t in rev_by_type.index]
wedges,texts,auto = ax2.pie(rev_by_type.values, labels=rev_by_type.index,
                             colors=colors2, autopct="%1.1f%%", startangle=140,
                             wedgeprops={"edgecolor":"white","linewidth":2})
for at in auto: at.set_fontsize(8); at.set_fontweight("bold")
ax2.set_title("Revenue Upside\nby Action Type", fontsize=11, fontweight="bold")

# 3. Action playbook cards (the visual showpiece)
ax3 = fig.add_subplot(gs[1,:])
ax3.set_xlim(0, 20); ax3.set_ylim(0, 10); ax3.axis("off")
ax3.set_title("Today's Top 6 Intervention Playbook Cards",
              fontsize=12, fontweight="bold", color=DARK, pad=10)

top_actions = [
    ("🔴 CRITICAL\nChurn Risk",   f"{df[df['action_priority']=='🔴 CRITICAL'].shape[0]:,} users",
     "Send 20% personalised\ndiscount + educator msg",
     f"Rev at risk: {inr(df[df['action_priority']=='🔴 CRITICAL']['monthly_spend_inr'].sum())}/mo", RED),
    ("🟠 HIGH RISK\nChurn Watch",  f"{df[df['action_priority']=='🟠 HIGH RISK'].shape[0]:,} users",
     "Re-engagement push\nnotification",
     f"Rev at risk: {inr(df[df['action_priority']=='🟠 HIGH RISK']['monthly_spend_inr'].sum())}/mo", ACCENT),
    ("⚡ HOT LEAD\nPost Mock Test", f"{df[df['action_priority']=='⚡ HOT LEAD'].shape[0]:,} users",
     "Show Plus offer\nimmediately after result",
     f"Conv upside: {inr(df[df['action_priority']=='⚡ HOT LEAD']['action_rev_upside'].sum())}/mo", "#DC2626"),
    ("🎯 HIGH INTENT\nFree Learner", f"{df[df['action_priority']=='🎯 HIGH INTENT'].shape[0]:,} users",
     "Trigger 7-day free\nPlus trial",
     f"Conv upside: {inr(df[df['action_priority']=='🎯 HIGH INTENT']['action_rev_upside'].sum())}/mo", BRAND),
    ("🚀 UPGRADE\nPlus → Iconic",   f"{df[df['action_priority']=='🚀 UPGRADE'].shape[0]:,} users",
     "Social proof upgrade\noffer + feature preview",
     f"Upsell upside: {inr(df[df['action_priority']=='🚀 UPGRADE']['action_rev_upside'].sum())}/mo", PURPLE),
    ("😴 DORMANT\nWin-Back",        f"{df[df['action_priority']=='😴 DORMANT'].shape[0]:,} users",
     "Exam deadline urgency\nwin-back email",
     f"Reactivation potential", GREY),
]

for idx, (title, count, action, impact, color) in enumerate(top_actions):
    x = (idx % 3) * 6.8 + 0.5
    y = 5.5 if idx < 3 else 0.3
    # Card background
    ax3.add_patch(mpatches.FancyBboxPatch((x, y), 6.0, 4.2,
        boxstyle="round,pad=0.2", facecolor="white",
        edgecolor=color, linewidth=2.5))
    # Colour header strip
    ax3.add_patch(mpatches.FancyBboxPatch((x, y+3.1), 6.0, 1.1,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="none"))
    ax3.text(x+3.0, y+3.65, title, ha="center", va="center",
             fontsize=9, fontweight="bold", color="white")
    ax3.text(x+3.0, y+2.6, count, ha="center", va="center",
             fontsize=12, fontweight="bold", color=DARK)
    ax3.text(x+3.0, y+1.8, "ACTION:", ha="center", va="center",
             fontsize=7.5, color=GREY, fontweight="bold")
    ax3.text(x+3.0, y+1.2, action, ha="center", va="center",
             fontsize=9, color=DARK)
    ax3.text(x+3.0, y+0.4, impact, ha="center", va="center",
             fontsize=8.5, color=color, fontweight="bold")

# 4. Daily action queue — cumulative revenue over time
ax4 = fig.add_subplot(gs[2,:])
months = np.arange(1,13)
# Base revenue
base_mo = df[df["tier"]!="Free"]["monthly_spend_inr"].sum()

# Scenario: with action engine vs without
critical_saved_pct   = 0.35   # 35% of critical users saved
hotlead_conv_pct     = 0.22   # 22% of hot leads convert
highintent_conv_pct  = 0.12
upgrade_pct          = 0.08

critical_rev  = df[df["action_priority"]=="🔴 CRITICAL"]["monthly_spend_inr"].sum() * critical_saved_pct
hotlead_rev   = df[df["action_priority"]=="⚡ HOT LEAD"]["action_rev_upside"].sum() * hotlead_conv_pct
highint_rev   = df[df["action_priority"]=="🎯 HIGH INTENT"]["action_rev_upside"].sum() * highintent_conv_pct
upgrade_rev   = df[df["action_priority"]=="🚀 UPGRADE"]["action_rev_upside"].sum() * upgrade_pct
monthly_gain  = critical_rev + hotlead_rev + highint_rev + upgrade_rev

baseline_cumulative  = [base_mo * m for m in months]
# With action engine: compounds as more users converted/retained over time
with_engine = [base_mo * m + monthly_gain * m * (1 + m*0.02) for m in months]

ax4.fill_between(months, baseline_cumulative, with_engine,
                 alpha=0.25, color=GREEN, label="Additional revenue from actions")
ax4.plot(months, baseline_cumulative, color=BRAND, lw=2.5, label="Baseline (no action)", linestyle="--")
ax4.plot(months, with_engine, color=GREEN, lw=2.5, label="With Action Engine", marker="o", markersize=6)

gap_12mo = with_engine[-1] - baseline_cumulative[-1]
ax4.annotate(f"12-month gap:\n{inr(gap_12mo)} additional revenue",
             xy=(12, with_engine[-1]),
             xytext=(9.5, with_engine[-1]*0.75),
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
             fontsize=10, color=GREEN, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor=GREEN))

ax4.set_title("12-Month Revenue Projection: Baseline vs With Action Engine",
              fontsize=12, fontweight="bold")
ax4.set_xlabel("Month", fontsize=11)
ax4.set_ylabel("Cumulative Revenue (₹)", fontsize=11)
ax4.set_xticks(months); ax4.set_xticklabels([f"M{m}" for m in months])
ax4.yaxis.set_major_formatter(FuncFormatter(lambda x,_: inr(x)))
ax4.legend(fontsize=10)
ax4.set_facecolor(LIGHT)

plt.savefig("/home/claude/growth_engine/charts/layer5_action_engine.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Layer 5 saved")
print(f"\n── Monthly Revenue Gain from Action Engine ──")
print(f"Critical users saved:    {inr(critical_rev)}/mo")
print(f"Hot leads converted:     {inr(hotlead_rev)}/mo")
print(f"High intent converted:   {inr(highint_rev)}/mo")
print(f"Plus → Iconic upgrades:  {inr(upgrade_rev)}/mo")
print(f"Total monthly gain:      {inr(monthly_gain)}/mo")
print(f"12-month gap:            {inr(gap_12mo)}")

"""
Unacademy Growth Intelligence Engine
Dataset Generator — 12,000 learner records
Designed to support: KMeans clustering, churn ML model, revenue simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

np.random.seed(42)
random.seed(42)

N = 12000

EXAM_CONFIG = {
    "UPSC":    {"weight":0.18,"prep_months":(18,36),"base_conv":0.22,"base_churn":0.08,"avg_session_min":45},
    "JEE":     {"weight":0.22,"prep_months":(12,24),"base_conv":0.18,"base_churn":0.19,"avg_session_min":38},
    "NEET":    {"weight":0.20,"prep_months":(12,24),"base_conv":0.17,"base_churn":0.21,"avg_session_min":36},
    "CAT":     {"weight":0.12,"prep_months":(6,12), "base_conv":0.24,"base_churn":0.14,"avg_session_min":32},
    "Banking": {"weight":0.15,"prep_months":(4,10), "base_conv":0.15,"base_churn":0.26,"avg_session_min":28},
    "SSC":     {"weight":0.13,"prep_months":(4,10), "base_conv":0.13,"base_churn":0.28,"avg_session_min":25},
}
EXAMS   = list(EXAM_CONFIG.keys())
EXAM_W  = [EXAM_CONFIG[e]["weight"] for e in EXAMS]
STATES  = ["UP","Bihar","Maharashtra","Karnataka","Rajasthan","MP","TN","AP","WB","Gujarat"]
STATE_W = [0.18,0.14,0.12,0.09,0.10,0.08,0.07,0.07,0.08,0.07]
TIER_PRICE = {"Free":0,"Plus":1562,"Iconic":2396}

FUNNEL_STAGES = ["Signed Up","First Lesson","3+ Lessons","Mock Test Taken","Subscription Page Visited","Subscribed"]

def rand_date(lo=30, hi=730):
    return datetime.today() - timedelta(days=random.randint(lo, hi))

records = []
for i in range(N):
    exam    = np.random.choice(EXAMS, p=EXAM_W)
    cfg     = EXAM_CONFIG[exam]
    state   = np.random.choice(STATES, p=STATE_W)
    age     = int(np.clip(np.random.normal(21,3), 17, 32))
    signup  = rand_date(30, 730)
    days_since = (datetime.today() - signup).days

    # ── BEHAVIOURAL FEATURES (core ML inputs) ────────────────────────
    eng = {"UPSC":0.72,"JEE":0.66,"NEET":0.63,"CAT":0.56,"Banking":0.46,"SSC":0.41}[exam]

    lessons_total       = int(np.random.negative_binomial(5, max(0.01, 1 - eng*0.08)) + 1)
    mock_tests          = int(np.random.poisson(eng * 9))
    doubt_sessions      = int(np.random.poisson(eng * 5))
    live_classes        = int(np.random.poisson(eng * 12))
    days_active         = int(min(np.random.poisson(eng * 42), days_since))
    avg_session_min     = round(np.random.normal(cfg["avg_session_min"], 8), 1)
    avg_session_min     = max(5, avg_session_min)
    total_time_hrs      = round((lessons_total * avg_session_min) / 60, 1)
    days_since_last     = int(np.random.exponential(14))   # recency signal
    streak_days         = int(np.random.poisson(eng * 7))
    notes_saved         = int(np.random.poisson(eng * 20))
    videos_rewatched    = int(np.random.poisson(eng * 3))
    login_frequency_wk  = round(np.random.poisson(eng * 5), 1)  # logins/week

    # ── TIER ASSIGNMENT ──────────────────────────────────────────────
    conv_score = cfg["base_conv"]
    conv_score += min(lessons_total, 60) * 0.0025
    conv_score += mock_tests   * 0.016
    conv_score += doubt_sessions * 0.013
    conv_score += days_active  * 0.001
    conv_score += (avg_session_min > 30) * 0.04
    conv_score  = min(conv_score, 0.78)

    r = random.random()
    if   r < conv_score * 0.62: tier = "Plus"
    elif r < conv_score:         tier = "Iconic"
    else:                        tier = "Free"

    # ── SUBSCRIPTION FIELDS ──────────────────────────────────────────
    months_sub = sub_start = renewal = None
    spend = 0
    lessons_30d = mock_30d = spend_to_date = 0

    if tier != "Free":
        months_sub       = random.randint(1, 20)
        spend            = TIER_PRICE[tier]
        sub_start        = signup + timedelta(days=random.randint(3,60))
        renewal          = sub_start + timedelta(days=30*months_sub)
        lessons_30d      = int(np.random.poisson(eng * 14))
        mock_30d         = int(np.random.poisson(eng * 3))
        spend_to_date    = spend * months_sub

    days_to_renewal = int((renewal - datetime.today()).days) if renewal else None

    # ── CHURN LABEL ──────────────────────────────────────────────────
    churned = 0
    if tier != "Free":
        churn_p  = cfg["base_churn"]
        churn_p -= lessons_30d   * 0.009
        churn_p -= mock_30d      * 0.018
        churn_p -= (streak_days > 7) * 0.05
        churn_p += (months_sub > 14) * 0.07
        churn_p += (days_since_last > 21) * 0.12
        churn_p  = max(0.03, min(churn_p, 0.85))
        churned  = int(random.random() < churn_p)

    # ── CHURN RISK SCORE (0–100) ──────────────────────────────────────
    if tier == "Free":
        churn_risk = 0
    else:
        risk  = cfg["base_churn"] * 100
        risk -= lessons_30d * 1.1
        risk -= mock_30d    * 1.8
        risk -= (streak_days > 7) * 8
        risk += (days_since_last > 21) * 18
        risk += (months_sub > 14) * 10
        risk += churned * 22
        churn_risk = int(np.clip(risk, 5, 95))

    # ── FUNNEL STAGE (free users only) ───────────────────────────────
    if tier != "Free":
        funnel = "Subscribed"
    else:
        drops = [0.0, 0.31, 0.27, 0.21, 0.16, 0.0]
        funnel = "Signed Up"
        for stage, dp in zip(FUNNEL_STAGES, drops):
            if random.random() < dp:
                funnel = stage
                break
        else:
            funnel = "Signed Up"

    # ── LTV ───────────────────────────────────────────────────────────
    prep_mo = random.randint(*cfg["prep_months"])
    ltv = round(spend * min(months_sub or 0, prep_mo) * (1 - churned*0.35), 0)

    records.append({
        # Identity
        "learner_id":           f"UA{200000+i}",
        "signup_date":          signup.date(),
        "state":                state,
        "age":                  age,
        "exam_category":        exam,
        # Tier / subscription
        "tier":                 tier,
        "monthly_spend_inr":    spend,
        "months_subscribed":    months_sub or 0,
        "spend_to_date_inr":    spend_to_date,
        "days_to_renewal":      days_to_renewal,
        # Behavioural (ML features)
        "lessons_total":        lessons_total,
        "lessons_last_30d":     lessons_30d,
        "mock_tests_total":     mock_tests,
        "mock_tests_last_30d":  mock_30d,
        "doubt_sessions":       doubt_sessions,
        "live_classes":         live_classes,
        "days_active":          days_active,
        "days_since_last_login":days_since_last,
        "avg_session_min":      avg_session_min,
        "total_time_hrs":       total_time_hrs,
        "streak_days":          streak_days,
        "login_freq_per_week":  login_frequency_wk,
        "notes_saved":          notes_saved,
        "videos_rewatched":     videos_rewatched,
        # Outcomes
        "funnel_stage":         funnel,
        "churned":              churned,
        "churn_risk_score":     churn_risk,
        "ltv_inr":              ltv,
    })

df = pd.DataFrame(records)
df.to_csv("/home/claude/growth_engine/data/learner_data.csv", index=False)

print(f"✅  {len(df):,} records generated")
print(f"\nTier split:\n{df['tier'].value_counts().to_string()}")
print(f"\nChurn rate (paid): {df[df['tier']!='Free']['churned'].mean():.2%}")
print(f"Paid learners:     {(df['tier']!='Free').sum():,}")
print(f"Free learners:     {(df['tier']=='Free').sum():,}")

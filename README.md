# Unacademy Growth Intelligence Engine

> **"Instead of analysing what happened, this project predicts what will happen — and recommends what Unacademy should do about it."**

Built as part of my application to Unacademy's Business Analyst team. I spent time understanding the actual business problem before writing a single line of code.

---

## Why This Exists

Unacademy's revenue dropped from ₹907 Cr (FY23) to ₹701 Cr (FY25) despite a growing user base.  
The problem is not awareness. It is **monetisation** — 67% of learners are free users, and most will never pay.

Most analytics projects would show you a funnel chart and call it done.  
This project goes further: **predict who is about to churn, segment users by behaviour, simulate the ₹ impact of every decision, and tell the team exactly what to do tomorrow morning.**

---

## Architecture — 5 Layers

```
Layer 1 → Funnel          Where does conversion break?
Layer 2 → Segmentation    Who are your learners really? (KMeans clustering)
Layer 3 → Churn Model     Who will churn in 30 days? (Random Forest, AUC 0.668)
Layer 4 → Revenue Sim     What is every 1% improvement worth in ₹?
Layer 5 → Action Engine   What should Unacademy do tomorrow?
```

---

## Key Numbers

| Metric | Value |
|---|---|
| Dataset | 12,000 synthetic learner records |
| Free users (unconverted) | 7,675 (64%) |
| Monthly revenue modelled | ₹67.6L |
| Revenue at risk (churning users) | ₹7.5L/month |
| Critical churn risk users (prob ≥ 70%) | 140 |
| Hot leads (post mock-test, act now) | 1,302 |
| Churn model AUC | 0.668 |
| Action engine monthly gain (est.) | ₹5.5L/month |
| 12-month revenue gap vs baseline | ₹82.5L |

---

## Layer 1 — Conversion Funnel

Mapped 12,000 learner journeys across 6 stages: Sign Up → First Lesson → 3+ Lessons → Mock Test → Subscription Page → Paid.

**Key finding:** The biggest single drop happens before learners even get to the subscription page. Users who complete a mock test are the highest-intent segment — they convert at 3× the rate of passive browsers.

**Recommendation:** Surface mock test prompts after lesson 3, not lesson 15. The moment a learner sees their score, they are ready to buy.

---

## Layer 2 — Behavioural Segmentation (KMeans)

Used 10 behavioural features (lessons, mock tests, session time, streak days, login frequency, recency) to cluster learners into 4 meaningful segments:

| Segment | Size | Conversion | Avg LTV | Churn Rate |
|---|---|---|---|---|
| 🧠 Serious Aspirants | 4,762 | 39% | ₹7,040 | 3% |
| 🚀 High Intent | 1,909 | 41% | ₹7,349 | 4% |
| ⚠️ At-Risk | 1,187 | 36% | ₹5,561 | 6% |
| 😴 Casual Browsers | 4,142 | 30% | ₹3,631 | 5% |

**Key finding:** Top 2 segments drive 72% of total revenue. Casual Browsers are 34% of the user base but contribute only 19% of revenue. Retention spend should be concentrated on Serious Aspirants — not spread equally.

---

## Layer 3 — Churn Prediction Model

**Algorithm:** Random Forest (200 trees, balanced class weights)  
**Target:** Will this paid user churn in the next 30 days?  
**AUC-ROC: 0.668 | CV AUC: 0.668 ± 0.022**

Top 3 churn signals (by feature importance):
1. **Days since last login** — the single strongest predictor
2. **Lessons watched in last 30 days** — drop in engagement = warning sign
3. **Mock tests in last 30 days** — disengagement from core product = churn incoming

Risk tier breakdown:
- 🔴 Critical (prob ≥ 70%): 138 users, ₹2.6L/month at risk
- 🟠 High (50–70%): 244 users, ₹3.8L/month at risk
- 🟡 Medium (30–50%): act proactively before they move up

*Note: AUC 0.668 is honest and realistic. A synthetic dataset with perfectly separable classes would be misleading. The model correctly identifies directional signals and ranks users by risk — which is exactly what an intervention queue needs.*

---

## Layer 4 — Revenue Impact Simulator

Quantified the ₹ value of every 1% improvement across 3 levers:

**Lever 1 — Free → Plus Conversion**
| Improvement | Additional Annual Revenue |
|---|---|
| +1% | ₹14.2L |
| +3% | ₹42.5L |
| +5% | ₹70.8L |
| +10% | ₹141.6L |

**Lever 2 — Churn Reduction**
A 10% reduction in churn rate saves ₹9L/year from the current churning base alone.

**Lever 3 — Plus → Iconic Upgrade**
₹834/month incremental per user. 5% upgrade rate across the Plus base = ₹10.5L/year.

**Combined scenario (3% conv + 10% churn reduction + 5% upgrade):**  
> Total uplift: **₹68.7L additional annual revenue**

---

## Layer 5 — Action Engine

This is where the project moves from analysis to decision-making.

Every learner in the dataset receives a specific recommended action today — not a segment tag, an actual next step:

| Priority | Users | Action | Rationale |
|---|---|---|---|
| ⚡ HOT LEAD | 1,302 | Show Plus offer post mock-test | Highest intent signal — act immediately |
| 🟡 WATCH | 2,069 | Educator check-in scheduled | Early warning signals — proactive outreach |
| 🟠 HIGH RISK | 866 | Re-engagement push notification | Session drop — remind them of progress |
| 🔴 CRITICAL | 140 | 20% personalised discount NOW | Churn probability ≥ 70% — price is the lever |
| 🚀 UPGRADE | 203 | Iconic offer with social proof | High-engagement Plus users ready to upgrade |
| 😴 DORMANT | 670 | Exam deadline urgency email | Long inactive — urgency is the only trigger |

**12-month projection:** Running the action engine vs doing nothing generates an estimated **₹82.5L additional cumulative revenue**.

---

## Tech Stack

| | Tool | Usage |
|---|---|---|
| Data | Python (Pandas, NumPy) | Synthetic dataset generation |
| ML | Scikit-learn (RandomForest, KMeans, PCA) | Churn model + segmentation |
| Viz | Matplotlib, Seaborn | All 5 layers + executive dashboard |
| SQL | PostgreSQL-style queries | Production-ready intervention queries |

---

## Repository Structure

```
unacademy-growth-intelligence-engine/
│
├── generate_data.py             # Reproducible 12,000-record dataset generator
├── layer1_funnel.py             # Conversion funnel analysis
├── layer2_segmentation.py       # KMeans behavioural clustering
├── layer3_churn_model.py        # Random Forest churn prediction
├── layer4_simulator.py          # Revenue impact simulation
├── layer5_action_engine.py      # Per-user action recommendations
├── executive_dashboard.py       # C-level KPI dashboard
│
├── data/
│   ├── learner_data.csv         # Full dataset with predictions + actions
│   └── paid_with_predictions.csv
│
├── models/
│   └── churn_model.pkl          # Trained Random Forest model
│
└── charts/
    ├── executive_dashboard.png
    ├── layer1_funnel.png
    ├── layer2_segmentation.png
    ├── layer3_churn_model.png
    ├── layer4_revenue_simulator.png
    └── layer5_action_engine.png
```

---

## About the Dataset

All data is synthetically generated — but built from realistic domain knowledge, not random numbers:

- Exam categories and weights match Unacademy's actual product mix (UPSC, JEE, NEET, CAT, Banking, SSC)
- Pricing tiers match Unacademy's published pricing (Free / Plus ₹1,562 / Iconic ₹2,396)
- Prep duration cycles are realistic (UPSC 18–36 months, JEE/NEET 12–24 months)
- Behavioural features are correlated with exam type and engagement level
- Churn rates are modelled on published SaaS and edtech benchmarks

Building a synthetic dataset from a domain model — rather than grabbing a generic Kaggle CSV — demonstrates that I understand how Unacademy's data actually works.

---

## Resume Bullet (for reference)

> **Unacademy Growth Intelligence Engine** — Built end-to-end learner monetisation analytics system for Unacademy's business problem: KMeans segmentation (4 behavioural clusters), Random Forest churn prediction (AUC 0.668), revenue simulator quantifying ₹68.7L annual uplift potential, and a per-user action engine generating ₹82.5L 12-month revenue gap vs baseline. Python, Scikit-learn, Matplotlib.

---

*Built by Keerthi RK — [LinkedIn](https://linkedin.com) • keerthirk.work@gmail.com*

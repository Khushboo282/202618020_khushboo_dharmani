# Applied Statistical Modeling & Interactive Web Dashboard
**ID** 202618020
**NAME** KHUSHBOO DHARMANI
**Lab:** Lab-4
**Dataset:** Medical Insurance Costs

**🔗 Live Demo:** _[https://202618020khushboodharmani-fusq8rhahqnyp5lqejk2pa.streamlit.app/]_

---

## 1. Project Overview

This project performs an end-to-end statistical analysis on the Medical Insurance Costs dataset,
covering exploratory data analysis, hypothesis testing, multiple linear regression with
Gauss-Markov diagnostics, and an interactive Streamlit dashboard for exploring the data,
running hypothesis tests dynamically, and generating live predictions.

---

## 2. Dataset Summary

| Feature   | Type        | Description                                      |
|-----------|-------------|---------------------------------------------------|
| age       | Continuous  | Age of primary beneficiary                        |
| sex       | Categorical | Gender (male / female)                            |
| bmi       | Continuous  | Body Mass Index                                   |
| children  | Discrete    | Number of dependents covered by insurance         |
| smoker    | Categorical | Smoking status (yes / no)                         |
| region    | Categorical | Residential region (northeast, northwest, southeast, southwest) |
| charges   | Continuous  | Individual medical costs billed by insurance (target variable) |

**Source:** Publicly available Medical Insurance Costs dataset (equivalent open-access
version may be substituted; see `data/` folder or fetch URL below).

**Key characteristics:**
- `charges` is right-skewed, with a long tail driven by high-cost claims (largely smokers).
- `smoker` status shows a strong, visually separable effect on `charges`.
- `bmi` × `smoker` shows an interaction effect: high BMI amplifies the cost impact of smoking.

---

## 3. Repository Structure

```
.
├── app.py                # Streamlit dashboard 
├── requirements.txt      # Python dependencies
├── data/
│   └── insurance.csv     # Dataset
├── notebooks/
│   └── analysis.ipynb    # EDA, hypothesis tests, OLS model, diagnostics
└── README.md
```

## 4. How to Run

###  Local IDE 

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```


---

## 5. Summary of Statistical Findings

### 5.1 Descriptive Statistics
- `charges`: mean = **$13,270**, median = **$9,382** — mean is 41% higher than median,
  indicating strong **right skew** (skewness = **1.51**, excess kurtosis = **1.60**,
  i.e. leptokurtic with heavier-than-normal tails driven by high-cost outliers).
- `age` and `bmi` are close to symmetric (skewness 0.06 and 0.28 respectively);
  `age` is notably platykurtic (kurtosis −1.25, flatter than normal).
- No missing values across all 1,338 records; no numeric predictor pair is highly
  correlated (max |r| = 0.30, age–charges), so no multicollinearity concern going in.

### 5.2 Hypothesis Test 1 — Smokers vs. Non-Smokers (Charges)
- H0: No difference in charges between smokers and non-smokers.
- H1: A significant difference exists.
- Normality (Shapiro-Wilk): **failed for both groups** (smoker p = 3.6×10⁻⁹,
  non-smoker p = 1.4×10⁻²⁸)
- Equal variance (Levene's): **failed** (p = 1.6×10⁻⁶⁶)
- Test used: **Mann-Whitney U** (normality violated → non-parametric test required)
- Result: U = 284,133, **p = 5.3×10⁻¹³⁰**
- **Conclusion (α = 0.05): Reject H0.** Median charges: smokers $34,456 vs.
  non-smokers $7,345 — smokers pay roughly **4.7× more**.

### 5.3 Hypothesis Test 2 — Charges Across Regions
- H0: Mean charges are equal across all four regions.
- H1: At least one region differs.
- Normality (Shapiro-Wilk): **failed for all 4 regions** (all p < 10⁻¹⁷)
- Equal variance (Levene's): **failed** (p = 0.00086)
- Test used: **Kruskal-Wallis** (the assumption-appropriate test, since both ANOVA
  preconditions failed)
- Result: One-Way ANOVA (reference only, assumptions violated): F = 2.97, p = 0.031
  (would suggest "reject"). Kruskal-Wallis (appropriate test): H = 4.73, **p = 0.192**
- **Conclusion (α = 0.05): Fail to Reject H0.** No statistically significant
  regional effect on charges once the correct assumption-respecting test is used.
  **Note:** this is a case where ANOVA and Kruskal-Wallis disagree — ANOVA's result
  is not trustworthy here since its own normality/variance assumptions failed, and
  the apparent regional difference is more likely driven by uneven outlier density
  (southeast has the heaviest tail) than a genuine shift in the bulk of the
  distribution.

### 5.4 OLS Regression Model
- Model specification: `charges ~ age + C(sex) + children + C(region) + C(smoker) * bmi`
- R² = **0.841**, Adjusted R² = **0.840**
- Key coefficients (all in $ per unit unless noted):
  - `age`: +263.62 (p < 0.001) — older applicants charged more
  - `children`: +516.40 (p < 0.001)
  - `smoker[yes]`: −20,420 base offset, but see interaction below — not meaningful alone
  - `bmi`: +23.53 (p = 0.358, **not significant**) — for non-smokers, BMI barely matters
  - `smoker[yes] × bmi` (interaction): **+1,443.10 (p < 0.001)** — for smokers,
    each additional BMI point adds ~$1,443, on top of the main effects. This
    interaction is the single most important structural finding in the model:
    BMI is a strong cost driver *only when combined with smoking*.
  - `region[southeast]`: −1,210 (p = 0.002), `region[southwest]`: −1,231 (p = 0.001),
    both significantly lower than the reference region (northeast)
  - `sex[male]`: −500 (p = 0.061, borderline, not significant at α = 0.05)

### 5.5 Gauss-Markov Diagnostics
- **Linearity/Homoscedasticity** (Residuals vs. Fitted): **Violated.** Clear
  diagonal banding/funnel pattern rather than random scatter around zero —
  residual variance is not constant across fitted values.
- **Normality of residuals** (Q-Q plot, Jarque-Bera): **Violated.** Jarque-Bera =
  4465.2, p ≈ 0.00; residual skewness = 2.53, kurtosis = 10.38 — heavy right tail,
  consistent with `charges` being right-skewed at the source.
- **Multicollinearity** (VIF, continuous predictors): **No concern.** age = 1.01,
  bmi = 1.01, children = 1.00 — all essentially at the VIF = 1 floor.
- **Remediation attempted:** re-fit the same specification on `log(charges)`.
  Result: residual skewness improved from 2.53 → 1.85 and kurtosis from 10.38 →
  7.99, with visibly tighter Residuals-vs-Fitted and Q-Q behavior — a real
  improvement, though Jarque-Bera still rejects normality (real-world cost data
  rarely fits any linear model's assumptions exactly). The log model is the more
  diagnostically defensible model for inference; the raw-`charges` model is kept
  for the dashboard's prediction tab since dollar-scale output is more directly
  interpretable for end users.

### 5.6 Overall Synthesis
Smoking status is overwhelmingly the dominant driver of medical charges, and its
effect compounds strongly with BMI — the smoker×BMI interaction term alone explains
much of why a simple additive model would have understated cost risk for
high-BMI smokers. Region has no statistically defensible effect once assumption
violations are properly accounted for (a result that only emerged by running and
respecting the normality/variance checks rather than defaulting to ANOVA). The
regression model explains 84% of variance in charges but its raw form violates
normality and homoscedasticity assumptions due to the right-skewed nature of
medical cost data; a log-transformed specification meaningfully — though not
fully — resolves this.

---

## 6. Dashboard Structure

| Tab | Contents |
|-----|----------|
| **1. Data Exploration** | Sidebar filters (age/BMI sliders, category multi-select), reactive Plotly/Seaborn plots, summary statistics |
| **2. Hypothesis Testing Lab** | Dropdown-driven test selection, automatic computation of test statistics/p-values, plain-language conclusions |
| **3. Live Prediction & Diagnostics** | Interactive inputs for model prediction with 95% CI/PI, residual diagnostic plots |

---

## 7. Deployment

**Platform:** Streamlit Community Cloud (share.streamlit.io)


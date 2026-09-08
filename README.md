# Applied Statistical Modeling & Interactive Web Dashboard
202618020
KHUSHBOO DHARMANI

**Dataset:** Medical Insurance Costs

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


**Key characteristics:**
- `charges` is right-skewed, with a long tail driven by high-cost claims (largely smokers).
- `smoker` status shows a strong, visually separable effect on `charges`.
- `bmi` × `smoker` shows an interaction effect: high BMI amplifies the cost impact of smoking.

---

## 3. Repository Structure

```
.
├── app.py                
├── requirements.txt      
├── data/
│   └── insurance.csv     
├── notebooks/
│   └── analysis.ipynb    
└── README.md

```
## 4. Summary of Statistical Findings

> _To be completed after analysis is finalized. Suggested structure below._

### 4.1 Descriptive Statistics
- Mean vs. median charges: [fill in] — indicates [right/left]-skewed distribution.
- Skewness: [value] | Kurtosis: [value]

### 4.2 Hypothesis Test 1 — Smokers vs. Non-Smokers (Charges)
- H0: No difference in mean/median charges between smokers and non-smokers.
- H1: A significant difference exists.
- Normality (Shapiro-Wilk): [result]
- Equal variance (Levene's): [result]
- Test used: [t-test / Mann-Whitney U]
- Result: [statistic, p-value]
- **Conclusion (α = 0.05):** [Reject / Fail to Reject H0]

### 4.3 Hypothesis Test 2 — Charges Across Regions (One-Way ANOVA)
- H0: Mean charges are equal across all four regions.
- H1: At least one region differs.
- Result: [F-statistic, p-value]
- **Conclusion (α = 0.05):** [Reject / Fail to Reject H0]

### 4.4 OLS Regression Model
- Model specification: `charges ~ age + bmi + children + smoker + region [+ interaction terms]`
- R²: [value] | Adjusted R²: [value]
- Key coefficients and interpretation: [fill in]

### 4.5 Gauss-Markov Diagnostics
- Linearity/Homoscedasticity (Residuals vs. Fitted): [observation]
- Normality of residuals (Q-Q plot, Jarque-Bera/Omnibus): [result]
- Multicollinearity (VIF): [values; any predictor of concern?]

---

## 5. Dashboard Structure

| Tab | Contents |
|-----|----------|
| **1. Data Exploration** | Sidebar filters (age/BMI sliders, category multi-select), reactive Plotly/Seaborn plots, summary statistics |
| **2. Hypothesis Testing Lab** | Dropdown-driven test selection, automatic computation of test statistics/p-values, plain-language conclusions |
| **3. Live Prediction & Diagnostics** | Interactive inputs for model prediction with 95% CI/PI, residual diagnostic plots |

---

## 6.Deployment 

Live app (Streamlit Community Cloud): `[insert URL if deployed]`

---



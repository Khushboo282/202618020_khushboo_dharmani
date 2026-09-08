"""
Medical Insurance Costs — Interactive Dashboard
Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import (
    shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal
)

st.set_page_config(
    page_title="Medical Insurance Costs Dashboard",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data & model loading (cached so they run once per session, not per widget)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance.csv")
    return df


@st.cache_resource
def fit_model(df):
    """Raw-charges OLS model with smoker*bmi interaction — used for live prediction
    since dollar-scale output is directly interpretable for end users."""
    model = smf.ols(
        "charges ~ age + C(sex) + children + C(region) + C(smoker) * bmi",
        data=df,
    ).fit()
    return model


df = load_data()
model = fit_model(df)

st.title("Medical Insurance Costs — Interactive Dashboard")
st.caption(
    "Lab-4 · M.Sc. Data Science, Sem 1 · Dataset: Medical Insurance Costs (1,338 records)"
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)

# ===========================================================================
# TAB 1 — DATA EXPLORATION
# ===========================================================================
with tab1:
    st.header("Data Exploration")

    st.sidebar.header("Filters (Tab 1)")
    age_range = st.sidebar.slider(
        "Age range", int(df["age"].min()), int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max()))
    )
    bmi_range = st.sidebar.slider(
        "BMI range", float(df["bmi"].min()), float(df["bmi"].max()),
        (float(df["bmi"].min()), float(df["bmi"].max()))
    )
    regions_selected = st.sidebar.multiselect(
        "Region", options=sorted(df["region"].unique()),
        default=sorted(df["region"].unique())
    )
    smoker_selected = st.sidebar.multiselect(
        "Smoker status", options=sorted(df["smoker"].unique()),
        default=sorted(df["smoker"].unique())
    )
    sex_selected = st.sidebar.multiselect(
        "Sex", options=sorted(df["sex"].unique()),
        default=sorted(df["sex"].unique())
    )

    filtered = df[
        (df["age"].between(*age_range)) &
        (df["bmi"].between(*bmi_range)) &
        (df["region"].isin(regions_selected)) &
        (df["smoker"].isin(smoker_selected)) &
        (df["sex"].isin(sex_selected))
    ]

    st.write(f"**{len(filtered):,} records** match the current filters "
             f"(out of {len(df):,} total).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean charges", f"${filtered['charges'].mean():,.0f}")
    col2.metric("Median charges", f"${filtered['charges'].median():,.0f}")
    col3.metric("Mean BMI", f"{filtered['bmi'].mean():.1f}")
    col4.metric("Mean age", f"{filtered['age'].mean():.1f}")

    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe().T.round(2), width='stretch')

    st.subheader("Distribution of Charges")
    fig_hist = px.histogram(
        filtered, x="charges", color="smoker", nbins=40, marginal="box",
        barmode="overlay", opacity=0.7,
        title="Charges Distribution (by Smoker Status)"
    )
    st.plotly_chart(fig_hist, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(
            filtered, x="age", y="charges", color="smoker", opacity=0.6,
            title="Age vs. Charges"
        )
        st.plotly_chart(fig_scatter, width='stretch')
    with c2:
        fig_scatter2 = px.scatter(
            filtered, x="bmi", y="charges", color="smoker", opacity=0.6,
            title="BMI vs. Charges"
        )
        st.plotly_chart(fig_scatter2, width='stretch')

    st.subheader("Correlation Matrix (Numeric Features)")
    numeric_cols = ["age", "bmi", "children", "charges"]
    corr = filtered[numeric_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, width='stretch')

# ===========================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ===========================================================================
with tab2:
    st.header("Hypothesis Testing Lab")
    st.write(
        "Pick a categorical factor and a numerical metric. The app checks "
        "normality (Shapiro-Wilk) and equal variance (Levene's), then "
        "automatically selects the appropriate test."
    )

    cat_cols = ["sex", "smoker", "region"]
    num_cols = ["age", "bmi", "children", "charges"]

    c1, c2 = st.columns(2)
    with c1:
        factor = st.selectbox("Categorical factor", cat_cols, index=1)
    with c2:
        metric = st.selectbox("Numerical metric", num_cols, index=3)

    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01)

    groups = {level: df.loc[df[factor] == level, metric] for level in df[factor].unique()}
    n_groups = len(groups)

    st.subheader("Group Summary")
    summary_rows = [
        {"group": k, "n": len(v), "mean": v.mean(), "median": v.median(), "std": v.std()}
        for k, v in groups.items()
    ]
    st.dataframe(pd.DataFrame(summary_rows).round(2), width='stretch')

    st.subheader("Assumption Checks")
    sw_results = {}
    for k, v in groups.items():
        # Shapiro-Wilk is unreliable/undefined for very small or very large n;
        # dataset groups here are all well within a safe range.
        sw_results[k] = shapiro(v)
    sw_df = pd.DataFrame(
        [{"group": k, "W": r.statistic, "p_value": r.pvalue, "normal (p>alpha)": r.pvalue > alpha}
         for k, r in sw_results.items()]
    )
    st.write("**Shapiro-Wilk (normality, per group):**")
    st.dataframe(sw_df.round(4), width='stretch')

    lev = levene(*groups.values())
    equal_var = lev.pvalue > alpha
    st.write(f"**Levene's test (equal variance):** statistic = {lev.statistic:.4f}, "
             f"p = {lev.pvalue:.4g} → equal variance assumption "
             f"{'holds' if equal_var else 'does NOT hold'} at α={alpha}")

    all_normal = all(r.pvalue > alpha for r in sw_results.values())

    st.subheader("Test Result")
    if n_groups == 2:
        keys = list(groups.keys())
        g1, g2 = groups[keys[0]], groups[keys[1]]
        if all_normal:
            stat, p = ttest_ind(g1, g2, equal_var=equal_var)
            test_name = "Two-Sample t-test"
        else:
            stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U test"
    else:
        if all_normal and equal_var:
            stat, p = f_oneway(*groups.values())
            test_name = "One-Way ANOVA"
        else:
            stat, p = kruskal(*groups.values())
            test_name = "Kruskal-Wallis test"

    conclusion = "Reject H0" if p < alpha else "Fail to Reject H0"

    st.write(f"**H0:** No difference in {metric} across levels of {factor}.")
    st.write(f"**H1:** A significant difference exists.")
    st.write(f"**Test used:** {test_name} (auto-selected based on assumption checks above)")
    st.write(f"**Statistic:** {stat:.4f}  |  **p-value:** {p:.4g}")

    if conclusion == "Reject H0":
        st.success(f"**Conclusion at α={alpha}: {conclusion}** — "
                    f"there is a statistically significant difference in {metric} across {factor} groups.")
    else:
        st.info(f"**Conclusion at α={alpha}: {conclusion}** — "
                f"no statistically significant difference detected in {metric} across {factor} groups.")

    fig_box = px.box(df, x=factor, y=metric, color=factor, title=f"{metric} by {factor}")
    st.plotly_chart(fig_box, width='stretch')

# ===========================================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ===========================================================================
with tab3:
    st.header("Live Prediction & Diagnostics")

    st.subheader("Enter Applicant Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), 35)
        in_sex = st.selectbox("Sex", sorted(df["sex"].unique()))
    with c2:
        in_bmi = st.number_input("BMI", float(df["bmi"].min()), float(df["bmi"].max()), 28.0)
        in_children = st.slider("Children", 0, int(df["children"].max()), 0)
    with c3:
        in_smoker = st.selectbox("Smoker", sorted(df["smoker"].unique()))
        in_region = st.selectbox("Region", sorted(df["region"].unique()))

    input_row = pd.DataFrame([{
        "age": in_age, "sex": in_sex, "bmi": in_bmi,
        "children": in_children, "smoker": in_smoker, "region": in_region,
    }])

    pred = model.get_prediction(input_row)
    pred_summary = pred.summary_frame(alpha=0.05)

    point_estimate = pred_summary["mean"].iloc[0]
    ci_low, ci_high = pred_summary["mean_ci_lower"].iloc[0], pred_summary["mean_ci_upper"].iloc[0]
    pi_low, pi_high = pred_summary["obs_ci_lower"].iloc[0], pred_summary["obs_ci_upper"].iloc[0]

    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Charges", f"${point_estimate:,.0f}")
    m2.metric("95% CI (mean)", f"${ci_low:,.0f} – ${ci_high:,.0f}")
    m3.metric("95% Prediction Interval", f"${pi_low:,.0f} – ${pi_high:,.0f}")
    st.caption(
        "The **confidence interval** bounds the average charges for people with these "
        "characteristics. The (wider) **prediction interval** bounds a single new "
        "individual's charges, which is why it's larger — it accounts for both "
        "estimation uncertainty and natural person-to-person variability."
    )

    st.divider()
    st.subheader("Residual Diagnostic Plots (Model-Wide)")
    st.caption(
        "These reflect the fitted model's overall behavior across all training data, "
        "not just this one prediction."
    )

    fitted = model.fittedvalues
    residuals = model.resid

    d1, d2 = st.columns(2)
    with d1:
        fig_resid = px.scatter(
            x=fitted, y=residuals,
            labels={"x": "Fitted values", "y": "Residuals"},
            title="Residuals vs. Fitted Values"
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, width='stretch')

    with d2:
        qq = sm.ProbPlot(residuals)
        theoretical_q = qq.theoretical_quantiles
        sample_q = qq.sample_quantiles
        fig_qq = px.scatter(
            x=theoretical_q, y=sample_q,
            labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
            title="Q-Q Plot of Residuals"
        )
        min_v, max_v = min(theoretical_q), max(theoretical_q)
        fig_qq.add_shape(
            type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_qq, width='stretch')

    jb_stat, jb_p, jb_skew, jb_kurt = sm.stats.jarque_bera(residuals)
    st.write(
        f"**Model fit:** R² = {model.rsquared:.3f}, Adjusted R² = {model.rsquared_adj:.3f}  |  "
        f"**Jarque-Bera:** statistic = {jb_stat:.1f}, p = {jb_p:.2e} "
        f"(residuals depart from normality — see README for the log-transform comparison)"
    )

"""
app.py  ─  AI Career Platform Analytics Dashboard
Streamlit multi-page app covering:
  1. Descriptive analysis
  2. Diagnostic analysis (ARM + clustering + correlations)
  3. Predictive analysis (classifier + regressor metrics)
  4. Prescriptive analysis (segment playbooks)
  5. New customer prediction (CSV upload → score → download)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerPath Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "career_survey_dataset.csv"
CLF_PATH     = "model_classifier.pkl"
REG_PATH     = "model_regressor.pkl"
KM_PATH      = "model_kmeans.pkl"
ARM_PATH     = "arm_rules.pkl"
METRICS_PATH = "model_metrics.json"

# ── Loaders (cached) ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "wtp_monthly_numeric" not in df.columns:
        wtp_map = {"Nothing - free only": 0, "Up to Rs99/mo": 70,
                   "Rs100-299/mo": 200, "Rs300-499/mo": 400,
                   "Rs500-999/mo": 750, "Above Rs1000/mo": 1200}
        df["wtp_monthly_numeric"] = df["Q28_monthly_wtp"].map(wtp_map).fillna(0)
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for key, path in [("clf", CLF_PATH), ("reg", REG_PATH),
                      ("km", KM_PATH), ("arm", ARM_PATH)]:
        if os.path.exists(path):
            out[key] = joblib.load(path)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            out["metrics"] = json.load(f)
    return out


def models_trained(models):
    return all(k in models for k in ["clf", "reg", "km", "metrics"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 CareerPath Analytics")
    st.markdown("*AI-powered career guidance platform*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Descriptive Analysis",
         "🔍 Diagnostic Analysis",
         "🤖 Predictive Analysis",
         "🎯 Prescriptive Analysis",
         "📤 New Customer Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Train models button
    if st.button("⚙️ Train / Retrain Models", use_container_width=True):
        if not os.path.exists(DATA_PATH):
            st.error("Dataset not found. Upload career_survey_dataset.csv first.")
        else:
            with st.spinner("Training all models… this takes ~60 seconds."):
                try:
                    from train_models import train_all
                    train_all(DATA_PATH)
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.caption("Built with Python · scikit-learn · Streamlit")
    st.caption("Sriju — Career Platform v1.0")

# ── Load data and models ──────────────────────────────────────────────────────
df       = load_dataset()
models   = load_models()
trained  = models_trained(models)

from charts import (
    age_distribution, location_pie, income_bar, stream_pie,
    clarity_bar, wtp_by_persona, wtp_by_location, target_distribution,
    urgency_distribution, state_map, crosstab_heatmap,
    correlation_heatmap, psycho_radar, arm_scatter, arm_top_rules,
    cluster_elbow, cluster_wtp_bar,
    roc_curve_plot, confusion_matrix_plot, feature_importance_plot,
    metrics_gauge,
    priority_donut, wtp_prediction_hist, adoption_prob_hist,
    scatter_wtp_vs_prob,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🎯 AI Career Platform — Analytics Dashboard")
    st.markdown(
        "Data-driven insights from the student career guidance survey. "
        "Use the sidebar to navigate across all four analysis layers, "
        "or upload new respondent data to get instant predictions."
    )
    st.markdown("---")

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total respondents", f"{len(df):,}")
        adopt_pct = (df["Q31_platform_adoption"].isin(
            ["Definitely would use", "Likely would use"]).sum() / len(df) * 100)
        c2.metric("Platform adoption intent", f"{adopt_pct:.1f}%")
        avg_wtp = df["wtp_monthly_numeric"].mean()
        c3.metric("Avg monthly WTP", f"₹{avg_wtp:.0f}")
        urgent = (df["Q14_decision_urgency"].isin(
            ["Within 3 months", "3-6 months"]).sum() / len(df) * 100)
        c4.metric("Urgent decision (≤6 mo)", f"{urgent:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analysis layers")
        layers = {
            "📊 Descriptive": "Who is in the market — distributions, crosstabs, geographic spread",
            "🔍 Diagnostic": "Why students behave as they do — ARM rules, clusters, correlations",
            "🤖 Predictive": "Which students will convert — RF classifier + GBM regressor + SHAP",
            "🎯 Prescriptive": "What to do — persona playbooks, pricing tiers, channel strategy",
        }
        for k, v in layers.items():
            st.markdown(f"**{k}** — {v}")

    with col2:
        st.subheader("Model status")
        if trained:
            m = models["metrics"]
            st.success("All models trained and ready")
            st.markdown(f"- Classifier accuracy: **{m['classifier']['accuracy']:.1%}**")
            st.markdown(f"- Classifier ROC-AUC: **{m['classifier']['roc_auc']:.3f}**")
            st.markdown(f"- Classifier F1 score: **{m['classifier']['f1_score']:.3f}**")
            st.markdown(f"- WTP regressor MAE: **₹{m['regressor']['mae']:.0f}**")
            st.markdown(f"- WTP regressor R²: **{m['regressor']['r2']:.4f}**")
            st.markdown(f"- ARM rules found: **{m['arm']['total_rules']}**")
            st.markdown(f"- Best k (clusters): **{m['clustering']['best_k']}**")
            st.markdown(f"- Training rows: **{m['training_rows']:,}**")
        else:
            st.warning("Models not yet trained. Click **⚙️ Train / Retrain Models** in the sidebar.")

    if not trained:
        st.info(
            "**Quick start:** Make sure `career_survey_dataset.csv` is in the same folder "
            "as the app, then click **⚙️ Train / Retrain Models** in the sidebar."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Descriptive Analysis":
    st.title("📊 Descriptive Analysis")
    st.markdown("*What does our student audience look like?*")

    if df is None:
        st.error("Dataset not found. Place `career_survey_dataset.csv` in the app folder.")
        st.stop()

    st.markdown("### Demographics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(age_distribution(df), use_container_width=True)
    with c2:
        st.plotly_chart(location_pie(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(income_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(stream_pie(df), use_container_width=True)

    st.markdown("### Geographic spread")
    st.plotly_chart(state_map(df), use_container_width=True)

    st.markdown("### Career clarity and urgency")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(clarity_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(urgency_distribution(df), use_container_width=True)

    st.markdown("### Willingness to pay")
    c1, c2 = st.columns(2)
    with c1:
        if "persona_label" in df.columns:
            st.plotly_chart(wtp_by_persona(df), use_container_width=True)
    with c2:
        st.plotly_chart(wtp_by_location(df), use_container_width=True)

    st.markdown("### Target variable — platform adoption (Q31)")
    st.plotly_chart(target_distribution(df), use_container_width=True)

    st.markdown("### Cross-tab explorer")
    col_opts = ["Q4_location", "Q5_income", "Q7_stream", "Q1_age",
                "Q10_career_clarity", "Q14_decision_urgency",
                "Q28_monthly_wtp", "persona_label"]
    col_opts = [c for c in col_opts if c in df.columns]
    cx, cy = st.columns(2)
    x_col = cx.selectbox("X axis", col_opts, index=0)
    y_col = cy.selectbox("Y axis", col_opts, index=3)
    if x_col != y_col:
        st.plotly_chart(
            crosstab_heatmap(df, x_col, y_col,
                             f"Cross-tab: {x_col} × {y_col}"),
            use_container_width=True,
        )

    st.markdown("### Raw data sample")
    st.dataframe(df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analysis":
    st.title("🔍 Diagnostic Analysis")
    st.markdown("*Why do students behave the way they do?*")

    if not trained:
        st.warning("Train models first using the sidebar button.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["🔗 Association Rule Mining", "👥 Clustering", "📈 Correlations & Psychographics"]
    )

    # ── ARM tab ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Association Rule Mining — Apriori")
        st.markdown(
            "Finds relationships between student interests, career choices, "
            "learning styles and feature preferences."
        )
        rules_df = models.get("arm")
        if rules_df is None or (isinstance(rules_df, pd.DataFrame) and rules_df.empty):
            st.info("No association rules found at current thresholds.")
        else:
            m = models["metrics"]["arm"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total rules", m["total_rules"])
            c2.metric("Min support", m["min_support"])
            c3.metric("Min confidence", m["min_confidence"])

            st.plotly_chart(arm_scatter(rules_df), use_container_width=True)
            st.plotly_chart(arm_top_rules(rules_df, n=25), use_container_width=True)

            st.markdown("#### Filter rules")
            min_lift = st.slider("Minimum lift", 1.0, float(rules_df["lift"].max()),
                                 value=1.2, step=0.05)
            filtered = rules_df[rules_df["lift"] >= min_lift].reset_index(drop=True)
            st.markdown(f"**{len(filtered)} rules** with lift ≥ {min_lift}")
            st.dataframe(
                filtered[["antecedent", "consequent", "support", "confidence", "lift"]],
                use_container_width=True,
            )

    # ── Clustering tab ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("K-Means Persona Clustering")
        km_bundle = models.get("km", {})
        if not km_bundle:
            st.info("K-Means model not found.")
        else:
            m_cl = models["metrics"]["clustering"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best k", m_cl["best_k"])
            c2.metric("Training rows", models["metrics"]["training_rows"])
            c3.metric("Silhouette score", max(m_cl["silhouette_scores"]))

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(cluster_elbow(km_bundle), use_container_width=True)
            with c2:
                st.plotly_chart(cluster_wtp_bar(km_bundle), use_container_width=True)

            if df is not None and "persona_label" in df.columns:
                st.subheader("Persona profiles from training data")
                psych_cols = [c for c in df.columns if "Q25_psych" in c]
                grp_cols = ["wtp_monthly_numeric", "urgency_score", "clarity_score"] + psych_cols
                grp_cols = [c for c in grp_cols if c in df.columns]
                profile = df.groupby("persona_label")[grp_cols].mean().round(2)
                st.dataframe(profile, use_container_width=True)

    # ── Correlations tab ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Correlation matrix")
        if df is not None:
            st.plotly_chart(correlation_heatmap(df), use_container_width=True)

        st.subheader("Psychographic radar by persona")
        if df is not None and "persona_label" in df.columns:
            st.plotly_chart(psycho_radar(df), use_container_width=True)
        else:
            st.info("Persona labels needed for radar chart.")

        if df is not None:
            st.subheader("WTP driver analysis")
            st.markdown("Mean monthly WTP grouped by key variables:")
            for col in ["Q4_location", "Q5_income", "Q1_age",
                        "Q7_stream", "Q14_decision_urgency"]:
                if col in df.columns and "wtp_monthly_numeric" in df.columns:
                    grp = (df.groupby(col)["wtp_monthly_numeric"]
                           .agg(["mean", "median", "count"])
                           .round(1)
                           .rename(columns={"mean": "Mean WTP ₹",
                                            "median": "Median WTP ₹",
                                            "count": "n"}))
                    with st.expander(f"WTP by {col}"):
                        st.dataframe(grp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Analysis":
    st.title("🤖 Predictive Analysis")
    st.markdown("*Which students will convert, and how much will they pay?*")

    if not trained:
        st.warning("Train models first using the sidebar button.")
        st.stop()

    m = models["metrics"]
    mc = m["classifier"]
    mr = m["regressor"]

    tab1, tab2 = st.tabs(["🎯 Classifier — Platform Adoption", "💰 Regressor — WTP Prediction"])

    with tab1:
        st.subheader("Random Forest Classifier — predicts Q31 platform adoption")
        st.markdown("Binary target: **1** = Definitely/Likely would use | **0** = Neutral/Unlikely/No")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{mc['accuracy']:.1%}")
        c2.metric("Precision", f"{mc['precision']:.1%}")
        c3.metric("Recall",    f"{mc['recall']:.1%}")
        c4.metric("F1 Score",  f"{mc['f1_score']:.1%}")
        c5.metric("ROC-AUC",   f"{mc['roc_auc']:.3f}")

        cv_mean = mc.get("cv_f1_mean", 0)
        cv_std  = mc.get("cv_f1_std", 0)
        st.info(f"**5-fold cross-validation F1:** {cv_mean:.4f} ± {cv_std:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(roc_curve_plot(m), use_container_width=True)
        with c2:
            st.plotly_chart(confusion_matrix_plot(m), use_container_width=True)

        st.plotly_chart(feature_importance_plot(m, "classifier"), use_container_width=True)

        st.markdown("#### Detailed classification report")
        report = mc.get("classification_report", {})
        if report:
            report_rows = []
            for label, vals in report.items():
                if isinstance(vals, dict):
                    report_rows.append({
                        "Class": label,
                        "Precision": round(vals.get("precision", 0), 3),
                        "Recall": round(vals.get("recall", 0), 3),
                        "F1-score": round(vals.get("f1-score", 0), 3),
                        "Support": int(vals.get("support", 0)),
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

    with tab2:
        st.subheader("Gradient Boosting Regressor — predicts monthly WTP (₹)")

        c1, c2 = st.columns(2)
        c1.metric("Mean Absolute Error", f"₹{mr['mae']:.0f}")
        c2.metric("R² Score", f"{mr['r2']:.4f}")

        st.plotly_chart(feature_importance_plot(m, "regressor"), use_container_width=True)

        if df is not None:
            st.markdown("#### Actual WTP distribution (training data)")
            import plotly.express as px
            fig_hist = px.histogram(
                df, x="wtp_monthly_numeric", nbins=20,
                color_discrete_sequence=["#185FA5"],
                labels={"wtp_monthly_numeric": "Monthly WTP (₹)"},
            )
            fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20),
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prescriptive Analysis":
    st.title("🎯 Prescriptive Analysis")
    st.markdown("*What should we actually do — per segment, per channel, per rupee?*")

    st.markdown("### Priority segment targeting")

    segments = {
        "🔴 Hot Leads — Focused Climbers + Anxious Achievers": {
            "Who": "UG students, metro/Tier-2, income ₹5–20L, urgency ≤6 months, past payer",
            "Size": "~22% of audience",
            "Avg WTP": "₹400–500/mo",
            "Channel": "Instagram + LinkedIn ads, WhatsApp parent messages",
            "Offer": "7-day free trial → Standard plan ₹299/mo",
            "Message tone": "Aspiration + urgency: 'Your peers already have a roadmap.'",
            "Priority": "🔴 Act this week",
        },
        "🟡 Freemium Converts — Confused Drifters": {
            "Who": "Class 10–12, any location, confused, no past spend",
            "Size": "~30% of audience",
            "Avg WTP": "₹100–200/mo",
            "Channel": "YouTube pre-roll, school counsellor partnerships",
            "Offer": "Free psychometric test → ₹99/mo starter plan",
            "Message tone": "Reassurance: 'It's okay to be confused — we'll figure it out together.'",
            "Priority": "🟡 Build awareness funnel",
        },
        "🟠 Re-engage — Pragmatic Followers (parent-driven)": {
            "Who": "Parents decide + pay, income ₹5–15L, Tier-2/3 cities",
            "Size": "~10% of audience",
            "Avg WTP": "₹300–400/mo",
            "Channel": "WhatsApp forward content, Facebook parent groups, school events",
            "Offer": "One-time career report ₹499 → subscription upsell",
            "Message tone": "ROI-focused: 'Your child's career decision is worth ₹499.'",
            "Priority": "🟠 Parent-first messaging",
        },
        "🟢 Curious Explorers — Low urgency, open mindset": {
            "Who": "Class 9–11, exploratory, no deadline pressure",
            "Size": "~16% of audience",
            "Avg WTP": "₹100–250/mo",
            "Channel": "YouTube organic, Instagram reels, school ambassador programme",
            "Offer": "Free emerging careers quiz, newsletter sign-up",
            "Message tone": "Discovery: 'Have you heard of these 10 careers that didn't exist 5 years ago?'",
            "Priority": "🟢 Plant early seeds",
        },
    }

    for seg_name, seg_data in segments.items():
        with st.expander(seg_name, expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                for k in ["Who", "Size", "Avg WTP", "Priority"]:
                    st.markdown(f"**{k}:** {seg_data[k]}")
            with c2:
                for k in ["Channel", "Offer", "Message tone"]:
                    st.markdown(f"**{k}:** {seg_data[k]}")

    st.markdown("---")
    st.markdown("### Pricing strategy from WTP data")

    pricing_tiers = pd.DataFrame([
        {"Tier": "Free",      "Price": "₹0",        "Target persona": "Confused Drifter, Curious Explorer", "Features": "Psychometric test + 1 career suggestion"},
        {"Tier": "Starter",   "Price": "₹99/mo",    "Target persona": "Confused Drifter, Curious Explorer", "Features": "Free + AI chatbot (limited) + 3 roadmap steps"},
        {"Tier": "Standard",  "Price": "₹299/mo",   "Target persona": "Anxious Achiever, Pragmatic Follower","Features": "Starter + full roadmap + college shortlist + exam tracker"},
        {"Tier": "Premium",   "Price": "₹599/mo",   "Target persona": "Focused Climber, Anxious Achiever",  "Features": "Standard + 2 mentor sessions/mo + salary dashboard"},
        {"Tier": "Elite",     "Price": "₹999/mo",   "Target persona": "Focused Climber",                    "Features": "Premium + unlimited mentorship + parent dashboard + priority support"},
        {"Tier": "One-time",  "Price": "₹499",      "Target persona": "Pragmatic Follower (parents)",       "Features": "Full career report PDF + college shortlist + action plan"},
    ])
    st.dataframe(pricing_tiers, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Channel strategy by persona")
    channel_data = pd.DataFrame([
        {"Persona": "Confused Drifter",   "Primary channel": "YouTube",       "Secondary": "School counsellors", "Message": "Reassurance-first"},
        {"Persona": "Anxious Achiever",   "Primary channel": "Instagram",     "Secondary": "WhatsApp (student)", "Message": "Urgency + aspiration"},
        {"Persona": "Focused Climber",    "Primary channel": "LinkedIn",      "Secondary": "Email drip",         "Message": "Data + outcomes"},
        {"Persona": "Curious Explorer",   "Primary channel": "Instagram reel","Secondary": "YouTube Shorts",     "Message": "Discovery + wonder"},
        {"Persona": "Pragmatic Follower", "Primary channel": "WhatsApp",      "Secondary": "Facebook (parents)", "Message": "ROI + safety"},
    ])
    st.dataframe(channel_data, use_container_width=True, hide_index=True)

    if trained:
        st.markdown("---")
        st.markdown("### Top ARM-derived product recommendations")
        rules_df = models.get("arm")
        if rules_df is not None and not rules_df.empty:
            st.markdown(
                "Showing association rules where the **consequent** is a platform feature — "
                "these rules tell us which student profiles to push which features to."
            )
            feature_rules = rules_df[
                rules_df["consequent"].str.startswith("feature:")
            ].head(15)
            if not feature_rules.empty:
                st.dataframe(
                    feature_rules[["antecedent", "consequent",
                                   "support", "confidence", "lift"]],
                    use_container_width=True,
                )
            else:
                st.dataframe(
                    rules_df[["antecedent", "consequent",
                               "support", "confidence", "lift"]].head(15),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEW CUSTOMER PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤 New Customer Prediction":
    st.title("📤 New Customer Prediction")
    st.markdown(
        "Upload a CSV file of new survey respondents. "
        "The app will score every row with: **persona**, **adoption probability**, "
        "**predicted WTP**, **priority tier**, and **recommended marketing action**."
    )

    if not trained:
        st.warning("Train models first using the sidebar button.")
        st.stop()

    st.markdown("---")

    # Template download
    if os.path.exists(DATA_PATH):
        df_template = pd.read_csv(DATA_PATH, nrows=3)
        template_cols = [c for c in df_template.columns
                         if not c.startswith("pred_") and c != "persona_label"]
        csv_template = df_template[template_cols].to_csv(index=False)
        st.download_button(
            "⬇️ Download sample CSV template (3 rows)",
            data=csv_template,
            file_name="new_customers_template.csv",
            mime="text/csv",
        )
        st.caption("Use this template to format your new respondent data before uploading.")

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload new respondent CSV",
        type=["csv"],
        help="CSV must have the same column names as the training dataset.",
    )

    if uploaded is not None:
        with st.spinner("Scoring respondents…"):
            try:
                df_new = pd.read_csv(uploaded, low_memory=False)
                st.success(f"Loaded {len(df_new):,} respondents from uploaded file.")

                from predictor import predict_new_customers, score_summary
                scored = predict_new_customers(df_new)
                summary = score_summary(scored)

                st.markdown("### Prediction summary")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total scored", summary["total"])
                c2.metric("Hot Leads 🔴", summary["hot_leads"])
                c3.metric("Freemium 🟢", summary["freemium"])
                c4.metric("Re-engage 🟠", summary["reengage"])
                c5.metric("Nurture ⚪", summary["nurture"])

                c1, c2 = st.columns(2)
                c1.metric("Avg adoption probability",
                           f"{summary['avg_adoption_prob']:.1%}")
                c2.metric("Avg predicted WTP",
                           f"₹{summary['avg_predicted_wtp']:.0f}/mo")

                st.markdown("---")
                st.markdown("### Visualisations")
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.plotly_chart(priority_donut(scored), use_container_width=True)
                with vc2:
                    st.plotly_chart(scatter_wtp_vs_prob(scored), use_container_width=True)

                vc1, vc2 = st.columns(2)
                with vc1:
                    st.plotly_chart(adoption_prob_hist(scored), use_container_width=True)
                with vc2:
                    st.plotly_chart(wtp_prediction_hist(scored), use_container_width=True)

                st.markdown("---")
                st.markdown("### Scored respondents")

                pred_cols = [c for c in scored.columns
                             if c.startswith("pred_") or c in ["respondent_id", "Q3_state", "Q4_location", "Q1_age"]]
                st.dataframe(scored[pred_cols].head(200), use_container_width=True)

                # Download
                scored_csv = scored.to_csv(index=False)
                st.download_button(
                    "⬇️ Download full scored CSV",
                    data=scored_csv,
                    file_name="scored_new_customers.csv",
                    mime="text/csv",
                )

                # Hot leads table
                hot = scored[scored["pred_priority_tier"] == "Hot Lead"].sort_values(
                    "pred_adoption_probability", ascending=False
                )
                if not hot.empty:
                    st.markdown(f"### 🔴 Hot leads ({len(hot)} respondents)")
                    st.markdown(
                        "These respondents have both high adoption probability (≥65%) "
                        "and high predicted WTP (≥₹300/mo). Act first."
                    )
                    show_cols = [c for c in [
                        "respondent_id", "Q1_age", "Q3_state", "Q4_location",
                        "Q7_stream", "Q14_decision_urgency",
                        "pred_adoption_probability", "pred_wtp_monthly_inr",
                        "pred_wtp_tier", "pred_marketing_action"
                    ] if c in hot.columns]
                    st.dataframe(hot[show_cols].head(50), use_container_width=True)

            except FileNotFoundError as e:
                st.error(f"Model files missing: {e}. Click 'Train / Retrain Models' first.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

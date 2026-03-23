"""
charts.py
All Plotly chart factory functions used by the Streamlit app.
Each function returns a plotly Figure object.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colour palette
PERSONA_COLORS = {
    "Confused Drifter":   "#534AB7",
    "Anxious Achiever":   "#0F6E56",
    "Focused Climber":    "#185FA5",
    "Curious Explorer":   "#854F0B",
    "Pragmatic Follower": "#993C1D",
}
PRIORITY_COLORS = {
    "Hot Lead":          "#E24B4A",
    "Freemium Convert":  "#1D9E75",
    "Re-engage":         "#EF9F27",
    "Nurture":           "#888780",
}
PALETTE   = px.colors.qualitative.Set2
BLUE_SEQ  = px.colors.sequential.Blues
TEAL_SEQ  = px.colors.sequential.Teal


def _fig_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ── DESCRIPTIVE ───────────────────────────────────────────────────────────────

def age_distribution(df):
    order = ["Under 15", "15-17", "18-20", "21-23", "24 or above"]
    vc = df["Q1_age"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Age group", "Count"]
    fig = px.bar(vc, x="Age group", y="Count", color="Age group",
                 color_discrete_sequence=PALETTE)
    return _fig_layout(fig, "Age distribution")


def location_pie(df):
    vc = df["Q4_location"].value_counts().reset_index()
    vc.columns = ["Location", "Count"]
    fig = px.pie(vc, names="Location", values="Count",
                 color_discrete_sequence=PALETTE, hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _fig_layout(fig, "Location split", height=340)


def income_bar(df):
    order = ["Below Rs2L", "Rs2-5L", "Rs5-10L", "Rs10-20L", "Above Rs20L", "Prefer not to say"]
    vc = df["Q5_income"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Income", "Count"]
    fig = px.bar(vc, x="Income", y="Count", color="Income",
                 color_discrete_sequence=PALETTE)
    return _fig_layout(fig, "Household income distribution")


def stream_pie(df):
    vc = df["Q7_stream"].value_counts().reset_index()
    vc.columns = ["Stream", "Count"]
    fig = px.pie(vc, names="Stream", values="Count",
                 color_discrete_sequence=PALETTE, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _fig_layout(fig, "Stream / field distribution", height=340)


def clarity_bar(df):
    order = ["Very confused", "Somewhat confused", "Just exploring",
             "Mostly clear", "Completely clear"]
    vc = df["Q10_career_clarity"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Clarity level", "Count"]
    colors = ["#E24B4A", "#EF9F27", "#888780", "#1D9E75", "#185FA5"]
    fig = px.bar(vc, x="Clarity level", y="Count",
                 color="Clarity level",
                 color_discrete_sequence=colors)
    return _fig_layout(fig, "Career clarity distribution")


def wtp_by_persona(df):
    if "persona_label" not in df.columns or "wtp_monthly_numeric" not in df.columns:
        return go.Figure()
    grp = df.groupby("persona_label")["wtp_monthly_numeric"].mean().reset_index()
    grp.columns = ["Persona", "Avg WTP (₹/mo)"]
    grp = grp.sort_values("Avg WTP (₹/mo)", ascending=True)
    colors = [PERSONA_COLORS.get(p, "#888") for p in grp["Persona"]]
    fig = px.bar(grp, x="Avg WTP (₹/mo)", y="Persona",
                 orientation="h", color="Persona",
                 color_discrete_map=PERSONA_COLORS)
    return _fig_layout(fig, "Mean monthly WTP by persona")


def wtp_by_location(df):
    if "wtp_monthly_numeric" not in df.columns:
        return go.Figure()
    grp = df.groupby("Q4_location")["wtp_monthly_numeric"].mean().reset_index()
    grp.columns = ["Location", "Avg WTP (₹/mo)"]
    grp = grp.sort_values("Avg WTP (₹/mo)", ascending=False)
    fig = px.bar(grp, x="Location", y="Avg WTP (₹/mo)",
                 color="Location", color_discrete_sequence=PALETTE)
    return _fig_layout(fig, "Mean monthly WTP by location")


def target_distribution(df):
    order = ["Definitely would use", "Likely would use", "Neutral",
             "Unlikely to use", "Definitely would NOT use"]
    vc = df["Q31_platform_adoption"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Response", "Count"]
    colors = ["#085041", "#1D9E75", "#888780", "#EF9F27", "#E24B4A"]
    fig = px.bar(vc, x="Response", y="Count", color="Response",
                 color_discrete_sequence=colors)
    return _fig_layout(fig, "Platform adoption intent (Q31 — target variable)")


def urgency_distribution(df):
    order = ["Within 3 months", "3-6 months", "6-12 months",
             "1-2 years", "More than 2 years"]
    vc = df["Q14_decision_urgency"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Urgency", "Count"]
    fig = px.bar(vc, x="Urgency", y="Count", color="Urgency",
                 color_discrete_sequence=PALETTE)
    return _fig_layout(fig, "Decision urgency distribution (Q14)")


def state_map(df):
    vc = df["Q3_state"].value_counts().reset_index()
    vc.columns = ["State", "Count"]
    fig = px.bar(vc.head(15), x="Count", y="State", orientation="h",
                 color="Count", color_continuous_scale=BLUE_SEQ)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _fig_layout(fig, "Top 15 states by respondent count", height=420)


def crosstab_heatmap(df, col_x, col_y, title=""):
    ct = pd.crosstab(df[col_y], df[col_x])
    fig = px.imshow(ct, color_continuous_scale="Blues",
                    text_auto=True, aspect="auto")
    return _fig_layout(fig, title or f"{col_x} × {col_y}", height=400)


# ── DIAGNOSTIC ────────────────────────────────────────────────────────────────

def correlation_heatmap(df):
    num_cols = [
        "wtp_monthly_numeric", "urgency_score", "income_numeric",
        "clarity_score", "psycho_composite",
        "Q25_psych_fear_wrong_choice", "Q25_psych_prefer_independent",
        "Q25_psych_financial_over_passion", "Q25_psych_risk_tolerance",
        "Q25_psych_long_term_thinking",
    ]
    present = [c for c in num_cols if c in df.columns]
    corr = df[present].corr().round(3)
    short = [c.replace("Q25_psych_", "").replace("_", " ")[:16] for c in present]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale="RdBu_r", zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    return _fig_layout(fig, "Correlation matrix — key numeric columns", height=460)


def psycho_radar(df):
    if "persona_label" not in df.columns:
        return go.Figure()
    psych_cols = [
        "Q25_psych_fear_wrong_choice", "Q25_psych_prefer_independent",
        "Q25_psych_financial_over_passion", "Q25_psych_risk_tolerance",
        "Q25_psych_long_term_thinking",
    ]
    labels = ["Fear of wrong\nchoice", "Autonomy", "Money > Passion",
              "Risk tolerance", "Long-term thinking"]
    present = [c for c in psych_cols if c in df.columns]
    if not present:
        return go.Figure()

    fig = go.Figure()
    for persona, color in PERSONA_COLORS.items():
        sub = df[df["persona_label"] == persona]
        if sub.empty:
            continue
        vals = [sub[c].mean() for c in present]
        vals += vals[:1]
        lbl  = labels[:len(present)] + labels[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=lbl, fill="toself",
            name=persona, line=dict(color=color),
            opacity=0.7,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[1, 5])))
    return _fig_layout(fig, "Psychographic radar by persona", height=440)


def arm_scatter(rules_df: pd.DataFrame):
    if rules_df is None or rules_df.empty:
        return go.Figure()
    fig = px.scatter(
        rules_df.head(80),
        x="support", y="confidence", size="lift", color="lift",
        hover_data=["antecedent", "consequent", "support", "confidence", "lift"],
        color_continuous_scale="Viridis",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        size_max=22,
    )
    return _fig_layout(fig, "Association rules — support vs confidence (bubble size = lift)", height=420)


def arm_top_rules(rules_df: pd.DataFrame, n=20):
    if rules_df is None or rules_df.empty:
        return go.Figure()
    top = rules_df.head(n).copy()
    top["rule"] = top["antecedent"].str[:28] + " → " + top["consequent"].str[:28]
    fig = px.bar(top, x="lift", y="rule", orientation="h",
                 color="confidence", color_continuous_scale="Teal",
                 hover_data=["support", "confidence", "lift"])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _fig_layout(fig, f"Top {n} association rules by lift", height=520)


def cluster_elbow(km_bundle: dict):
    fig = go.Figure()
    ks = list(range(2, 2 + len(km_bundle["inertias"])))
    fig.add_trace(go.Scatter(x=ks, y=km_bundle["inertias"],
                             mode="lines+markers", name="Inertia",
                             line=dict(color="#185FA5")))
    fig.add_trace(go.Scatter(x=ks, y=km_bundle["sil_scores"],
                             mode="lines+markers", name="Silhouette",
                             yaxis="y2", line=dict(color="#E24B4A", dash="dash")))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="Silhouette score"),
        yaxis=dict(title="Inertia"),
        xaxis=dict(title="k (number of clusters)"),
    )
    best_k = km_bundle.get("best_k", 5)
    fig.add_vline(x=best_k, line_dash="dot", line_color="#EF9F27",
                  annotation_text=f"Best k={best_k}")
    return _fig_layout(fig, "K-Means elbow + silhouette curve", height=360)


def cluster_wtp_bar(km_bundle: dict):
    data = km_bundle.get("cluster_wtp", {})
    sizes = km_bundle.get("cluster_sizes", {})
    clusters = list(data.keys())
    wtps = [data[c] for c in clusters]
    szs  = [sizes.get(c, 0) for c in clusters]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Cluster {c}" for c in clusters],
        y=wtps, name="Avg WTP (₹/mo)",
        text=[f"n={s}" for s in szs],
        textposition="outside",
        marker_color=PALETTE[:len(clusters)],
    ))
    fig.update_layout(yaxis_title="Avg WTP (₹/mo)")
    return _fig_layout(fig, "Mean WTP per discovered cluster")


# ── PREDICTIVE — CLASSIFICATION ───────────────────────────────────────────────

def roc_curve_plot(metrics: dict):
    roc = metrics["classifier"]["roc_curve"]
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    auc = roc["auc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc:.3f})",
                             line=dict(color="#185FA5", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random baseline",
                             line=dict(color="#888", dash="dash")))
    fig.update_layout(xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return _fig_layout(fig, "ROC curve — platform adoption classifier", height=380)


def confusion_matrix_plot(metrics: dict):
    cm = np.array(metrics["classifier"]["confusion_matrix"])
    labels = ["Will not adopt", "Will adopt"]
    fig = px.imshow(cm, x=labels, y=labels,
                    color_continuous_scale="Blues",
                    text_auto=True, aspect="auto")
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return _fig_layout(fig, "Confusion matrix", height=320)


def feature_importance_plot(metrics: dict, model_key="classifier", n=20):
    feats = metrics[model_key]["top_features"]
    df_fi = pd.DataFrame(list(feats.items()), columns=["Feature", "Importance"])
    df_fi = df_fi.sort_values("Importance").tail(n)
    df_fi["Feature"] = df_fi["Feature"].str.replace("_enc", "").str.replace("__", " — ").str[:40]
    fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    title = "Top feature importances — classifier" if model_key == "classifier" \
        else "Top feature importances — WTP regressor"
    return _fig_layout(fig, title, height=520)


def metrics_gauge(value: float, title: str, max_val: float = 1.0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100 if max_val == 1.0 else value,
        title={"text": title, "font": {"size": 13}},
        number={"suffix": "%" if max_val == 1.0 else "", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100 if max_val == 1.0 else max_val]},
            "bar": {"color": "#185FA5"},
            "steps": [
                {"range": [0, 60], "color": "#fee2e2"},
                {"range": [60, 80], "color": "#fef3c7"},
                {"range": [80, 100], "color": "#d1fae5"},
            ],
        },
    ))
    return _fig_layout(fig, "", height=220)


# ── PRESCRIPTIVE / NEW CUSTOMER ───────────────────────────────────────────────

def priority_donut(scored_df: pd.DataFrame):
    vc = scored_df["pred_priority_tier"].value_counts().reset_index()
    vc.columns = ["Priority", "Count"]
    colors = [PRIORITY_COLORS.get(p, "#888") for p in vc["Priority"]]
    fig = px.pie(vc, names="Priority", values="Count",
                 color="Priority",
                 color_discrete_map=PRIORITY_COLORS, hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _fig_layout(fig, "Lead priority distribution", height=340)


def wtp_prediction_hist(scored_df: pd.DataFrame):
    fig = px.histogram(scored_df, x="pred_wtp_monthly_inr",
                       nbins=20, color_discrete_sequence=["#185FA5"])
    fig.update_layout(xaxis_title="Predicted monthly WTP (₹)",
                      yaxis_title="Count")
    return _fig_layout(fig, "Predicted WTP distribution — new customers")


def adoption_prob_hist(scored_df: pd.DataFrame):
    fig = px.histogram(scored_df, x="pred_adoption_probability",
                       nbins=20, color_discrete_sequence=["#1D9E75"])
    fig.update_layout(xaxis_title="Adoption probability",
                      yaxis_title="Count")
    return _fig_layout(fig, "Adoption probability distribution — new customers")


def scatter_wtp_vs_prob(scored_df: pd.DataFrame):
    fig = px.scatter(
        scored_df,
        x="pred_adoption_probability",
        y="pred_wtp_monthly_inr",
        color="pred_priority_tier",
        color_discrete_map=PRIORITY_COLORS,
        hover_data=["pred_wtp_tier", "pred_priority_tier"],
        opacity=0.7,
    )
    fig.add_vline(x=0.65, line_dash="dot", line_color="#888",
                  annotation_text="Adoption threshold")
    fig.add_hline(y=300, line_dash="dot", line_color="#888",
                  annotation_text="High WTP threshold")
    fig.update_layout(xaxis_title="Adoption probability",
                      yaxis_title="Predicted WTP (₹/mo)")
    return _fig_layout(fig, "Lead quadrant map — probability vs WTP", height=420)

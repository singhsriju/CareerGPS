"""
predictor.py
Loads trained models and scores new customer CSV files.
Returns a DataFrame with persona, adoption probability, WTP prediction,
priority tier, and recommended marketing action per respondent.
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

from data_loader import (
    load_data, encode_for_model, engineer_features, ENCODER_PATH,
)

CLF_PATH     = "model_classifier.pkl"
REG_PATH     = "model_regressor.pkl"
KM_PATH      = "model_kmeans.pkl"
FEATURE_PATH = "feature_names.pkl"

WTP_NUM_MAP = {
    "Nothing - free only": 0, "Up to Rs99/mo": 70,
    "Rs100-299/mo": 200, "Rs300-499/mo": 400,
    "Rs500-999/mo": 750, "Above Rs1000/mo": 1200,
}

CLUSTER_PERSONA_MAP = {}   # populated at runtime from training data profile


def _align_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Align new data columns to match training feature set."""
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names]


def _wtp_tier(wtp_val: float) -> str:
    if wtp_val <= 0:
        return "Free only"
    elif wtp_val <= 99:
        return "Starter (≤₹99/mo)"
    elif wtp_val <= 299:
        return "Basic (₹100–299/mo)"
    elif wtp_val <= 499:
        return "Standard (₹300–499/mo)"
    elif wtp_val <= 999:
        return "Premium (₹500–999/mo)"
    else:
        return "Elite (₹1000+/mo)"


def _priority_tier(prob: float, wtp: float) -> str:
    high_intent = prob >= 0.65
    high_wtp    = wtp >= 300
    if high_intent and high_wtp:
        return "Hot Lead"
    elif high_intent and not high_wtp:
        return "Freemium Convert"
    elif not high_intent and high_wtp:
        return "Re-engage"
    else:
        return "Nurture"


def _marketing_action(priority: str, wtp: float, urgency: float) -> str:
    actions = {
        "Hot Lead": (
            "Direct outreach via WhatsApp + Instagram. "
            "Offer 7-day free trial with personal onboarding call. "
            "Highlight roadmap + mentorship features."
        ),
        "Freemium Convert": (
            "Push freemium sign-up with psychometric test hook. "
            "Email drip campaign showing platform value. "
            "Upsell after first assessment completion."
        ),
        "Re-engage": (
            "Target parents via WhatsApp/Facebook with ROI messaging. "
            "Offer detailed career report at one-time fee (₹499). "
            "Emphasise salary data and college shortlist features."
        ),
        "Nurture": (
            "Add to awareness email list. "
            "Retarget via YouTube pre-roll with confusion-based messaging. "
            "Share free career clarity quiz as entry point."
        ),
    }
    base = actions.get(priority, "Add to general nurture list.")
    if urgency and urgency <= 2:
        base += " URGENT: Decision within 6 months — prioritise this week."
    return base


def predict_new_customers(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Score a new DataFrame of survey respondents.
    Returns original df with prediction columns appended.
    """
    if not all(os.path.exists(p) for p in [CLF_PATH, REG_PATH, FEATURE_PATH]):
        raise FileNotFoundError(
            "Trained models not found. Run train_models.py first."
        )

    clf_bundle  = joblib.load(CLF_PATH)
    reg_bundle  = joblib.load(REG_PATH)
    km_bundle   = joblib.load(KM_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    clf     = clf_bundle["model"]
    imp_clf = clf_bundle["imputer"]
    reg     = reg_bundle["model"]
    imp_reg = reg_bundle["imputer"]
    km      = km_bundle["model"]
    imp_km  = km_bundle["imputer"]
    reg_features = reg_bundle.get("feature_names", feature_names)

    # Encode new data using saved encoders
    X_new, _ = encode_for_model(df_new.copy(), fit=False, encoder_path=ENCODER_PATH)
    X_new = _align_features(X_new, feature_names)

    # Classification
    X_clf = imp_clf.transform(X_new)
    adopt_prob = clf.predict_proba(X_clf)[:, 1]
    adopt_pred = (adopt_prob >= 0.5).astype(int)

    # Regression
    X_reg = _align_features(X_new.copy(), reg_features)
    X_reg_imp = imp_reg.transform(X_reg)
    wtp_pred = np.clip(reg.predict(X_reg_imp), 0, 2500)

    # Clustering
    X_km_imp = imp_km.transform(X_new)
    cluster_labels = km.predict(X_km_imp)

    # Urgency score
    urgency_map = {
        "Within 3 months": 1, "3-6 months": 2, "6-12 months": 3,
        "1-2 years": 4, "More than 2 years": 5,
    }
    urgency = df_new.get("Q14_decision_urgency", pd.Series(["6-12 months"] * len(df_new)))
    urgency_scores = urgency.map(urgency_map).fillna(3).values

    # Build output
    results = df_new.copy()
    results["pred_adoption_probability"] = np.round(adopt_prob, 3)
    results["pred_will_adopt"]           = adopt_pred
    results["pred_wtp_monthly_inr"]      = np.round(wtp_pred, 0).astype(int)
    results["pred_wtp_tier"]             = [_wtp_tier(w) for w in wtp_pred]
    results["pred_cluster"]              = cluster_labels
    results["pred_priority_tier"]        = [
        _priority_tier(p, w) for p, w in zip(adopt_prob, wtp_pred)
    ]
    results["pred_marketing_action"]     = [
        _marketing_action(pt, w, u)
        for pt, w, u in zip(
            results["pred_priority_tier"], wtp_pred, urgency_scores
        )
    ]

    return results


def score_summary(scored_df: pd.DataFrame) -> dict:
    """Return summary stats from a scored dataframe."""
    pt = scored_df["pred_priority_tier"].value_counts().to_dict()
    return {
        "total": len(scored_df),
        "hot_leads": pt.get("Hot Lead", 0),
        "freemium": pt.get("Freemium Convert", 0),
        "reengage": pt.get("Re-engage", 0),
        "nurture": pt.get("Nurture", 0),
        "avg_adoption_prob": round(scored_df["pred_adoption_probability"].mean(), 3),
        "avg_predicted_wtp": round(scored_df["pred_wtp_monthly_inr"].mean(), 1),
        "priority_distribution": pt,
    }

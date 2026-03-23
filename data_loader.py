"""
data_loader.py
Handles loading, cleaning, encoding, and feature engineering
for both the training dataset and any newly uploaded CSV files.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import joblib
import os

# ── Column definitions ────────────────────────────────────────────────────────

ORDINAL_COLS = {
    "Q1_age": ["Under 15", "15-17", "18-20", "21-23", "24 or above"],
    "Q6_edu_level": ["Class 9 or below", "Class 10", "Class 11", "Class 12",
                     "UG 1st year", "UG 2nd year+", "PG / Other"],
    "Q9_academic_perf": ["Below 60%", "60-70%", "70-80%", "80-90%", "Above 90%", "Not applicable"],
    "Q10_career_clarity": ["Very confused", "Somewhat confused", "Just exploring",
                           "Mostly clear", "Completely clear"],
    "Q13_guidance_satisfaction": ["1 - Very dissatisfied", "2 - Dissatisfied",
                                  "3 - Neutral", "4 - Satisfied", "5 - Very satisfied", "NA"],
    "Q14_decision_urgency": ["Within 3 months", "3-6 months", "6-12 months",
                             "1-2 years", "More than 2 years"],
    "Q28_monthly_wtp": ["Nothing - free only", "Up to Rs99/mo", "Rs100-299/mo",
                        "Rs300-499/mo", "Rs500-999/mo", "Above Rs1000/mo"],
    "Q29_onetime_wtp": ["No - would not pay", "Yes up to Rs199", "Yes up to Rs499",
                        "Yes up to Rs999", "Yes above Rs999"],
}

NOMINAL_COLS = [
    "Q2_gender", "Q3_state", "Q4_location", "Q5_income",
    "Q7_stream", "Q8_board", "Q12_past_guidance",
    "Q20_who_pays", "Q27_tradeoff_wtp", "Q30_primary_source",
]

MULTI_SELECT_COLS = [
    "Q11_challenges", "Q15_subject_interests", "Q16_career_domain",
    "Q17_learning_style", "Q18_top_skills", "Q19_influencers",
    "Q21_career_factors", "Q22_past_spending", "Q23_info_sources",
    "Q26_feature_prefs",
]

NUMERIC_COLS = [
    "wtp_monthly_numeric", "urgency_score", "income_numeric",
    "clarity_score", "psycho_composite",
    "Q25_psych_fear_wrong_choice", "Q25_psych_prefer_independent",
    "Q25_psych_financial_over_passion", "Q25_psych_risk_tolerance",
    "Q25_psych_long_term_thinking",
]

TARGET_COL = "Q31_platform_adoption"
TARGET_ORDER = [
    "Definitely would NOT use",
    "Unlikely to use",
    "Neutral",
    "Likely would use",
    "Definitely would use",
]

WTP_NUM_MAP = {
    "Nothing - free only": 0,
    "Up to Rs99/mo": 70,
    "Rs100-299/mo": 200,
    "Rs300-499/mo": 400,
    "Rs500-999/mo": 750,
    "Above Rs1000/mo": 1200,
}

ENCODER_PATH = "encoders.pkl"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and do basic cleaning."""
    df = pd.read_csv(path, low_memory=False)
    df = df.replace("nan", np.nan)
    return df


def expand_multi_select(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-hot encode a pipe-separated multi-select column."""
    if col not in df.columns:
        return df
    dummies = df[col].fillna("").str.get_dummies(sep="|")
    dummies.columns = [f"{col}__{c.strip().replace(' ', '_').replace('/', '_')}"
                       for c in dummies.columns]
    if "" in dummies.columns or f"{col}__" in dummies.columns:
        dummies = dummies.drop(columns=[c for c in dummies.columns if c.endswith("__")], errors="ignore")
    return pd.concat([df, dummies], axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive extra numeric features."""
    df = df.copy()

    # WTP numeric from monthly WTP string if missing
    if "wtp_monthly_numeric" not in df.columns or df["wtp_monthly_numeric"].isna().all():
        df["wtp_monthly_numeric"] = df["Q28_monthly_wtp"].map(WTP_NUM_MAP).fillna(0)

    # Urgency numeric
    urgency_map = {
        "Within 3 months": 1, "3-6 months": 2, "6-12 months": 3,
        "1-2 years": 4, "More than 2 years": 5,
    }
    if "urgency_score" not in df.columns or df["urgency_score"].isna().all():
        df["urgency_score"] = df["Q14_decision_urgency"].map(urgency_map).fillna(3)

    # Clarity score
    clarity_map = {
        "Very confused": 1, "Somewhat confused": 2, "Just exploring": 3,
        "Mostly clear": 4, "Completely clear": 5,
    }
    if "clarity_score" not in df.columns or df["clarity_score"].isna().all():
        df["clarity_score"] = df["Q10_career_clarity"].map(clarity_map).fillna(3)

    # High intent flag
    df["high_intent"] = df[TARGET_COL].isin(
        ["Definitely would use", "Likely would use"]
    ).astype(int) if TARGET_COL in df.columns else 0

    # Past payer flag
    if "Q22_past_spending" in df.columns:
        df["past_payer"] = (~df["Q22_past_spending"].str.contains(
            "Nothing", na=True)).astype(int)
    else:
        df["past_payer"] = 0

    # Urban flag
    if "Q4_location" in df.columns:
        df["urban_flag"] = df["Q4_location"].isin(["Metro city", "Tier-2 city"]).astype(int)
    else:
        df["urban_flag"] = 0

    return df


def encode_for_model(df: pd.DataFrame, fit: bool = True,
                     encoder_path: str = ENCODER_PATH):
    """
    Encode all columns for ML.
    fit=True  → fit new encoders on training data, save to disk.
    fit=False → load saved encoders, transform new data.
    Returns (X_encoded DataFrame, feature_names list)
    """
    df = engineer_features(df)

    # Expand multi-select columns
    for col in MULTI_SELECT_COLS:
        df = expand_multi_select(df, col)

    encoders = {}
    encoded_frames = []

    # Ordinal encoding
    for col, order in ORDINAL_COLS.items():
        if col not in df.columns:
            continue
        ser = df[col].fillna(order[len(order) // 2])
        if fit:
            enc = OrdinalEncoder(categories=[order],
                                 handle_unknown="use_encoded_value",
                                 unknown_value=-1)
            vals = enc.fit_transform(ser.values.reshape(-1, 1))
            encoders[f"ord_{col}"] = enc
        else:
            saved = _load_encoders(encoder_path)
            enc = saved.get(f"ord_{col}")
            if enc is None:
                vals = np.zeros((len(df), 1))
            else:
                vals = enc.transform(ser.values.reshape(-1, 1))
        encoded_frames.append(
            pd.DataFrame(vals, columns=[f"{col}_enc"], index=df.index)
        )

    # Label encoding for nominals
    for col in NOMINAL_COLS:
        if col not in df.columns:
            continue
        ser = df[col].fillna("Unknown").astype(str)
        if fit:
            enc = LabelEncoder()
            vals = enc.fit_transform(ser)
            encoders[f"lbl_{col}"] = enc
        else:
            saved = _load_encoders(encoder_path)
            enc = saved.get(f"lbl_{col}")
            if enc is None:
                vals = np.zeros(len(df))
            else:
                known = set(enc.classes_)
                ser = ser.apply(lambda x: x if x in known else enc.classes_[0])
                vals = enc.transform(ser)
        encoded_frames.append(
            pd.DataFrame(vals, columns=[f"{col}_enc"], index=df.index)
        )

    # Numeric columns
    num_cols_present = [c for c in NUMERIC_COLS + ["urgency_score", "clarity_score",
                                                    "wtp_monthly_numeric", "past_payer",
                                                    "urban_flag", "high_intent"]
                        if c in df.columns]
    encoded_frames.append(df[num_cols_present].fillna(0))

    # Multi-select dummies
    dummy_cols = [c for c in df.columns if any(c.startswith(f"{m}__") for m in MULTI_SELECT_COLS)]
    if dummy_cols:
        encoded_frames.append(df[dummy_cols].fillna(0))

    X = pd.concat(encoded_frames, axis=1)

    # Remove target and ID cols from features
    drop_cols = [TARGET_COL, "respondent_id", "persona_label",
                 "Q24_missing_gap", "wtp_monthly_numeric"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    X = X.loc[:, ~X.columns.duplicated()]

    if fit and encoders:
        # Merge with any existing saved encoders
        try:
            existing = joblib.load(encoder_path)
            existing.update(encoders)
            joblib.dump(existing, encoder_path)
        except Exception:
            joblib.dump(encoders, encoder_path)

    return X, list(X.columns)


def _load_encoders(path: str) -> dict:
    if os.path.exists(path):
        return joblib.load(path)
    return {}


def prepare_target_classification(df: pd.DataFrame) -> pd.Series:
    """Binary target: 1 = likely/definitely use, 0 = otherwise."""
    mapping = {
        "Definitely would use": 1,
        "Likely would use": 1,
        "Neutral": 0,
        "Unlikely to use": 0,
        "Definitely would NOT use": 0,
    }
    return df[TARGET_COL].map(mapping).fillna(0).astype(int)


def prepare_target_regression(df: pd.DataFrame) -> pd.Series:
    """Continuous WTP in rupees."""
    if "wtp_monthly_numeric" in df.columns:
        return df["wtp_monthly_numeric"].fillna(0).astype(float)
    return df["Q28_monthly_wtp"].map(WTP_NUM_MAP).fillna(0).astype(float)


def get_arm_transactions(df: pd.DataFrame) -> list:
    """
    Build transaction list for Apriori ARM from multi-select columns.
    Each row becomes a set of items like 'interest:Mathematics' etc.
    """
    transactions = []
    col_prefix_map = {
        "Q15_subject_interests": "subject",
        "Q16_career_domain": "career",
        "Q17_learning_style": "learn",
        "Q18_top_skills": "skill",
        "Q22_past_spending": "spend",
        "Q26_feature_prefs": "feature",
        "Q11_challenges": "challenge",
    }
    for _, row in df.iterrows():
        items = []
        for col, prefix in col_prefix_map.items():
            if col in df.columns and pd.notna(row.get(col)):
                for val in str(row[col]).split("|"):
                    val = val.strip()
                    if val:
                        items.append(f"{prefix}:{val[:30]}")
        # Add key categorical items
        for col, prefix in [("Q7_stream", "stream"), ("Q4_location", "loc"),
                             ("Q10_career_clarity", "clarity"),
                             ("Q28_monthly_wtp", "wtp")]:
            if col in df.columns and pd.notna(row.get(col)):
                items.append(f"{prefix}:{str(row[col])[:25]}")
        if items:
            transactions.append(items)
    return transactions

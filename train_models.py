"""
train_models.py
Trains and persists all ML models:
  - Random Forest classifier  (platform adoption)
  - Gradient Boosting regressor (WTP prediction)
  - K-Means clustering         (persona discovery)
  - Apriori ARM                (interest-career associations)

Run once: python train_models.py
Models saved as .pkl files, loaded by the Streamlit app.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

from data_loader import (
    load_data, encode_for_model, prepare_target_classification,
    prepare_target_regression, get_arm_transactions, ENCODER_PATH
)

DATA_PATH   = "career_survey_dataset.csv"
CLF_PATH    = "model_classifier.pkl"
REG_PATH    = "model_regressor.pkl"
KM_PATH     = "model_kmeans.pkl"
ARM_PATH    = "arm_rules.pkl"
METRICS_PATH = "model_metrics.json"
FEATURE_PATH = "feature_names.pkl"


# ── Apriori implementation (no mlxtend dependency) ──────────────────────────

def _get_freq_itemsets(transactions, min_support):
    """Simple Apriori — finds frequent 1- and 2-itemsets."""
    from collections import defaultdict
    n = len(transactions)
    trans_sets = [set(t) for t in transactions]

    # 1-itemsets
    counts1 = defaultdict(int)
    for t in trans_sets:
        for item in t:
            counts1[frozenset([item])] += 1

    freq1 = {k: v / n for k, v in counts1.items() if v / n >= min_support}

    # 2-itemsets
    items = list({list(k)[0] for k in freq1})
    counts2 = defaultdict(int)
    for t in trans_sets:
        t_items = [i for i in items if i in t]
        for i in range(len(t_items)):
            for j in range(i + 1, len(t_items)):
                pair = frozenset([t_items[i], t_items[j]])
                counts2[pair] += 1

    freq2 = {k: v / n for k, v in counts2.items() if v / n >= min_support}
    return {**freq1, **freq2}, n, trans_sets


def run_apriori(transactions, min_support=0.05, min_confidence=0.3, min_lift=1.0):
    """Generate association rules with support, confidence, and lift."""
    freq_sets, n, trans_sets = _get_freq_itemsets(transactions, min_support)

    rules = []
    # Generate rules from 2-itemsets
    for itemset, support in freq_sets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for i, antecedent in enumerate(items):
            consequent = items[1 - i]
            ant_set = frozenset([antecedent])
            con_set = frozenset([consequent])

            ant_support = freq_sets.get(ant_set, 0)
            con_support = freq_sets.get(con_set, 0)

            if ant_support == 0 or con_support == 0:
                continue

            confidence = support / ant_support
            lift = confidence / con_support

            if confidence >= min_confidence and lift >= min_lift:
                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                })

    rules_df = pd.DataFrame(rules).drop_duplicates()
    if not rules_df.empty:
        rules_df = rules_df.sort_values("lift", ascending=False)
    return rules_df


# ── Main training pipeline ────────────────────────────────────────────────────

def train_all(data_path: str = DATA_PATH):
    print(f"\n{'='*60}")
    print("  Career Platform — Model Training Pipeline")
    print(f"{'='*60}\n")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Place the CSV in the same folder.")
        sys.exit(1)

    df = load_data(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Feature encoding ──────────────────────────────────────────────────
    print("\n[1/5] Encoding features...")
    X, feature_names = encode_for_model(df, fit=True, encoder_path=ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    print(f"  Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    # ── Classification ────────────────────────────────────────────────────
    print("\n[2/5] Training Random Forest classifier (Q31 adoption)...")
    y_clf = prepare_target_classification(df)
    print(f"  Class distribution — Positive (will use): {y_clf.sum()} | Negative: {(y_clf == 0).sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    imputer_clf = SimpleImputer(strategy="median")
    X_tr_imp = imputer_clf.fit_transform(X_tr)
    X_te_imp = imputer_clf.transform(X_te)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_tr_imp, y_tr)
    joblib.dump({"model": clf, "imputer": imputer_clf}, CLF_PATH)

    y_pred = clf.predict(X_te_imp)
    y_prob = clf.predict_proba(X_te_imp)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob)
    cv   = cross_val_score(clf, imputer_clf.transform(X), y_clf, cv=5,
                           scoring="f1", n_jobs=-1)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV F1 (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

    # Feature importance
    feat_imp = pd.Series(clf.feature_importances_, index=feature_names)
    top_features = feat_imp.nlargest(20).to_dict()

    # ROC curve data
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_te, y_prob)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(), "auc": auc}

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred).tolist()
    clf_report = classification_report(y_te, y_pred, output_dict=True)

    # ── Regression ────────────────────────────────────────────────────────
    print("\n[3/5] Training Gradient Boosting regressor (WTP prediction)...")

    # Exclude wtp_monthly_numeric from features for regression (data leakage)
    X_reg = X.drop(columns=["wtp_monthly_numeric"], errors="ignore")
    y_reg = prepare_target_regression(df)

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    imputer_reg = SimpleImputer(strategy="median")
    X_tr_r_imp = imputer_reg.fit_transform(X_tr_r)
    X_te_r_imp = imputer_reg.transform(X_te_r)

    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    reg.fit(X_tr_r_imp, y_tr_r)
    joblib.dump({"model": reg, "imputer": imputer_reg,
                 "feature_names": list(X_reg.columns)}, REG_PATH)

    y_pred_r = reg.predict(X_te_r_imp)
    mae = mean_absolute_error(y_te_r, y_pred_r)
    r2  = r2_score(y_te_r, y_pred_r)

    print(f"  MAE  : ₹{mae:.1f}")
    print(f"  R²   : {r2:.4f}")

    reg_feat_imp = pd.Series(reg.feature_importances_,
                             index=list(X_reg.columns)).nlargest(20).to_dict()

    # ── Clustering ────────────────────────────────────────────────────────
    print("\n[4/5] Training K-Means clustering (persona discovery)...")
    imputer_km = SimpleImputer(strategy="median")
    X_km = imputer_km.fit_transform(X)

    # Elbow method: test k=2..8
    inertias = []
    sil_scores = []
    from sklearn.metrics import silhouette_score
    for k in range(2, 9):
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_km)
        inertias.append(km_temp.inertia_)
        sil_scores.append(silhouette_score(X_km, km_temp.labels_,
                                           sample_size=500, random_state=42))

    best_k = int(np.argmax(sil_scores)) + 2  # offset by 2
    print(f"  Best k by silhouette: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km.fit(X_km)
    joblib.dump({"model": km, "imputer": imputer_km,
                 "inertias": inertias, "sil_scores": sil_scores,
                 "best_k": best_k}, KM_PATH)

    df["cluster"] = km.labels_
    cluster_wtp = df.groupby("cluster")["wtp_monthly_numeric"].mean().to_dict()
    cluster_sizes = df["cluster"].value_counts().to_dict()
    print(f"  Cluster sizes: {cluster_sizes}")

    # ── Association Rule Mining ───────────────────────────────────────────
    print("\n[5/5] Running Association Rule Mining (Apriori)...")
    transactions = get_arm_transactions(df)
    print(f"  Transactions: {len(transactions)}")

    rules_df = run_apriori(transactions,
                           min_support=0.04,
                           min_confidence=0.25,
                           min_lift=1.1)
    print(f"  Rules generated: {len(rules_df)}")
    if not rules_df.empty:
        print(f"  Top rule: {rules_df.iloc[0]['antecedent']} → "
              f"{rules_df.iloc[0]['consequent']} "
              f"(lift={rules_df.iloc[0]['lift']:.2f})")
    joblib.dump(rules_df, ARM_PATH)

    # ── Save all metrics ──────────────────────────────────────────────────
    metrics = {
        "classifier": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_f1_mean": round(float(cv.mean()), 4),
            "cv_f1_std": round(float(cv.std()), 4),
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "roc_curve": roc_data,
            "top_features": {k: round(float(v), 6) for k, v in top_features.items()},
        },
        "regressor": {
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "top_features": {k: round(float(v), 6) for k, v in reg_feat_imp.items()},
        },
        "clustering": {
            "best_k": best_k,
            "inertias": [round(x, 1) for x in inertias],
            "silhouette_scores": [round(x, 4) for x in sil_scores],
            "cluster_wtp": {str(k): round(float(v), 1)
                            for k, v in cluster_wtp.items()},
            "cluster_sizes": {str(k): int(v)
                              for k, v in cluster_sizes.items()},
        },
        "arm": {
            "total_rules": len(rules_df),
            "min_support": 0.04,
            "min_confidence": 0.25,
            "min_lift": 1.1,
        },
        "training_rows": len(df),
        "feature_count": X.shape[1],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("  All models trained and saved successfully.")
    print(f"  Files: {CLF_PATH}, {REG_PATH}, {KM_PATH}, {ARM_PATH}")
    print(f"  Metrics: {METRICS_PATH}")
    print(f"{'='*60}\n")
    return metrics


if __name__ == "__main__":
    train_all()

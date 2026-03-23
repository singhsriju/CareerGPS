# 🎯 CareerPath Analytics — AI Career Platform Dashboard

A data-driven analytics system for an AI-powered student career guidance platform.
Built with Python, scikit-learn, and Streamlit.

## Features

- **Descriptive Analysis** — demographics, WTP distributions, geographic spread, cross-tabs
- **Diagnostic Analysis** — Apriori ARM (support, confidence, lift), K-Means clustering, correlation matrix, psychographic radar
- **Predictive Analysis** — Random Forest classifier (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importance), Gradient Boosting regressor (MAE, R²)
- **Prescriptive Analysis** — Segment playbooks, pricing tiers, channel strategy
- **New Customer Prediction** — Upload any new survey CSV → get persona, adoption probability, WTP prediction, priority tier and marketing action per row

## Quick Start (Local)

```bash
# 1. Clone or download the repo
git clone <your-repo-url>
cd careerpath-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the same folder
# File: career_survey_dataset.csv

# 4. Train models (run once)
python train_models.py

# 5. Launch the app
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push all files to a **public GitHub repo** (flat structure, no sub-folders except `.streamlit/`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

> **Important:** The `career_survey_dataset.csv` must be committed to the repo.  
> After first deploy, click **⚙️ Train / Retrain Models** in the sidebar to train models on the cloud instance.

## File Structure

```
app.py                        ← Main Streamlit app
train_models.py               ← Model training pipeline (run once)
data_loader.py                ← Data preprocessing + feature engineering
predictor.py                  ← New customer scoring engine
charts.py                     ← All Plotly chart functions
requirements.txt              ← Python dependencies
career_survey_dataset.csv     ← Training dataset (2,000 rows)
.streamlit/config.toml        ← Streamlit theme config
README.md                     ← This file
```

## ML Models Used

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| Classifier | Predict platform adoption (Q31) | Random Forest (200 trees) |
| Regressor | Predict monthly WTP in ₹ | Gradient Boosting |
| Clustering | Discover student personas | K-Means (elbow + silhouette) |
| ARM | Find interest-career associations | Apriori (custom implementation) |

## Analysis Layers

| Layer | Question answered |
|-------|-------------------|
| Descriptive | Who is in our market? |
| Diagnostic | Why do students behave this way? |
| Predictive | Which new students will convert? |
| Prescriptive | What should we do for each segment? |

## New Customer Prediction

1. Navigate to **📤 New Customer Prediction**
2. Download the CSV template
3. Fill in your new respondent data (same column format)
4. Upload and get instant scores:
   - `pred_adoption_probability` (0–1)
   - `pred_wtp_monthly_inr` (₹ amount)
   - `pred_wtp_tier` (pricing tier label)
   - `pred_priority_tier` (Hot Lead / Freemium Convert / Re-engage / Nurture)
   - `pred_marketing_action` (specific recommended action)
5. Download the fully scored CSV

## Built By

Sriju — AI Career Platform Founder  
Analytics system designed for data-driven go-to-market strategy.

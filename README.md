# EdTech Readiness Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://edtech-readiness-dashboard-z7wjgkqhnfze6drwcvbsw6.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![VADER](https://img.shields.io/badge/VADER-NLP%20Sentiment-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **Can online education replace face-to-face learning in India?**  
> A full-stack data analytics project — from primary survey collection to an interactive ML-powered dashboard.

 **[Live Dashboard →](https://edtech-readiness-dashboard-z7wjgkqhnfze6drwcvbsw6.streamlit.app/)**

---

##  Project Overview

This project investigates student perceptions of online education across demographic groups in Tamil Nadu, India — conducted against the backdrop of India's **NEP 2020 hybrid learning mandate** and a **₹7L crore EdTech industry**.

**Primary data** was collected via structured survey (Google Forms) with **543 student responses** across 16 variables including demographics, internet access patterns, and 9 Likert-scale attitudinal questions.

The project goes beyond simple EDA — it builds an end-to-end analytics pipeline including multi-model ML classification, SHAP explainability, and VADER sentiment analysis on structured survey text.

---

##  Key Findings

| Finding | Value |
|---|---|
| Students preferring face-to-face learning | **57.8%** |
| VADER sentiment correlation with Likert scores | **r = 0.808** |
| Best model (Gradient Boosting) ROC-AUC | **0.998** |
| Random Forest cross-validated F1 | **0.898 ± 0.029** |
| Top internet barrier (cost) | **33% of students** |
| Internet access gap in perception score | **+0.21 pts** |

**Most important SHAP features:** Belief that a full course can be delivered online, and ability to discuss with peers online — not demographics.

---

##  Project Structure

```
edtech-readiness-dashboard/
│
├── appedtech.py          # Main Streamlit dashboard (4 pages)
├── duplicate.xlsx        # Primary survey data (543 responses)
├── requirements.txt      # Python dependencies
└── README.md
```

---

##  Technical Stack

### Data & ML
- **pandas / numpy** — data wrangling and feature engineering
- **scikit-learn** — Random Forest, Gradient Boosting, Logistic Regression
- **SHAP** — model explainability (TreeExplainer)
- **StratifiedKFold** — cross-validation with class balance

### NLP & Sentiment
- **VADER Sentiment** — compound scoring on structured Likert text
- **NLTK** — tokenization and stopword filtering
- **WordCloud** — response pattern visualisation

### Dashboard
- **Streamlit** — multi-page interactive app
- **Plotly Express** — interactive charts (bar, scatter, pie, heatmap, gauge)
- **Matplotlib / Seaborn** — static visualisation layer

---

##  Dashboard Pages

| Page | What it shows |
|---|---|
| 🏠 **Overview** | KPI cards, F2F preference by age group, internet distribution, key findings |
| 📊 **EDA Explorer** | Mean Likert scores by group, correlation heatmap, internet access analysis |
| 🤖 **Live Predictor** | Input student profile → predict F2F preference + SHAP importance |
| 💬 **NLP Insights** | VADER sentiment distribution, Likert-sentiment scatter (r=0.808), word clouds, barrier analysis |

All pages respond to **global demographic filters** (gender, age, education) in the sidebar.

---

##  Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/dhanushc13/edtech-readiness-dashboard.git
cd edtech-readiness-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Launch the dashboard
streamlit run appedtech.py
```

---

##  Methodology Notes

**Target variable:** `q8_need_face_to_face >= 4` (binary) — predicts which students strongly prefer physical classroom support. Q8 is excluded from features to prevent data leakage.

**Why tree models outperform Logistic Regression (F1: 0.537):** Likert-scale attitudes interact non-linearly. Students who score high on peer discussion *and* time management show disproportionately lower F2F preference — a pattern trees capture, linear models cannot.

**On VADER sentiment (r = 0.808):** VADER detects the emotional valence of phrases like "strongly agree" vs "strongly disagree" — even in structured survey labels. The high correlation validates that Likert responses carry measurable sentiment signal beyond their ordinal value.

---

##  Data

Primary data collected via Google Forms survey, Tamil Nadu, India (2023). Data has been anonymised — names are not used in analysis. 543 complete responses across 16 columns.

---

## 👤 Author

**Dhanush C**  
Aspiring Data Analyst | Tamil Nadu, India  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/dhanush-c-a0611024a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/dhanushc13)

---

## 📄 License

MIT License — free to use with attribution.

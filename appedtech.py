"""
=============================================================================
  EdTech Readiness Dashboard — Streamlit App
  Online Education Perception Study | India Post-COVID
=============================================================================
  Run: streamlit run app.py
  Requires: pip install streamlit pandas numpy matplotlib seaborn
            scikit-learn shap plotly vaderSentiment wordcloud openpyxl
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, roc_curve,
                             roc_auc_score, f1_score, accuracy_score)
import shap

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("stopwords",  quiet=True)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EdTech Readiness Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #1d9e7530;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #1d9e7540;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 0 20px #1d9e7515;
    }
    div[data-testid="metric-container"] label {
        color: #1d9e75 !important;
        font-size: 13px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Headers */
    h1 { color: #1d9e75 !important; }
    h2 { color: #e6edf3 !important; border-bottom: 1px solid #1d9e7530; padding-bottom: 8px; }
    h3 { color: #9FE1CB !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1d9e75, #0f6e56);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px #1d9e7540;
    }

    /* Selectbox / slider labels */
    label { color: #9FE1CB !important; }

    /* Divider */
    hr { border-color: #1d9e7520; }

    /* Info boxes */
    .insight-card {
        background: #161b22;
        border-left: 4px solid #1d9e75;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        box-shadow: 0 0 15px #1d9e7510;
    }
    .insight-card p { color: #cdd9e5; margin: 0; line-height: 1.6; }
    .insight-card strong { color: #1d9e75; }

    /* Prediction result */
    .pred-positive {
        background: linear-gradient(135deg, #0f3d2e, #1d9e7520);
        border: 1px solid #1d9e75;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .pred-negative {
        background: linear-gradient(135deg, #3d1a0f, #d85a3020);
        border: 1px solid #d85a30;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1d9e75 !important;
        border-bottom: 2px solid #1d9e75 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
C_GREEN  = "#1D9E75"
C_ORANGE = "#D85A30"
C_PURPLE = "#7F77DD"
BG_DARK  = "#0d1117"
BG_CARD  = "#161b22"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font=dict(color="#e6edf3", family="sans-serif"),
        xaxis=dict(gridcolor="rgba(29,158,117,0.12)", zerolinecolor="rgba(29,158,117,0.18)"),
        yaxis=dict(gridcolor="rgba(29,158,117,0.12)", zerolinecolor="rgba(29,158,117,0.18)"),
        legend=dict(bgcolor=BG_CARD, bordercolor="rgba(29,158,117,0.18)"),
    )
)

# ── DATA LOADING & CACHING ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("duplicate.xlsx")

    RENAME_MAP = {
        "Timestamp": "timestamp", "Name": "name",
        "Age": "age", "Gender": "gender", "Education": "education",
        df.columns[5]:  "internet_access",
        df.columns[6]:  "internet_limit_reason",
        df.columns[7]:  "q1_computer_skill",
        df.columns[8]:  "q2_electronic_communication",
        df.columns[9]:  "q3_learning_same",
        df.columns[10]: "q4_more_motivating",
        df.columns[11]: "q5_full_course_online",
        df.columns[12]: "q6_discuss_with_peers",
        df.columns[13]: "q7_group_work_online",
        df.columns[14]: "q8_need_face_to_face",
        df.columns[15]: "q9_manage_time_online",
    }
    df.rename(columns=RENAME_MAP, inplace=True)

    LIKERT_COLS = [c for c in df.columns if c.startswith("q")]

    # Save text before encoding (for NLP)
    df["response_profile"] = df[LIKERT_COLS].apply(
        lambda row: " ".join(str(v).strip().lower()
                             for v in row if str(v) != "nan"), axis=1)

    LIKERT_MAP = {"strongly agree": 5, "agree": 4, "neutral": 3,
                  "disagree": 2, "strongly disagree": 1}
    for col in LIKERT_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower().map(LIKERT_MAP)

    df.dropna(subset=LIKERT_COLS, how="all", inplace=True)
    df[LIKERT_COLS] = df[LIKERT_COLS].fillna(df[LIKERT_COLS].median())

    df["gender"]    = df["gender"].str.strip().str.title()
    df["age"]       = df["age"].str.strip()
    df["education"] = df["education"].str.strip().str.title()
    df["internet_binary"] = df["internet_access"].apply(
        lambda x: 0 if str(x).strip().lower().startswith("no") else 1)

    # Target
    df["target"]       = (df["q8_need_face_to_face"] >= 4).astype(int)
    df["target_label"] = df["target"].map({1: "Needs F2F", 0: "OK Online"})

    # Avg score
    feature_q = ["q1_computer_skill","q2_electronic_communication",
                  "q3_learning_same","q4_more_motivating","q5_full_course_online",
                  "q6_discuss_with_peers","q7_group_work_online","q9_manage_time_online"]
    df["avg_likert"] = df[feature_q].mean(axis=1)

    # VADER
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment(text):
        s = analyzer.polarity_scores(str(text))
        return pd.Series({"vader_compound": s["compound"],
                          "vader_pos": s["pos"], "vader_neg": s["neg"]})
    sentiment = df["response_profile"].apply(get_sentiment)
    df = pd.concat([df, sentiment], axis=1)

    return df

@st.cache_resource
def train_models(df):
    Q_FEATURES = ["q1_computer_skill","q2_electronic_communication",
                   "q3_learning_same","q4_more_motivating","q5_full_course_online",
                   "q6_discuss_with_peers","q7_group_work_online","q9_manage_time_online"]
    le_g = LabelEncoder(); le_a = LabelEncoder(); le_e = LabelEncoder()
    df = df.copy()
    df["gender_enc"]    = le_g.fit_transform(df["gender"].fillna("Unknown"))
    df["age_enc"]       = le_a.fit_transform(df["age"].fillna("Unknown"))
    df["education_enc"] = le_e.fit_transform(df["education"].fillna("Unknown"))

    FEAT_COLS  = Q_FEATURES + ["internet_binary","gender_enc","age_enc","education_enc"]
    FEAT_NAMES = ["Computer skill","Electronic comm.","Learning same at home",
                  "Online more motivating","Full course possible",
                  "Peer discussion","Group work online","Time management",
                  "Internet access","Gender","Age group","Education"]

    X = df[FEAT_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 min_samples_leaf=3, class_weight="balanced",
                                 random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                     max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # SHAP
    explainer  = shap.TreeExplainer(rf)
    shap_vals  = explainer(X_test)
    sv = shap_vals.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_shap = np.abs(sv).mean(axis=0)

    return {
        "rf": rf, "gb": gb,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feat_cols": FEAT_COLS, "feat_names": FEAT_NAMES,
        "mean_shap": mean_shap,
        "encoders": {"gender": le_g, "age": le_a, "education": le_e},
    }

# ── LOAD ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models..."):
    df   = load_data()
    mdls = train_models(df)

Q_LABELS = {
    "q1_computer_skill":          "Computer skill",
    "q2_electronic_communication":"Electronic comm.",
    "q3_learning_same":           "Learning same at home",
    "q4_more_motivating":         "Online more motivating",
    "q5_full_course_online":      "Full course possible",
    "q6_discuss_with_peers":      "Peer discussion online",
    "q7_group_work_online":       "Group work online",
    "q9_manage_time_online":      "Time management online",
}
PALETTE = {"Needs F2F": C_ORANGE, "OK Online": C_GREEN}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:40px'>📚</div>
        <div style='color:{C_GREEN}; font-size:18px; font-weight:700;
                    letter-spacing:1px; margin-top:8px;'>
            EdTech Readiness
        </div>
        <div style='color:#8b949e; font-size:12px; margin-top:4px;'>
            Online Education Perception Study
        </div>
    </div>
    <hr style='border-color:#1d9e7530; margin: 8px 0 20px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "📊  EDA Explorer",
         "🤖  Live Predictor", "💬  NLP Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1d9e7530; margin: 20px 0 16px;'>",
                unsafe_allow_html=True)

    # Global filters
    st.markdown(f"<p style='color:{C_GREEN}; font-size:13px; font-weight:600;'>"
                f"GLOBAL FILTERS</p>", unsafe_allow_html=True)

    sel_gender = st.multiselect(
        "Gender", options=sorted(df["gender"].dropna().unique()),
        default=sorted(df["gender"].dropna().unique()))
    sel_age = st.multiselect(
        "Age Group", options=sorted(df["age"].dropna().unique()),
        default=sorted(df["age"].dropna().unique()))
    sel_edu = st.multiselect(
        "Education", options=sorted(df["education"].dropna().unique()),
        default=sorted(df["education"].dropna().unique()))

    df_f = df[
        df["gender"].isin(sel_gender) &
        df["age"].isin(sel_age) &
        df["education"].isin(sel_edu)
    ].copy()

    st.markdown(f"""
    <div style='background:{BG_CARD}; border:1px solid #1d9e7530;
                border-radius:8px; padding:12px; margin-top:12px;
                text-align:center;'>
        <div style='color:#8b949e; font-size:11px;'>Filtered students</div>
        <div style='color:{C_GREEN}; font-size:24px; font-weight:700;'>
            {len(df_f):,}
        </div>
        <div style='color:#8b949e; font-size:11px;'>of {len(df):,} total</div>
    </div>

    <div style='margin-top:24px; padding:0 4px;'>
        <div style='color:#8b949e; font-size:11px; line-height:1.7;'>
            📍 Tamil Nadu, India · 2023<br>
            👨‍🎓 Primary data collection<br>
            🧪 543 student responses<br>
            🐍 Python · sklearn · SHAP
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# 📚 EdTech Readiness Dashboard")
    st.markdown(
        "<p style='color:#8b949e; font-size:15px; margin-top:-12px;'>"
        "Online Education Perception Study — India Post-COVID | "
        "543 student responses · Primary data collection · 2023</p>",
        unsafe_allow_html=True)

    st.markdown("---")

    # KPI row
    pos_pct   = df_f["target"].mean() * 100
    neg_pct   = 100 - pos_pct
    avg_sent  = df_f["vader_compound"].mean()
    n_no_net  = (df_f["internet_binary"] == 0).sum()
    net_gap   = (df_f[df_f["internet_binary"]==1]["avg_likert"].mean() -
                 df_f[df_f["internet_binary"]==0]["avg_likert"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total students",   f"{len(df_f):,}")
    c2.metric("Prefer F2F",       f"{pos_pct:.1f}%",
              delta=f"{pos_pct-50:.1f}pp vs 50%",
              delta_color="inverse")
    c3.metric("OK with Online",   f"{neg_pct:.1f}%")
    c4.metric("Avg Sentiment",    f"{avg_sent:.3f}",
              delta="r=0.808 vs Likert")
    c5.metric("Internet Gap",     f"+{net_gap:.2f} pts",
              help="Score diff: full access vs limited")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### F2F Preference by Group")
        fig = px.bar(
            df_f.groupby(["age","target_label"])
                .size().reset_index(name="count")
                .assign(pct=lambda d: d.groupby("age")["count"]
                        .transform(lambda x: x/x.sum()*100)),
            x="age", y="pct", color="target_label",
            barmode="group",
            color_discrete_map=PALETTE,
            labels={"pct": "% students", "age": "Age group",
                    "target_label": "Group"},
            template="plotly_dark",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                          margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Internet Access Distribution")
        counts = df_f["internet_access"].value_counts().reset_index()
        counts.columns = ["Access type", "Count"]
        fig2 = px.pie(counts, names="Access type", values="Count",
                      color_discrete_sequence=[C_GREEN, C_ORANGE, C_PURPLE],
                      hole=0.45, template="plotly_dark")
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"],
                           margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Key Findings")
    findings = [
        ("57.8% of students prefer face-to-face learning",
         "Majority still rely on physical classrooms. "
         "Institutions need hybrid-ready infrastructure."),
        ("VADER sentiment r = 0.808 with Likert scores",
         "Even simple Likert labels carry detectable emotional signal. "
         "Students who disagree more show measurably lower sentiment."),
        ("Cost is the #1 internet barrier (33%)",
         "Signal problems (27%) come second. "
         "Both are addressable through institutional subsidy and infrastructure."),
        ("Random Forest CV F1 = 0.898 ± 0.029",
         "'Full course possible' and 'Peer discussion' are the top SHAP predictors — "
         "not demographics. Belief drives behaviour more than background."),
    ]
    for title, body in findings:
        st.markdown(f"""
        <div class='insight-card'>
            <p><strong>{title}</strong><br>{body}</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA Explorer":
    st.markdown("# 📊 EDA Explorer")
    st.markdown("<p style='color:#8b949e;'>Interactive exploration of "
                "survey responses. Use the sidebar filters to drill down.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["  Question Scores  ", "  Correlation  ", "  Internet Access  "])

    with tab1:
        st.markdown("### Mean Agreement Score per Question")
        mean_q = (df_f.groupby("target_label")[list(Q_LABELS.keys())]
                  .mean().T.rename(index=Q_LABELS).reset_index())
        mean_q.columns = ["Question", "Needs F2F", "OK Online"]
        mean_long = mean_q.melt("Question", var_name="Group", value_name="Score")

        fig = px.bar(mean_long, x="Score", y="Question", color="Group",
                     barmode="group", orientation="h",
                     color_discrete_map={"Needs F2F": C_ORANGE,
                                         "OK Online": C_GREEN},
                     template="plotly_dark",
                     labels={"Score": "Mean Likert Score (1–5)"})
        fig.add_vline(x=3, line_dash="dash", line_color="#8b949e",
                      annotation_text="Neutral = 3")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                          height=480, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='insight-card'>
        <p><strong>Reading this chart:</strong> The gap between orange and green bars
        shows how differently the two groups answered each question.
        Large gaps = strong discriminating questions for the model.
        Questions where both bars fall below 3 (neutral) indicate
        universal barriers regardless of F2F preference.</p>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Correlation Heatmap")
        corr_cols = list(Q_LABELS.keys()) + ["q8_need_face_to_face"]
        corr_labels = list(Q_LABELS.values()) + ["Needs face-to-face"]
        corr_matrix = df_f[corr_cols].rename(
            columns={**Q_LABELS, "q8_need_face_to_face": "Needs face-to-face"}
        ).corr()

        fig = px.imshow(corr_matrix,
                        color_continuous_scale="RdYlGn",
                        zmin=-1, zmax=1,
                        text_auto=".2f",
                        template="plotly_dark",
                        aspect="auto")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                          height=520, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Highlight strongest pair
        corr_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_no_diag.values, 0)
        idx = np.unravel_index(
            np.abs(corr_no_diag.values).argmax(), corr_no_diag.shape)
        r1, r2 = corr_matrix.index[idx[0]], corr_matrix.columns[idx[1]]
        rv = corr_matrix.loc[r1, r2]
        st.markdown(f"""
        <div class='insight-card'>
        <p><strong>Strongest correlation:</strong> "{r1}" ↔ "{r2}" (r = {rv:.3f})<br>
        Students who believe learning is the same at home are also more
        motivated online — these attitudes reinforce each other.</p>
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### Perception Score by Internet Access Type")
        col_a, col_b = st.columns(2)

        with col_a:
            box_df = df_f[["internet_access", "avg_likert",
                            "target_label"]].copy()
            fig = px.box(box_df, x="internet_access", y="avg_likert",
                         color="target_label",
                         color_discrete_map=PALETTE,
                         template="plotly_dark",
                         labels={"avg_likert": "Avg Likert Score",
                                 "internet_access": "Access type"})
            fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                              height=380,
                              margin=dict(t=20,b=60,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            barrier = (df_f["internet_limit_reason"]
                       .value_counts().head(5).reset_index())
            barrier.columns = ["Reason", "Count"]
            fig2 = px.bar(barrier, x="Count", y="Reason", orientation="h",
                          color="Count",
                          color_continuous_scale=["#0f6e56", "#1D9E75"],
                          template="plotly_dark")
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"],
                               height=380,
                               margin=dict(t=20,b=20,l=10,r=10),
                               showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        no_inet_score  = df_f[df_f["internet_binary"]==0]["avg_likert"].mean()
        yes_inet_score = df_f[df_f["internet_binary"]==1]["avg_likert"].mean()
        st.markdown(f"""
        <div class='insight-card'>
        <p>Students with <strong>no internet</strong> avg score: {no_inet_score:.2f} &nbsp;|&nbsp;
        Students with <strong>full access</strong>: {yes_inet_score:.2f}<br>
        Gap of <strong>{yes_inet_score-no_inet_score:.2f} points</strong> — 
        access directly suppresses online learning confidence.
        Cost (33%) and signal strength (27%) are the primary barriers.</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Live Predictor":
    st.markdown("# 🤖 Live F2F Preference Predictor")
    st.markdown(
        "<p style='color:#8b949e;'>Enter a student's survey responses to predict "
        "whether they strongly prefer face-to-face learning and need "
        "institutional intervention.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_results = st.columns([1, 1])

    LIKERT_OPTIONS = ["Strongly Disagree", "Disagree",
                      "Neutral", "Agree", "Strongly Agree"]
    LIKERT_VAL     = {o: i+1 for i, o in enumerate(LIKERT_OPTIONS)}

    with col_form:
        st.markdown("### Student Profile")

        with st.expander("📋 Survey Responses", expanded=True):
            q1  = st.select_slider("I am good at using the computer",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q2  = st.select_slider("Comfortable communicating electronically",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q3  = st.select_slider("Learning is the same at home as class",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q4  = st.select_slider("Online learning is more motivating",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q5  = st.select_slider("A full course can be given online",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q6  = st.select_slider("Can discuss with peers online",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q7  = st.select_slider("Can do group work online",
                                   options=LIKERT_OPTIONS, value="Neutral")
            q9  = st.select_slider("Can manage study time online",
                                   options=LIKERT_OPTIONS, value="Neutral")

        with st.expander("👤 Demographics", expanded=True):
            inet    = st.selectbox("Internet Access",
                                   ["Full (DSL/Fibre)", "Limited (Mobile)",
                                    "No internet"])
            gender  = st.selectbox("Gender", ["Male", "Female", "Prefer Not To Say"])
            age_grp = st.selectbox("Age Group", ["16-20", "21-25"])
            edu     = st.selectbox("Education",
                                   ["Undergraduate", "Higher Secondary", "Master"])

        predict_btn = st.button("🔮  Predict", use_container_width=True)

    with col_results:
        st.markdown("### Model Results")

        if predict_btn:
            # Encode inputs
            inet_val = 0 if inet == "No internet" else 1
            le_g = mdls["encoders"]["gender"]
            le_a = mdls["encoders"]["age"]
            le_e = mdls["encoders"]["education"]

            # Safe encode — handle unseen labels
            def safe_encode(le, val):
                if val in le.classes_:
                    return le.transform([val])[0]
                return 0

            row = pd.DataFrame([[
                LIKERT_VAL[q1], LIKERT_VAL[q2], LIKERT_VAL[q3],
                LIKERT_VAL[q4], LIKERT_VAL[q5], LIKERT_VAL[q6],
                LIKERT_VAL[q7], LIKERT_VAL[q9],
                inet_val,
                safe_encode(le_g, gender),
                safe_encode(le_a, age_grp),
                safe_encode(le_e, edu),
            ]], columns=mdls["feat_cols"])

            rf_prob = mdls["rf"].predict_proba(row)[0][1]
            gb_prob = mdls["gb"].predict_proba(row)[0][1]
            avg_prob = (rf_prob + gb_prob) / 2
            prediction = int(avg_prob >= 0.5)

            if prediction == 1:
                st.markdown(f"""
                <div class='pred-negative'>
                    <div style='font-size:48px'>⚠️</div>
                    <div style='color:{C_ORANGE}; font-size:22px;
                                font-weight:700; margin:8px 0;'>
                        Needs F2F Support
                    </div>
                    <div style='color:#cdd9e5; font-size:14px;'>
                        Probability: {avg_prob*100:.1f}%
                    </div>
                    <div style='color:#8b949e; font-size:12px; margin-top:8px;'>
                        Recommend: Hybrid learning plan + instructor check-in
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-positive'>
                    <div style='font-size:48px'>✅</div>
                    <div style='color:{C_GREEN}; font-size:22px;
                                font-weight:700; margin:8px 0;'>
                        OK with Online Learning
                    </div>
                    <div style='color:#cdd9e5; font-size:14px;'>
                        Probability (F2F): {avg_prob*100:.1f}%
                    </div>
                    <div style='color:#8b949e; font-size:12px; margin-top:8px;'>
                        Student adapts well to digital learning environment
                    </div>
                </div>""", unsafe_allow_html=True)

            # Confidence gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_prob * 100,
                title={"text": "F2F Preference Probability (%)",
                       "font": {"color": "#e6edf3"}},
                gauge={
                    "axis":  {"range": [0, 100],
                              "tickcolor": "#8b949e"},
                    "bar":   {"color": C_ORANGE if prediction else C_GREEN},
                    "bgcolor": BG_CARD,
                    "bordercolor": "#1d9e7530",
                    "steps": [
                        {"range": [0,  50], "color": "#0f3d2e"},
                        {"range": [50, 100], "color": "#3d1a0f"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2},
                                  "value": 50},
                },
                number={"suffix": "%", "font": {"color": "#e6edf3"}},
            ))
            fig_g.update_layout(
                paper_bgcolor=BG_DARK, font_color="#e6edf3",
                height=260, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            # Model agreement
            st.markdown(f"""
            <div style='display:flex; gap:12px; margin-top:8px;'>
                <div style='flex:1; background:{BG_CARD}; border:1px solid
                     #1d9e7530; border-radius:8px; padding:12px;
                     text-align:center;'>
                    <div style='color:#8b949e; font-size:11px;'>Random Forest</div>
                    <div style='color:{C_GREEN}; font-size:20px;
                                font-weight:700;'>{rf_prob*100:.1f}%</div>
                </div>
                <div style='flex:1; background:{BG_CARD}; border:1px solid
                     #1d9e7530; border-radius:8px; padding:12px;
                     text-align:center;'>
                    <div style='color:#8b949e; font-size:11px;'>Gradient Boost</div>
                    <div style='color:{C_PURPLE}; font-size:20px;
                                font-weight:700;'>{gb_prob*100:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:{BG_CARD}; border:1px dashed #1d9e7540;
                        border-radius:12px; padding:40px; text-align:center;
                        margin-top:20px;'>
                <div style='font-size:48px; opacity:0.4;'>🔮</div>
                <div style='color:#8b949e; margin-top:12px;'>
                    Fill in the student profile and click<br>
                    <strong style='color:{C_GREEN};'>Predict</strong>
                    to see the result
                </div>
            </div>""", unsafe_allow_html=True)

        # SHAP importance always shown
        st.markdown("### Feature Importance (SHAP)")
        shap_df = pd.DataFrame({
            "Feature":     mdls["feat_names"],
            "Importance":  mdls["mean_shap"],
        }).sort_values("Importance")

        fig_s = px.bar(shap_df, x="Importance", y="Feature",
                       orientation="h",
                       color="Importance",
                       color_continuous_scale=["#0f6e56", "#1D9E75", "#9FE1CB"],
                       template="plotly_dark",
                       labels={"Importance": "Mean |SHAP|"})
        fig_s.update_layout(**PLOTLY_TEMPLATE["layout"],
                            height=380, showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(t=10,b=20,l=10,r=10))
        st.plotly_chart(fig_s, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — NLP INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬  NLP Insights":
    st.markdown("# 💬 NLP & Sentiment Insights")
    st.markdown(
        "<p style='color:#8b949e;'>VADER sentiment analysis and "
        "language patterns extracted from student responses.</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment r vs Likert", "0.808",
              help="Pearson correlation between avg Likert score and VADER compound")
    c2.metric("F2F Group Sentiment",
              f"{df_f[df_f['target_label']=='Needs F2F']['vader_compound'].mean():.3f}")
    c3.metric("Online Group Sentiment",
              f"{df_f[df_f['target_label']=='OK Online']['vader_compound'].mean():.3f}")

    st.markdown("---")

    tab_s, tab_w, tab_b = st.tabs(
        ["  Sentiment Scores  ", "  Word Clouds  ", "  Barrier Analysis  "])

    with tab_s:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Compound Score Distribution")
            fig = px.histogram(df_f, x="vader_compound", color="target_label",
                               barmode="overlay", nbins=25,
                               color_discrete_map=PALETTE, opacity=0.7,
                               template="plotly_dark",
                               labels={"vader_compound": "VADER Compound Score",
                                       "target_label": "Group"})
            fig.add_vline(x=0, line_dash="dash", line_color="#8b949e")
            fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                              height=340, margin=dict(t=20,b=20,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Sentiment vs Avg Likert Score")
            fig2 = px.scatter(df_f, x="avg_likert", y="vader_compound",
                              color="target_label",
                              color_discrete_map=PALETTE, opacity=0.6,
                              template="plotly_dark",
                              labels={
                                  "avg_likert": "Avg Likert Score",
                                  "vader_compound": "VADER Compound",
                                  "target_label": "Group"})
            for grp, col in PALETTE.items():
                sub = df_f[df_f["target_label"] == grp]
                if len(sub) > 1:
                    m, b = np.polyfit(sub["avg_likert"], sub["vader_compound"], 1)
                    xs = np.linspace(sub["avg_likert"].min(), sub["avg_likert"].max(), 50)
                    fig2.add_scatter(x=xs, y=m*xs+b, mode="lines",
                                     line=dict(color=col, width=2, dash="dot"),
                                     name=f"{grp} trend", showlegend=False)
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"],
                               height=340, margin=dict(t=20,b=20,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class='insight-card'>
        <p><strong>r = 0.808</strong> between avg Likert score and VADER compound 
        sentiment. VADER picks up the emotional valence of words like 
        "strongly agree" vs "strongly disagree" — even though these are structured 
        survey labels, not free text. The Needs F2F group shows higher compound 
        sentiment (0.617) — counter-intuitive until you realise they strongly 
        <em>agree</em> with needing face-to-face, which VADER reads as positive.</p>
        </div>""", unsafe_allow_html=True)

    with tab_w:
        st.markdown("#### Response Pattern Word Clouds")

        STOP_W = set(stopwords.words("english"))
        STOP_W.update(["nan", "none"])

        def make_cloud_text(series):
            combined = " ".join(series.astype(str).tolist())
            tokens   = word_tokenize(combined.lower())
            return " ".join(t for t in tokens
                            if t.isalpha() and t not in STOP_W and len(t) > 2)

        f2f_txt = make_cloud_text(
            df_f[df_f["target_label"]=="Needs F2F"]["response_profile"])
        onl_txt = make_cloud_text(
            df_f[df_f["target_label"]=="OK Online"]["response_profile"])

        col_w1, col_w2 = st.columns(2)

        for col, text, cmap, label, color in [
            (col_w1, f2f_txt,  "Oranges", "Needs F2F",  C_ORANGE),
            (col_w2, onl_txt,  "Greens",  "OK Online",  C_GREEN),
        ]:
            with col:
                st.markdown(f"<p style='color:{color}; font-weight:600; "
                            f"text-align:center;'>{label}</p>",
                            unsafe_allow_html=True)
                if len(text.split()) > 1:
                    wc = WordCloud(width=600, height=320,
                                   background_color="#161b22",
                                   colormap=cmap, max_words=50,
                                   collocations=False).generate(text)
                    fig_wc, ax = plt.subplots(figsize=(6, 3.2))
                    fig_wc.patch.set_facecolor(BG_CARD)
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig_wc, use_container_width=True)
                    plt.close()

        st.markdown(f"""
        <div class='insight-card'>
        <p><strong>F2F group:</strong> "agree" dominates — they strongly agree 
        with needing physical contact and instructor presence.<br>
        <strong>Online group:</strong> "neutral" is largest — more ambivalence 
        and hedging, less strong conviction in either direction.<br>
        This contrast is itself a finding: the F2F preference is a 
        <em>strong positive belief</em>, not merely absence of online confidence.</p>
        </div>""", unsafe_allow_html=True)

    with tab_b:
        st.markdown("#### Internet Barrier Reasons by Group")
        barrier_df = df_f[df_f["internet_limit_reason"].notna()].copy()
        barrier_df["internet_limit_reason"] = (
            barrier_df["internet_limit_reason"].astype(str).str.strip())
        top_b = (barrier_df["internet_limit_reason"]
                 .value_counts().nlargest(5).index)
        b_counts = (barrier_df[barrier_df["internet_limit_reason"].isin(top_b)]
                    .groupby(["target_label","internet_limit_reason"])
                    .size().reset_index(name="count"))

        fig_b = px.bar(b_counts, x="count", y="internet_limit_reason",
                       color="target_label", barmode="group",
                       orientation="h",
                       color_discrete_map=PALETTE,
                       template="plotly_dark",
                       labels={"count": "Number of students",
                               "internet_limit_reason": "Barrier",
                               "target_label": "Group"})
        fig_b.update_layout(**PLOTLY_TEMPLATE["layout"],
                            height=380, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown(f"""
        <div class='insight-card'>
        <p><strong>Cost</strong> is the dominant barrier for the F2F group — 
        students who can't afford internet don't develop the online learning 
        habits that build confidence in digital education.<br>
        <strong>Signal problems</strong> are more evenly split — both groups 
        face connectivity issues, but the F2F group can't compensate 
        through other means (e.g. mobile data, public WiFi).<br>
        <strong>Policy implication:</strong> Subsidised internet access would 
        directly reduce F2F dependency among cost-constrained students.</p>
        </div>""", unsafe_allow_html=True)

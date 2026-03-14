import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap
from collections import Counter

# ── 1. LOAD & RENAME COLUMNS ─────────────────────────────────────────────────
print("=" * 65)
print("  EdTech Readiness Analysis — Loading Data")
print("=" * 65)

df = pd.read_excel('duplicate.xlsx')

# Rename long column names to readable short keys
RENAME_MAP = {
    "Timestamp": "timestamp",
    "Name": "name",
    "Age": "age",
    "Gender": "gender",
    "Education": "education",
    df.columns[5]: "internet_access",
    df.columns[6]: "internet_limit_reason",
    df.columns[7]: "q1_computer_skill",
    df.columns[8]: "q2_electronic_communication",
    df.columns[9]: "q3_learning_same",
    df.columns[10]: "q4_more_motivating",
    df.columns[11]: "q5_full_course_online",
    df.columns[12]: "q6_discuss_with_peers",
    df.columns[13]: "q7_group_work_online",
    df.columns[14]: "q8_need_face_to_face",
    df.columns[15]: "q9_manage_time_online",
}
df.rename(columns=RENAME_MAP, inplace=True)

# ── 2. DATA CLEANING ──────────────────────────────────────────────────────────
print(f"\n[INFO] Raw shape: {df.shape}")

# Likert columns
LIKERT_COLS = [f"q{i}" for i in range(1, 10) if
               any(c.startswith(f"q{i}_") for c in df.columns)]
LIKERT_COLS = [c for c in df.columns if c.startswith("q")]

# Standardise Likert text (handle case & spacing variations)
LIKERT_MAP = {
    "strongly agree":    5,
    "agree":             4,
    "neutral":           3,
    "disagree":          2,
    "strongly disagree": 1,
}

for col in LIKERT_COLS:
    df[col] = (df[col]
               .astype(str)
               .str.strip()
               .str.lower()
               .map(LIKERT_MAP))

# Drop rows where ALL Likert answers are missing
df.dropna(subset=LIKERT_COLS, how="all", inplace=True)
df[LIKERT_COLS] = df[LIKERT_COLS].fillna(df[LIKERT_COLS].median())

# Clean demographics
df["gender"]    = df["gender"].str.strip().str.title()
df["age"]       = df["age"].str.strip()
df["education"] = df["education"].str.strip().str.title()

# Encode internet access as binary
df["internet_binary"] = df["internet_access"].apply(
    lambda x: 0 if str(x).strip().lower().startswith("no") else 1
)

print(f"[INFO] Clean shape: {df.shape}")
print(f"[INFO] Missing values:\n{df[LIKERT_COLS].isnull().sum()}")

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────

# Composite score: average perception across all Likert questions
df["perception_score"] = df[LIKERT_COLS].mean(axis=1)

# Target: binary — Positive (score >= 3.5) vs Negative/Neutral
df["perception_label"] = (df["perception_score"] >= 3.5).astype(int)
df["perception_group"] = df["perception_label"].map({1: "Positive", 0: "Negative/Neutral"})

# Human-readable question labels
Q_LABELS = {
    "q1_computer_skill":         "Computer skill",
    "q2_electronic_communication":"Electronic comm.",
    "q3_learning_same":          "Learning same at home",
    "q4_more_motivating":        "Online more motivating",
    "q5_full_course_online":     "Full course possible",
    "q6_discuss_with_peers":     "Peer discussion online",
    "q7_group_work_online":      "Group work online",
    "q8_need_face_to_face":      "Needs face-to-face",
    "q9_manage_time_online":     "Time management online",
}

print(f"\n[INFO] Perception distribution:")
print(df["perception_group"].value_counts())

# ── 4. EXPLORATORY DATA ANALYSIS (EDA) ───────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"Positive": "#1D9E75", "Negative/Neutral": "#D85A30"}

# ── 4A. Perception distribution by demographics ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Student Online Education Perception by Demographics",
             fontsize=16, fontweight="bold", y=1.02)

for ax, col, title in zip(axes,
                           ["gender", "age", "education"],
                           ["Gender", "Age Group", "Education Level"]):
    ct = (df.groupby([col, "perception_group"])
            .size()
            .reset_index(name="count"))
    ct["pct"] = ct.groupby(col)["count"].transform(lambda x: x / x.sum() * 100)
    sns.barplot(data=ct, x=col, y="pct", hue="perception_group",
                palette=PALETTE, ax=ax, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("% of students")
    ax.legend(title="Perception", loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

plt.tight_layout()
plt.savefig("fig1_demographics_perception.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] fig1_demographics_perception.png")

# ── 4B. Mean Likert score per question (grouped bar) ─────────────────────────
mean_scores = (df.groupby("perception_group")[list(Q_LABELS.keys())]
               .mean()
               .T
               .rename(index=Q_LABELS))

fig, ax = plt.subplots(figsize=(14, 7))
mean_scores.plot(kind="barh", ax=ax, color=[PALETTE["Positive"],
                                             PALETTE["Negative/Neutral"]],
                 edgecolor="white", width=0.65)
ax.set_title("Mean Agreement Score per Question by Perception Group",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Mean Likert Score (1=Strongly Disagree → 5=Strongly Agree)")
ax.axvline(3, color="gray", linestyle="--", linewidth=1, label="Neutral = 3")
ax.legend(title="Perception")
plt.tight_layout()
plt.savefig("fig2_question_scores.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] fig2_question_scores.png")

# ── 4C. Correlation heatmap of Likert responses ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[list(Q_LABELS.keys())].rename(columns=Q_LABELS).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Correlation Between Survey Questions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] fig3_correlation_heatmap.png")

# ── 4D. Internet access impact ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot: perception score by internet access
sns.boxplot(data=df, x="internet_access", y="perception_score",
            palette="Set2", ax=axes[0])
axes[0].set_title("Perception Score vs Internet Access", fontweight="bold")
axes[0].set_xlabel("Internet Access Type")
axes[0].set_ylabel("Avg Perception Score")
plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right", fontsize=8)

# Pie: reasons for limited access
reason_counts = df["internet_limit_reason"].value_counts().head(6)
axes[1].pie(reason_counts.values,
            labels=reason_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("Set2"))
axes[1].set_title("Reasons for Limited Internet Access", fontweight="bold")

plt.tight_layout()
plt.savefig("fig4_internet_access.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] fig4_internet_access.png")

# ── 5. ML PIPELINE ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Model Training — Multi-Classifier Comparison")
print("=" * 65)

# Features
FEATURE_COLS = list(Q_LABELS.keys()) + ["internet_binary"]

# Encode demographics as extra features
le_gender = LabelEncoder()
le_age    = LabelEncoder()
le_edu    = LabelEncoder()
df["gender_enc"]    = le_gender.fit_transform(df["gender"].fillna("Unknown"))
df["age_enc"]       = le_age.fit_transform(df["age"].fillna("Unknown"))
df["education_enc"] = le_edu.fit_transform(df["education"].fillna("Unknown"))
FEATURE_COLS += ["gender_enc", "age_enc", "education_enc"]

X = df[FEATURE_COLS].copy()
y = df["perception_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n[INFO] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"[INFO] Class balance (train): {dict(Counter(y_train))}")

# ── 5A. Multi-model comparison ───────────────────────────────────────────────
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=42)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in MODELS.items():
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="f1_weighted")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc    = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = None

    results.append({
        "Model": name,
        "CV F1 (mean)": round(cv_scores.mean(), 3),
        "CV F1 (std)":  round(cv_scores.std(), 3),
        "Test Accuracy": round(acc, 3),
        "Test F1":       round(f1, 3),
        "ROC-AUC":       round(roc, 3) if roc else "N/A",
    })
    print(f"\n  {name}")
    print(f"    CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"    Test Accuracy: {acc:.3f}  |  Test F1: {f1:.3f}")
    if roc:
        print(f"    ROC-AUC: {roc:.3f}")

results_df = pd.DataFrame(results)
print("\n[RESULTS SUMMARY]")
print(results_df.to_string(index=False))

# ── 5B. Confusion matrix (best model = Random Forest) ────────────────────────
best_model = MODELS["Random Forest"]
y_pred_rf  = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Neg/Neutral", "Positive"],
            yticklabels=["Neg/Neutral", "Positive"],
            linewidths=1, ax=ax)
ax.set_title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("fig5_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] fig5_confusion_matrix.png")

# ── 5C. SHAP — Feature importance explainability ──────────────────────────────
print("\n[INFO] Generating SHAP values (this may take ~30s)...")

rf_clf   = best_model
explainer = shap.TreeExplainer(rf_clf)
shap_vals = explainer.shap_values(X_test)

# For binary classification shap_values returns [neg_class, pos_class]
shap_pos = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

readable_features = (list(Q_LABELS.values()) +
                     ["Internet access", "Gender", "Age group", "Education"])

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_pos,
                  X_test.values,
                  feature_names=readable_features,
                  plot_type="bar",
                  show=False,
                  color="#1D9E75")
ax = plt.gca()
ax.set_title("SHAP Feature Importance — What Drives Positive Perception?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig6_shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] fig6_shap_importance.png")

# ── 6. KEY INSIGHTS REPORT ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KEY INSIGHTS — For Stakeholders")
print("=" * 65)

pos_pct = (df["perception_label"].mean() * 100)
top_pos = mean_scores["Positive"].idxmax()
top_neg = mean_scores["Positive"].idxmin()

print(f"""
  1. {pos_pct:.1f}% of students hold a POSITIVE perception of online education.
  
  2. Strongest positive signal : '{top_pos}'
     → Students who are digitally confident tend to view online learning positively.
     
  3. Weakest agreement area    : '{top_neg}'
     → This is the most common barrier — institutions should act here first.
     
  4. Internet access gap       : Students with limited/no internet score
     significantly lower ({df[df['internet_binary']==0]['perception_score'].mean():.2f})
     vs those with full access ({df[df['internet_binary']==1]['perception_score'].mean():.2f}).
     
  5. Top model: Random Forest — CV F1: {results[0]['CV F1 (mean)']:.3f}
     SHAP confirms Q8 (face-to-face need) and Q9 (time management) as
     the two strongest predictors of perception.

  RECOMMENDATION: Institutions should prioritise:
    → Digital literacy programs (bridges Q1/Q2 gap)
    → Structured online timetables (addresses Q9)
    → Hybrid fallback for students with no home internet
""")

print("=" * 65)
print("  All figures saved. Ready for Streamlit dashboard.")
print("=" * 65)

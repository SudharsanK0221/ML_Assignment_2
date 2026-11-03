"""
Online Shopper Buying Predictor — Professional Edition
Machine Learning - Assignment 2 (M.Tech AIML, BITS Pilani WILP)

What this app does:
  1. You upload a test CSV file (it must have all 17 clue columns + the Revenue column).
  2. You explore the data (class balance, correlations, key clue patterns).
  3. You pick one of the 5 saved models.
  4. The app loads the SAVED model (no re-training) and shows the scores,
     the confusion matrix and the classification report.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_curve)

# Page settings
st.set_page_config(page_title="Shopper Buying Predictor", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titleweight"] = "bold"

# A little bit of custom CSS for a cleaner, more professional look
CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(120deg, #4C72B0 0%, #6f9bd1 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.1rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: 6px 0 0 0; opacity: 0.92; font-size: 1.02rem;}
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4C72B0 !important;
        color: white !important;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fc;
        border: 1px solid #e3e7f0;
        border-radius: 12px;
        padding: 10px 14px 4px 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .section-title {
        border-left: 5px solid #4C72B0;
        padding-left: 10px;
        margin: 0.4rem 0 0.8rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Fixed facts about our dataset and saved models
HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "Logistic Regression":    os.path.join(HERE, "model", "logistic_regression.pkl"),
    "Decision Tree":          os.path.join(HERE, "model", "decision_tree.pkl"),
    "KNN":                    os.path.join(HERE, "model", "knn.pkl"),
    "Naive Bayes":            os.path.join(HERE, "model", "naive_bayes.pkl"),
    "Support Vector Machine": os.path.join(HERE, "model", "svm.pkl"),
}

CLUE_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend",
]
WORD_COLUMNS = ["Month", "VisitorType"]
NUMBER_COLUMNS = [c for c in CLUE_COLUMNS if c not in WORD_COLUMNS]
KEY_FEATURES = ["PageValues", "ExitRates", "BounceRates", "ProductRelated_Duration"]
ANSWER_COLUMN = "Revenue"
SAMPLE_FILE = os.path.join(HERE, "test_data.csv")
CLASS_NAMES = ["No buy (0)", "Buy (1)"]


# Small helper functions
@st.cache_resource
def load_saved_model(model_name):
    """Load one saved model from the model/ folder (done only once per model)."""
    return joblib.load(MODEL_FILES[model_name])


def to_zero_one(column):
    """Turn True/False (or 1/0, or 'True'/'False' words) into 1/0."""
    word_map = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0}
    return column.astype(str).str.strip().str.lower().map(word_map)


def check_and_prepare(table):
    """
    Check the uploaded table has the columns we need.
    Returns (clues, answers, error_message).
    """
    missing_columns = [c for c in CLUE_COLUMNS + [ANSWER_COLUMN] if c not in table.columns]
    if missing_columns:
        return None, None, "These required columns are missing: " + ", ".join(missing_columns)

    table = table.copy()
    table["Weekend"] = to_zero_one(table["Weekend"])
    table[ANSWER_COLUMN] = to_zero_one(table[ANSWER_COLUMN])

    if table[["Weekend", ANSWER_COLUMN]].isnull().any().any():
        return None, None, "The 'Weekend' and 'Revenue' columns must only have True/False or 1/0 values."

    clues = table[CLUE_COLUMNS]
    answers = table[ANSWER_COLUMN].astype(int)
    return clues, answers, None


def calculate_scores(real_answers, guessed_answers, guessed_chances):
    """The 6 scores asked in the assignment."""
    only_one_class = real_answers.nunique() < 2
    return {
        "Accuracy":  accuracy_score(real_answers, guessed_answers),
        "AUC":       np.nan if only_one_class else roc_auc_score(real_answers, guessed_chances),
        "Precision": precision_score(real_answers, guessed_answers, zero_division=0),
        "Recall":    recall_score(real_answers, guessed_answers, zero_division=0),
        "F1":        f1_score(real_answers, guessed_answers, zero_division=0),
        "MCC":       matthews_corrcoef(real_answers, guessed_answers),
    }


def run_model(model_name, clues, answers):
    """Load the saved model, make guesses, and return the guesses, chances and scores."""
    model = load_saved_model(model_name)
    guessed_answers = model.predict(clues)
    guessed_chances = model.predict_proba(clues)[:, 1]
    scores = calculate_scores(answers, guessed_answers, guessed_chances)
    return guessed_answers, guessed_chances, scores


# Top of the page
st.markdown(
    """
    <div class="hero">
        <h1>🛒 Online Shopper Buying Predictor</h1>
        <p>Guess whether a website visit ends in a purchase, using 5 trained ML models — evaluated live, never re-trained.</p>
        <span class="badge">5 ML Models</span>
        <span class="badge">6 Evaluation Metrics</span>
        <span class="badge">Interactive EDA</span>
        <span class="badge">No Retraining</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: upload file + choose model
with st.sidebar:
    st.markdown("### 🛒 Shopper Predictor")
    st.caption("ML Assignment 2 · BITS Pilani WILP")
    st.divider()

    st.markdown("**1. Give the test data**")
    uploaded_file = st.file_uploader("Upload test data (CSV file)", type=["csv"])

    use_sample = False
    if os.path.exists(SAMPLE_FILE):
        use_sample = st.checkbox("No file? Use the sample test_data.csv from the repo")
        with open(SAMPLE_FILE, "rb") as sample_handle:
            st.download_button("⬇️ Download sample test_data.csv", sample_handle.read(),
                               file_name="test_data.csv", mime="text/csv")

    st.markdown("**2. Choose a model**")
    chosen_model_name = st.selectbox("Model", list(MODEL_FILES.keys()))

    with st.expander("Required columns"):
        st.caption(", ".join(CLUE_COLUMNS + [ANSWER_COLUMN]))

# Read the data (uploaded file first, then the sample file)
if uploaded_file is not None:
    raw_table = pd.read_csv(uploaded_file)
    data_source_text = f"Uploaded file: {getattr(uploaded_file, 'name', 'test data')}"
elif use_sample:
    raw_table = pd.read_csv(SAMPLE_FILE)
    data_source_text = "Sample file: test_data.csv"
else:
    st.info("👈 Please upload a test CSV file in the sidebar (or tick the sample file option) to begin.")
    st.stop()

clues, answers, error_message = check_and_prepare(raw_table)
if error_message:
    st.error(error_message)
    st.stop()

# Make sure all model files are there before going further
missing_model_files = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]
if missing_model_files:
    st.error("Saved model files are missing for: " + ", ".join(missing_model_files))
    st.stop()

st.success(f"✅ {data_source_text}  |  {clues.shape[0]} rows, {clues.shape[1]} clue columns")
buy_percent = answers.mean() * 100

# Results
tab_overview, tab_eda, tab_one_model, tab_all_models = st.tabs(
    ["🏠 Overview", "📊 Data Analysis", "🤖 Model Evaluation", "🏆 Compare All Models"]
)

# ---- Tab 0: project & dataset overview --------------------------------------
with tab_overview:
    st.markdown('<div class="section-title"><h4>Project snapshot</h4></div>', unsafe_allow_html=True)
    stat_cols = st.columns(4)
    stat_cols[0].metric("Rows loaded", f"{clues.shape[0]:,}")
    stat_cols[1].metric("Clue columns", clues.shape[1])
    stat_cols[2].metric("Purchase rate", f"{buy_percent:.1f}%")
    stat_cols[3].metric("Models available", len(MODEL_FILES))

    st.markdown('<div class="section-title"><h4>Problem statement</h4></div>', unsafe_allow_html=True)
    st.write(
        "A shopping website gets many visitors, but only a few actually buy something. "
        "This app uses the details of a website visit (pages seen, time spent, month, "
        "visitor type, etc.) to **predict if the visit will end in a purchase** "
        "(`Revenue` = 1) or not (`Revenue` = 0), using 5 saved classification models."
    )

    st.markdown('<div class="section-title"><h4>Dataset description</h4></div>', unsafe_allow_html=True)
    st.markdown(
        "- **Source:** Online Shoppers Purchasing Intention Dataset (UCI Machine Learning Repository)\n"
        "- **Problem type:** Binary classification\n"
        "- **Features:** 17 clue columns (numeric browsing behaviour + month + visitor type)\n"
        "- **Target:** `Revenue` — whether the visit ended in a purchase\n"
        "- **Class balance:** Imbalanced — only a small share of visits end in a purchase, "
        "which is why Accuracy alone is not enough and AUC/Precision/Recall/F1/MCC are also tracked."
    )

    with st.expander("📄 Preview the loaded data (first 10 rows)"):
        st.dataframe(raw_table.head(10))

# ---- Tab 1: exploratory data analysis on the loaded data --------------------
with tab_eda:
    st.markdown('<div class="section-title"><h4>Class balance</h4></div>', unsafe_allow_html=True)
    balance_counts = answers.value_counts().sort_index()
    pie_col, bar_col = st.columns(2)

    with pie_col:
        figure, axis = plt.subplots(figsize=(4.2, 4.2))
        axis.pie(balance_counts.values, labels=["No buy", "Buy"], autopct="%1.1f%%",
                colors=["#4C72B0", "#DD8452"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
        axis.set_title("Buy vs No buy")
        st.pyplot(figure)
        plt.close(figure)

    with bar_col:
        st.markdown("##### Revenue counts")
        st.dataframe(balance_counts.rename("Count").rename_axis("Revenue").reset_index(),
                    hide_index=True, width="stretch")
        st.caption(
            "A model that always guesses 'No buy' would already look accurate on this data. "
            "That is why Precision, Recall, F1 and MCC matter as much as Accuracy."
        )

    st.markdown('<div class="section-title"><h4>Correlation with the target</h4></div>', unsafe_allow_html=True)
    combined = clues[NUMBER_COLUMNS].copy()
    combined[ANSWER_COLUMN] = answers.values
    correlation_matrix = combined.corr()

    heat_col, rank_col = st.columns([2, 1])
    with heat_col:
        figure, axis = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, linewidths=0.4, ax=axis)
        axis.set_title("Correlation heatmap (numeric clues + Revenue)")
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with rank_col:
        st.markdown("##### Top correlated clues")
        top_correlations = correlation_matrix[ANSWER_COLUMN].drop(ANSWER_COLUMN)
        top_correlations = top_correlations.reindex(top_correlations.abs().sort_values(ascending=False).index)
        st.dataframe(top_correlations.round(3).rename("Correlation"), width="stretch")

    st.markdown('<div class="section-title"><h4>Key clues: Buy vs No buy</h4></div>', unsafe_allow_html=True)
    available_key_features = [f for f in KEY_FEATURES if f in clues.columns]
    if available_key_features:
        figure, boxes = plt.subplots(1, len(available_key_features), figsize=(4.4 * len(available_key_features), 4))
        boxes = np.atleast_1d(boxes)
        plot_frame = clues[available_key_features].copy()
        plot_frame[ANSWER_COLUMN] = answers.values
        for box, feature in zip(boxes, available_key_features):
            sns.boxplot(data=plot_frame, x=ANSWER_COLUMN, y=feature, hue=ANSWER_COLUMN,
                        palette=["#4C72B0", "#DD8452"], legend=False, ax=box)
            box.set_xticks([0, 1])
            box.set_xticklabels(["No buy", "Buy"])
            box.set_xlabel("")
            box.set_title(feature)
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    if "Month" in clues.columns or "VisitorType" in clues.columns:
        st.markdown('<div class="section-title"><h4>Purchase rate by category</h4></div>', unsafe_allow_html=True)
        cat_cols = st.columns(2)
        if "Month" in clues.columns:
            month_rate = pd.Series(answers.values, index=clues["Month"]).groupby(level=0).mean().sort_values(ascending=False)
            figure, axis = plt.subplots(figsize=(5.5, 4))
            sns.barplot(x=month_rate.index, y=month_rate.values, hue=month_rate.index,
                        palette="viridis", legend=False, ax=axis)
            axis.set_title("Purchase rate by Month")
            axis.set_ylabel("Share that bought")
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            cat_cols[0].pyplot(figure)
            plt.close(figure)
        if "VisitorType" in clues.columns:
            visitor_rate = pd.Series(answers.values, index=clues["VisitorType"]).groupby(level=0).mean().sort_values(ascending=False)
            figure, axis = plt.subplots(figsize=(5.5, 4))
            sns.barplot(x=visitor_rate.index, y=visitor_rate.values, hue=visitor_rate.index,
                        palette="magma", legend=False, ax=axis)
            axis.set_title("Purchase rate by Visitor Type")
            axis.set_ylabel("Share that bought")
            axis.set_xlabel("")
            plt.tight_layout()
            cat_cols[1].pyplot(figure)
            plt.close(figure)

# ---- Tab 2: the model chosen in the dropdown --------------------------------
with tab_one_model:
    guessed_answers, guessed_chances, scores = run_model(chosen_model_name, clues, answers)

    st.markdown(f'<div class="section-title"><h4>{chosen_model_name}: evaluation scores</h4></div>', unsafe_allow_html=True)
    score_names = list(scores.keys())
    for row_start in (0, 3):
        boxes = st.columns(3)
        for box, score_name in zip(boxes, score_names[row_start:row_start + 3]):
            value = scores[score_name]
            box.metric(score_name, "n/a" if np.isnan(value) else f"{value:.4f}")

    if np.isnan(scores["AUC"]):
        st.warning("AUC cannot be calculated because the Revenue column has only one kind of value.")

    left_side, right_side = st.columns(2)

    with left_side:
        st.markdown("##### Confusion matrix")
        matrix = confusion_matrix(answers, guessed_answers, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                    xticklabels=["No buy", "Buy"], yticklabels=["No buy", "Buy"])
        axis.set_xlabel("What the model guessed")
        axis.set_ylabel("What really happened")
        st.pyplot(figure)
        plt.close(figure)

        true_no, false_yes, false_no, true_yes = matrix.ravel()
        st.caption(
            f"Correct 'No buy': {true_no} | Wrongly guessed 'Buy': {false_yes} | "
            f"Missed buyers: {false_no} | Correct 'Buy': {true_yes}"
        )

    with right_side:
        st.markdown("##### Classification report")
        report_table = pd.DataFrame(
            classification_report(answers, guessed_answers, labels=[0, 1],
                                  target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        ).T.round(3)
        st.dataframe(report_table)

        if not np.isnan(scores["AUC"]):
            st.markdown("##### ROC curve")
            false_alarm_rate, catch_rate, _ = roc_curve(answers, guessed_chances)
            figure, axis = plt.subplots(figsize=(4.5, 3.4))
            axis.plot(false_alarm_rate, catch_rate, label=f"AUC = {scores['AUC']:.3f}")
            axis.plot([0, 1], [0, 1], "k--", label="Coin toss")
            axis.set_xlabel("False alarm rate")
            axis.set_ylabel("Catch rate (Recall)")
            axis.legend(loc="lower right")
            st.pyplot(figure)
            plt.close(figure)

    # Let the person download the guesses of the chosen model
    guess_table = raw_table.copy()
    guess_table["Model_Guess"] = guessed_answers
    guess_table["Chance_Of_Buying"] = guessed_chances.round(4)
    st.download_button("Download the guesses of this model (CSV)",
                       guess_table.to_csv(index=False).encode("utf-8"),
                       file_name=f"guesses_{chosen_model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")

# ---- Tab 3: all models side by side -----------------------------------------
with tab_all_models:
    st.markdown('<div class="section-title"><h4>All 5 models on the same test data</h4></div>', unsafe_allow_html=True)

    all_scores = {}
    for model_name in MODEL_FILES:
        _, _, model_scores = run_model(model_name, clues, answers)
        all_scores[model_name] = model_scores

    comparison_table = pd.DataFrame(all_scores).T.round(4)
    comparison_table.index.name = "ML Model Name"

    best_model_name = comparison_table["MCC"].idxmax()
    styled_table = comparison_table.style.highlight_max(axis=0, color="#d7e8fb")
    st.dataframe(styled_table, width="stretch")
    st.success(f"🏆 Best model on this data (highest MCC): **{best_model_name}**")

    figure, axis = plt.subplots(figsize=(10, 4.5))
    comparison_table.plot(kind="bar", ax=axis, width=0.8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.legend(loc="upper right", ncol=3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

st.divider()
st.caption("Built with Streamlit. Models were trained earlier in the notebook. This app only loads saved models and scores the uploaded data - it never re-trains.")

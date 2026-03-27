import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from generator import generate_bulk
from preprocessing import (
    clean_text,
    transform_tfidf,
    transform_lstm_sequences,
    load_preprocessors
)
from detection import detect
from analysis import analyze
from models.ml_model import load_all_models
from models.lstm_model import load_model_file

st.set_page_config(page_title="AI Phishing Detection System", layout="wide")
st.title("AI-Driven Phishing Email Generation and Detection Analysis Framework")


@st.cache_resource
def load_inference_assets():
    load_preprocessors()
    load_all_models()
    load_model_file()
    return True


def model_files_exist():
    required_files = [
        "saved_models/tfidf_vectorizer.pkl",
        "saved_models/tokenizer.pkl",
        "saved_models/logistic.pkl",
        "saved_models/naive_bayes.pkl",
        "saved_models/svm.pkl",
        "saved_models/random_forest.pkl",
        "saved_models/lstm_model.keras",
    ]
    return all(os.path.exists(f) for f in required_files)


st.subheader("Email Generation Settings")
prompt = st.text_area("Text Prompt", value="Generate realistic email samples")
num = st.slider("Number of Emails", 1, 5, 3)
email_type = st.selectbox("Email Type", ["phishing", "legitimate", "mixed"])
tone = st.selectbox("Tone", ["urgent", "formal", "friendly"])
difficulty = st.selectbox("Difficulty", ["easy", "medium", "advanced"])
target = st.text_input("Target", value="user")

if "emails" not in st.session_state:
    st.session_state["emails"] = []

if st.button("Generate Emails"):
    st.session_state["emails"] = generate_bulk(num, tone, target, difficulty, email_type)
    st.success(f"{len(st.session_state['emails'])} email(s) generated successfully.")

if st.session_state["emails"]:
    st.subheader("Generated Emails")
    for idx, e in enumerate(st.session_state["emails"], start=1):
        st.markdown(f"### Email {idx}")
        st.text_area(f"Generated Email {idx}", e["email"], height=250, key=f"email_{idx}")

if st.button("Run Detection & Analysis"):
    if not st.session_state["emails"]:
        st.warning("Please generate emails first.")
        st.stop()

    if not model_files_exist():
        st.error("Saved models not found. Run train_and_save.py locally first.")
        st.stop()

    with st.spinner("Loading saved models and preprocessors..."):
        load_inference_assets()

    df = pd.DataFrame(st.session_state["emails"])
    df["clean_email"] = df["email"].apply(clean_text)

    X_test_tfidf = transform_tfidf(df["clean_email"])
    X_test_lstm = transform_lstm_sequences(df["clean_email"])

    results = []
    analysis_results = []

    for i, row in df.iterrows():
        res = detect(row["email"], X_test_tfidf[i], X_test_lstm[i].reshape(1, -1))
        ana = analyze(row["email"])

        ml_preds = res["ml_models"]

        results.append({
            "Email": row["email"][:80],
            "Logistic Regression": ml_preds.get("logistic", "N/A"),
            "Naive Bayes": ml_preds.get("naive_bayes", "N/A"),
            "SVM": ml_preds.get("svm", "N/A"),
            "Random Forest": ml_preds.get("random_forest", "N/A"),
            "LSTM": res["lstm"],
            "BERT": res["bert"],
            "Final Decision": res["final"],
            "True Label": row["label"]
        })

        analysis_results.append({
            "Trigger Words": ", ".join(ana.get("triggers", [])),
            "Keywords": ", ".join(ana.get("keywords", [])),
            "Reason": ana.get("reason", ""),
            "Length": ana.get("length", 0),
            "Word Count": ana.get("word_count", 0),
            "Uppercase Count": ana.get("uppercase", 0),
            "Exclamation Count": ana.get("exclamation_count", 0),
            "Digit Count": ana.get("digit_count", 0),
            "URL-like Count": ana.get("url_like_count", 0)
        })

    results_df = pd.DataFrame(results)
    analysis_df = pd.DataFrame(analysis_results)

    st.subheader("Detection Results")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Analysis Results")
    st.dataframe(analysis_df, use_container_width=True)

    # Detection comparison chart
    st.subheader("Detection Comparison")

    model_columns = [
        "Logistic Regression",
        "Naive Bayes",
        "SVM",
        "Random Forest",
        "LSTM",
        "BERT",
        "Final Decision"
    ]

    model_sums = results_df[model_columns].sum()

    fig, ax = plt.subplots()
    model_sums.plot(kind="bar", ax=ax)
    ax.set_ylabel("Phishing Predictions")
    ax.set_xlabel("Models")
    ax.set_title("Model-wise Detection Comparison")
    st.pyplot(fig)

    # Bypass rate
    phishing_df = results_df[results_df["True Label"] == 1]
    if len(phishing_df) > 0:
        bypass_count = (phishing_df["Final Decision"] == 0).sum()
        bypass_rate = bypass_count / len(phishing_df)

        st.subheader("Bypass Rate")
        st.metric("Bypass Rate", f"{bypass_rate:.2%}")

        fig2, ax2 = plt.subplots()
        ax2.bar(["Detected", "Bypassed"], [len(phishing_df) - bypass_count, bypass_count])
        ax2.set_title("Phishing Email Detection vs Bypass")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
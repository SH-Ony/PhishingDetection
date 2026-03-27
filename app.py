import streamlit as st
import pandas as pd

from generator import generate_bulk
from preprocessing import (
    clean_text,
    fit_tfidf,
    transform_tfidf,
    fit_lstm_tokenizer,
    transform_lstm_sequences,
)
from detection import detect
from analysis import analyze
from utils import load_dataset
from models.ml_model import train_all_models
from models.lstm_model import build_model, train_model


st.set_page_config(page_title="AI Phishing Detection System", layout="wide")
st.title("AI-Driven Phishing Email Generation and Detection Analysis Framework")

# ---------------- Sidebar / Inputs ----------------
st.subheader("Email Generation Settings")

prompt = st.text_area("Text Prompt", value="Generate realistic email samples")
num = st.slider("Number of Emails", 1, 5, 3)
email_type = st.selectbox("Email Type", ["phishing", "legitimate", "mixed"])
tone = st.selectbox("Tone", ["urgent", "formal", "friendly"])
difficulty = st.selectbox("Difficulty", ["easy", "medium", "advanced"])
target = st.text_input("Target", value="user")


# ---------------- Session State ----------------
if "emails" not in st.session_state:
    st.session_state["emails"] = []

if "models_trained" not in st.session_state:
    st.session_state["models_trained"] = False


# ---------------- Generate Emails ----------------
if st.button("Generate Emails"):
    emails = generate_bulk(num, tone, target, difficulty, email_type)
    st.session_state["emails"] = emails
    st.success(f"{len(emails)} email(s) generated successfully.")


# ---------------- Display Generated Emails ----------------
if st.session_state["emails"]:
    st.subheader("Generated Emails")
    for idx, e in enumerate(st.session_state["emails"], start=1):
        st.markdown(f"### Email {idx}")
        st.text_area(
            f"Generated Email {idx}",
            e["email"],
            height=250,
            key=f"generated_email_{idx}"
        )


# ---------------- Train Models ----------------
def train_pipeline():
    train_df = load_dataset()

    # Optional: use smaller sample for faster experimentation
    # Comment this out if you want full dataset training
    train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)

    train_df["text"] = train_df["text"].apply(clean_text)

    X_train_tfidf = fit_tfidf(train_df["text"])
    X_train_lstm = fit_lstm_tokenizer(train_df["text"])
    y_train = train_df["label"]

    train_all_models(X_train_tfidf, y_train)

    build_model()
    train_model(X_train_lstm, y_train)

    st.session_state["models_trained"] = True


# ---------------- Run Detection ----------------
if st.button("Run Detection & Analysis"):
    if not st.session_state["emails"]:
        st.warning("Please generate emails first.")
    else:
        with st.spinner("Loading dataset and training models..."):
            train_pipeline()

        df = pd.DataFrame(st.session_state["emails"])

        # Clean generated emails
        df["clean_email"] = df["email"].apply(clean_text)

        # Transform generated emails using trained preprocessors
        X_test_tfidf = transform_tfidf(df["clean_email"])
        X_test_lstm = transform_lstm_sequences(df["clean_email"])

        results = []
        analysis_results = []

        for i, row in df.iterrows():
            res = detect(
                row["email"],
                X_test_tfidf[i],
                X_test_lstm[i].reshape(1, -1)
            )

            ana = analyze(row["email"])

            ml_preds = res.get("ml_models", {})

            results.append({
                "Email": row["email"][:80],
                "Logistic Regression": ml_preds.get("logistic", "N/A"),
                "Naive Bayes": ml_preds.get("naive_bayes", "N/A"),
                "SVM": ml_preds.get("svm", "N/A"),
                "Random Forest": ml_preds.get("random_forest", "N/A"),
                "LSTM": res.get("lstm", "N/A"),
                "BERT": res.get("bert", "N/A"),
                "Final Decision": res.get("final", "N/A"),
                "True Label": row.get("label", "N/A")
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
            

        st.subheader("Detection Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Analysis Results")
        analysis_df = pd.DataFrame(analysis_results)
        st.dataframe(analysis_df, use_container_width=True)
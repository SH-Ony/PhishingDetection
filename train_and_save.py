import os
from sklearn.model_selection import train_test_split

from utils import load_dataset
from preprocessing import (
    clean_text,
    fit_tfidf,
    fit_lstm_tokenizer,
    save_preprocessors
)
from models.ml_model import train_all_models, save_all_models, evaluate_all_models
from models.lstm_model import build_model, train_model, save_model_file


def main():
    os.makedirs("saved_models", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()

    # Optional smaller sample for faster training
    df = df.sample(n=min(5000, len(df)), random_state=42)

    print("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print("Fitting TF-IDF...")
    X_train_tfidf = fit_tfidf(X_train_text)

    print("Fitting tokenizer...")
    X_train_lstm = fit_lstm_tokenizer(X_train_text)

    print("Training ML models...")
    train_all_models(X_train_tfidf, y_train)

    print("Training LSTM model...")
    build_model()
    train_model(X_train_lstm, y_train, epochs=3, batch_size=32)

    print("Saving preprocessors...")
    save_preprocessors()

    print("Saving ML models...")
    save_all_models()

    print("Saving LSTM model...")
    save_model_file()

    print("Evaluating ML models...")
    from preprocessing import transform_tfidf
    X_test_tfidf = transform_tfidf(X_test_text)
    ml_metrics = evaluate_all_models(X_test_tfidf, y_test)

    print("\nTraining complete. Saved files are in /saved_models")
    print("\nML Metrics:")
    for model_name, metrics in ml_metrics.items():
        print(f"\n{model_name}")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
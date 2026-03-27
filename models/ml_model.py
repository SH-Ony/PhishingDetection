import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

models = {}


def train_all_models(X, y):
    global models

    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    for name, model in models.items():
        model.fit(X, y)


def predict_all_models(X):
    if not models:
        raise ValueError("ML models are not loaded or trained.")
    return {name: int(model.predict(X)[0]) for name, model in models.items()}


def evaluate_all_models(X_test, y_test):
    if not models:
        raise ValueError("ML models are not loaded or trained.")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    return results


def save_all_models(base_path="saved_models"):
    os.makedirs(base_path, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{base_path}/{name}.pkl")


def load_all_models(base_path="saved_models"):
    global models
    models = {
        "logistic": joblib.load(f"{base_path}/logistic.pkl"),
        "naive_bayes": joblib.load(f"{base_path}/naive_bayes.pkl"),
        "svm": joblib.load(f"{base_path}/svm.pkl"),
        "random_forest": joblib.load(f"{base_path}/random_forest.pkl")
    }
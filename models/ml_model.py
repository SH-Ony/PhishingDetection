from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

models = {}


def train_all_models(X, y, include_gradient_boosting=False):
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

    if include_gradient_boosting:
        models["gradient_boosting"] = GradientBoostingClassifier(random_state=42)

    for name, model in models.items():
        model.fit(X, y)


def predict_all_models(X):
    if not models:
        raise ValueError("ML models are not trained yet. Call train_all_models() first.")

    results = {}
    for name, model in models.items():
        results[name] = int(model.predict(X)[0])

    return results


def evaluate_all_models(X_test, y_test):
    if not models:
        raise ValueError("ML models are not trained yet. Call train_all_models() first.")

    metrics = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )
        }

    return metrics
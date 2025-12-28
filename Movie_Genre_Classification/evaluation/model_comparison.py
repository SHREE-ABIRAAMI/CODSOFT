# evaluation/model_comparison.py

import os
import sys
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
Model comparison module for Movie Genre Classification.

This script evaluates multiple trained models on a common validation set
and compares them based on predictive accuracy and inference speed.
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.data_loader import load_dataset
from utils.vectorizer import build_tfidf_vectorizer

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_data.txt")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")


def evaluate_model(model, X_val, y_val):
    """
    Measure validation accuracy and inference time for a given model.
    """
    start = time.time()
    y_pred = model.predict(X_val)
    infer_time = time.time() - start

    accuracy = accuracy_score(y_val, y_pred)
    return accuracy, infer_time


def compare_models():
    """
    Compare Naive Bayes, Logistic Regression, and SVM models
    using the same TF-IDF representation and validation split.
    """

    print("[INFO] Loading dataset...")
    texts, labels, _ = load_dataset(DATA_PATH)

    vectorizer = build_tfidf_vectorizer(min_df=2)
    X = vectorizer.fit_transform(texts)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    models = {
        "Naive Bayes": joblib.load(os.path.join(ARTIFACTS_PATH, "naive.pkl")),
        "Logistic Regression": joblib.load(os.path.join(ARTIFACTS_PATH, "logistic.pkl")),
        "SVM": joblib.load(os.path.join(ARTIFACTS_PATH, "svm.pkl")),
    }

    results = []

    print("\n[INFO] Evaluating models...\n")

    for name, model in models.items():
        acc, infer_time = evaluate_model(model, X_val, y_val)
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Inference Time (s)": round(infer_time, 4)
        })

    df = pd.DataFrame(results)
    print(df)

    return df


if __name__ == "__main__":
    compare_models()

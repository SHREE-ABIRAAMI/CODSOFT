# models/train_logistic.py

import os
import sys
import time
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


"""
Training script for Logistic Regression based movie genre classification.

This module loads the dataset, extracts TF-IDF features,
trains a Logistic Regression classifier, evaluates performance,
and saves the trained model and vectorizer as artifacts.
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.data_loader import load_dataset
from utils.vectorizer import build_tfidf_vectorizer


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_data.txt")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")


def train_logistic_regression():
    """
    Train and evaluate a Logistic Regression classifier.
    """

    print("[INFO] Loading dataset...")
    texts, labels, _ = load_dataset(DATA_PATH)
    print(f"[INFO] Total samples: {len(texts)}")

    print("[INFO] Creating TF-IDF features...")
    vectorizer = build_tfidf_vectorizer(min_df=2)
    X = vectorizer.fit_transform(texts)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print("[INFO] Training Logistic Regression model...")
    start_time = time.time()

    model = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}\n")

    print("Classification Report:\n")
    print(classification_report(y_val, y_pred, zero_division=0))

    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "logistic.pkl"))
    joblib.dump(vectorizer, os.path.join(ARTIFACTS_PATH, "tfidf.pkl"))

    print("[INFO] Model saved successfully")
    print(f"[INFO] Training Time: {training_time:.2f} seconds")


if __name__ == "__main__":
    train_logistic_regression()

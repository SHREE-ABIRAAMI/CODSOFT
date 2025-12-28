# models/train_naive.py

import os
import sys
import time
import pickle
import warnings

warnings.filterwarnings("ignore")

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier


"""
Training script for Naive Bayes based movie genre classification.

This module loads the dataset, extracts TF-IDF features,
trains a One-vs-Rest Multinomial Naive Bayes classifier,
evaluates performance, and saves the trained model.
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.data_loader import load_dataset
from utils.vectorizer import fit_transform_text


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_data.txt")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "naive.pkl")


def train_naive_bayes():
    """
    Train and evaluate a Multinomial Naive Bayes classifier.
    """

    print("[INFO] Loading dataset...")
    texts, labels, _ = load_dataset(DATA_PATH)
    print(f"[INFO] Total samples: {len(texts)}")

    print("[INFO] Creating TF-IDF features...")
    X, vectorizer = fit_transform_text(texts)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print("[INFO] Training Naive Bayes model...")
    start_time = time.time()

    model = OneVsRestClassifier(
        MultinomialNB(alpha=0.1)
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

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("[INFO] Model saved successfully")
    print(f"[INFO] Training Time: {training_time:.2f} seconds")


if __name__ == "__main__":
    train_naive_bayes()

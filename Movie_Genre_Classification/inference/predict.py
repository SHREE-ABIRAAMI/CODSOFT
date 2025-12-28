# inference/predict.py

import os
import sys
import joblib
import numpy as np


"""
Inference module for Movie Genre Classification.

This script loads a trained SVM model and TF-IDF vectorizer
to perform top-k genre predictions on user-provided movie plots.
It also optionally provides explainability for the top prediction.
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.explainability import explain_prediction


ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "svm.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_PATH, "tfidf.pkl")


def softmax(x):
    """
    Compute softmax probabilities from raw scores.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def predict_top_k(model, vectorizer, text, k=3):
    """
    Predict top-k genres with confidence scores.
    """
    X = vectorizer.transform([text])

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    top_indices = np.argsort(probs)[-k:][::-1]

    return [(classes[i], probs[i] * 100) for i in top_indices]


def main():
    """
    Interactive CLI for movie genre prediction.
    """

    print("\n=== Movie Genre Classification (Top-3 Prediction) ===\n")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    while True:
        text = input("Enter movie plot (or type 'exit'): ").strip()

        if text.lower() == "exit":
            print("\nExiting... Thank you!\n")
            break

        if len(text) < 10:
            print("Please enter a more descriptive plot.\n")
            continue

        predictions = predict_top_k(model, vectorizer, text, k=3)

        print("\nPredicted Genres:")
        for genre, conf in predictions:
            print(f"- {genre:<12} ({conf:.2f}%)")

        try:
            pred, explanation = explain_prediction(
                model, vectorizer, text, top_n=8
            )

            print("\nTop contributing words:")
            for word, score in explanation:
                print(f"{word:<15} {score}")

        except Exception as e:
            print(f"\nExplainability unavailable: {e}")

        print("\n" + "-" * 55 + "\n")


if __name__ == "__main__":
    main()

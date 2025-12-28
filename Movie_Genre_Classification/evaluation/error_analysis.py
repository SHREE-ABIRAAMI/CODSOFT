# evaluation/error_analysis.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


"""
Error analysis module for Movie Genre Classification.

This script evaluates model predictions on the training dataset to:
- quantify overall error rate
- identify commonly confused genres
- visualize confusion patterns
- inspect representative misclassified examples
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.data_loader import load_dataset

ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_data.txt")


def error_analysis():
    """
    Perform detailed error analysis on the trained SVM classifier.
    """

    print("[INFO] Loading model and vectorizer...")
    model = joblib.load(os.path.join(ARTIFACTS_PATH, "svm.pkl"))
    vectorizer = joblib.load(os.path.join(ARTIFACTS_PATH, "tfidf.pkl"))

    print("[INFO] Loading dataset...")
    texts, labels, _ = load_dataset(DATA_PATH)

    df = pd.DataFrame({
        "plot": texts,
        "genre": labels
    })

    print("[INFO] Vectorizing text...")
    X = vectorizer.transform(texts)

    print("[INFO] Generating predictions...")
    y_pred = model.predict(X)

    df["predicted_genre"] = y_pred
    df["is_correct"] = df["genre"] == df["predicted_genre"]

    total = len(df)
    incorrect = (~df["is_correct"]).sum()

    print(f"\nTotal Samples: {total}")
    print(f"Incorrect Predictions: {incorrect}")
    print(f"Error Rate: {(incorrect / total) * 100:.2f}%")

    confusion_pairs = (
        df[~df["is_correct"]]
        .groupby(["genre", "predicted_genre"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop Confused Genre Pairs:")
    print(confusion_pairs)

    top_genres = df["genre"].value_counts().head(10).index
    subset = df[df["genre"].isin(top_genres)]

    cm = confusion_matrix(
        subset["genre"],
        subset["predicted_genre"],
        labels=top_genres
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        xticklabels=top_genres,
        yticklabels=top_genres,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Confusion Matrix (Top 10 Genres)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    genre_error = (
        df.groupby("genre")["is_correct"]
        .apply(lambda x: 1 - x.mean())
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nGenres with Highest Error Rate:")
    print(genre_error)

    plt.figure(figsize=(10, 5))
    genre_error.plot(kind="bar")
    plt.title("Genres with Highest Error Rate")
    plt.ylabel("Error Rate")
    plt.xlabel("Genre")
    plt.tight_layout()
    plt.show()

    print("\nSample Misclassified Examples:\n")

    samples = df[~df["is_correct"]].sample(5, random_state=42)
    for _, row in samples.iterrows():
        print("Actual Genre   :", row["genre"])
        print("Predicted Genre:", row["predicted_genre"])
        print("Plot Snippet   :", row["plot"][:300], "...\n")
        print("-" * 80)

    error_report_path = os.path.join(ARTIFACTS_PATH, "error_analysis.csv")
    df.to_csv(error_report_path, index=False)

    print(f"\n[INFO] Error analysis report saved to: {error_report_path}")


if __name__ == "__main__":
    error_analysis()

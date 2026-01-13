import os
import joblib
from sklearn.naive_bayes import MultinomialNB

from utils.data_loader import load_data
from utils.text_preprocessor import clean_text
from utils.vectorizer import build_vectorizer


def main() -> None:
    """
    Train a Multinomial Naive Bayes model for SMS spam detection.

    Workflow:
    - Load the dataset
    - Clean and preprocess SMS messages
    - Convert text to TF-IDF features
    - Train a Multinomial Naive Bayes classifier
    - Save the trained model and vectorizer
    """
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Load and preprocess data
    df = load_data("data/spam.csv")
    df["message"] = df["message"].apply(clean_text)

    # Vectorize text
    X, _ = build_vectorizer(
        df["message"],
        "artifacts/tfidf_vectorizer.pkl",
    )
    y = df["label"]

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, "artifacts/naive_bayes_model.pkl")

    print("Naive Bayes model trained and saved successfully.")


if __name__ == "__main__":
    main()

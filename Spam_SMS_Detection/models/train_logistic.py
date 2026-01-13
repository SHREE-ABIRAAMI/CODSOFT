import os
import joblib
from sklearn.linear_model import LogisticRegression

from utils.data_loader import load_data
from utils.text_preprocessor import clean_text
from utils.vectorizer import build_vectorizer


def main() -> None:
    """
    Train a Logistic Regression model for SMS spam detection.

    Workflow:
    - Load the dataset
    - Clean and preprocess SMS text
    - Convert text to TF-IDF features
    - Train a Logistic Regression classifier
    - Persist the trained model and vectorizer
    """
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Load and preprocess data
    df = load_data("data/spam.csv")
    df["message"] = df["message"].apply(clean_text)

    # Vectorize text (consistent feature settings)
    X, _ = build_vectorizer(
        df["message"],
        "artifacts/tfidf_vectorizer.pkl",
    )
    y = df["label"]

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, "artifacts/logistic_model.pkl")

    print("Logistic Regression model trained and saved successfully.")


if __name__ == "__main__":
    main()

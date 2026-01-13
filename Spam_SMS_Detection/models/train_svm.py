import os
import joblib
from sklearn.svm import LinearSVC

from utils.data_loader import load_data
from utils.text_preprocessor import clean_text
from utils.vectorizer import build_vectorizer


def main() -> None:
    """
    Train a Support Vector Machine (LinearSVC) model for SMS spam detection.

    Workflow:
    - Load the dataset
    - Clean and preprocess SMS messages
    - Convert text to TF-IDF features
    - Train a Linear SVM classifier
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

    # Train SVM model
    model = LinearSVC()
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, "artifacts/svm_model.pkl")

    print("SVM model trained and saved successfully.")


if __name__ == "__main__":
    main()

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data
from utils.text_preprocessor import clean_text


def main() -> None:
    """
    Perform error analysis on the trained SVM spam classifier.

    Workflow:
    - Load and preprocess the dataset
    - Split data into training and testing sets
    - Load trained TF-IDF vectorizer and SVM model
    - Identify misclassified SMS messages
    - Save misclassified samples for inspection
    """
    # Load and preprocess data
    df = load_data("data/spam.csv")
    df["message"] = df["message"].apply(clean_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # Load trained artifacts
    vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
    model = joblib.load("artifacts/svm_model.pkl")

    # Vectorize test data
    X_test_vec = vectorizer.transform(X_test)

    # Generate predictions
    predictions = model.predict(X_test_vec)

    # Collect misclassified samples
    errors = pd.DataFrame(
        {
            "message": X_test.values,
            "actual_label": y_test.values,
            "predicted_label": predictions,
        }
    )

    errors = errors[errors["actual_label"] != errors["predicted_label"]]

    # Save error analysis for inspection
    errors.to_csv("artifacts/spam_error_analysis.csv", index=False)

    print(f"Total misclassified messages: {len(errors)}")
    print("Error analysis saved to artifacts/spam_error_analysis.csv")


if __name__ == "__main__":
    main()

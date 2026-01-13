import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils.data_loader import load_data
from utils.text_preprocessor import clean_text


def main() -> None:
    """
    Evaluate and compare multiple spam classification models.

    Workflow:
    - Load and preprocess the dataset
    - Split data into training and testing sets
    - Load the trained TF-IDF vectorizer
    - Evaluate Naive Bayes, Logistic Regression, and SVM models
    - Report Accuracy, Precision, Recall, and F1-score
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

    # Load vectorizer and transform test data
    vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
    X_test_vec = vectorizer.transform(X_test)

    # Load trained models
    models = {
        "Naive Bayes": joblib.load("artifacts/naive_bayes_model.pkl"),
        "Logistic Regression": joblib.load("artifacts/logistic_model.pkl"),
        "SVM": joblib.load("artifacts/svm_model.pkl"),
    }

    print("\nModel Performance Comparison:\n")

    for model_name, model in models.items():
        predictions = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(model_name)
        print(f" Accuracy : {accuracy:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall   : {recall:.4f}")
        print(f" F1-score : {f1:.4f}\n")


if __name__ == "__main__":
    main()

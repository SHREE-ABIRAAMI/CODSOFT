import sys
import joblib

from utils.text_preprocessor import clean_text


def predict_sms(message: str) -> str:
    """
    Predict whether an SMS message is spam or ham.

    This function loads the trained SVM model and TF-IDF vectorizer,
    preprocesses the input message, and returns the prediction label.

    Parameters
    ----------
    message : str
        Raw SMS text input.

    Returns
    -------
    str
        Prediction result: "SPAM" or "HAM".
    """
    # Load trained artifacts
    vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
    model = joblib.load("artifacts/svm_model.pkl")

    # Preprocess and vectorize input
    cleaned_text = clean_text(message)
    X = vectorizer.transform([cleaned_text])

    # Generate prediction
    prediction = model.predict(X)[0]
    return "SPAM" if prediction == 1 else "HAM"


def main() -> None:
    """
    Command-line interface for SMS spam prediction.

    Usage:
        python -m inference.predict "Your SMS text here"
    """
    if len(sys.argv) < 2:
        print('Usage: python -m inference.predict "Your SMS text here"')
        sys.exit(1)

    sms_text = sys.argv[1]
    result = predict_sms(sms_text)
    print("Prediction:", result)


if __name__ == "__main__":
    main()

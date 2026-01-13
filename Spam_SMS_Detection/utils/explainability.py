import joblib
import numpy as np


def show_top_spam_indicators(top_n: int = 15) -> None:
    """
    Display the most influential words for spam and ham classification.

    This function loads a trained Logistic Regression model and its
    corresponding TF-IDF vectorizer, then identifies the top features
    contributing to spam and ham predictions based on model coefficients.

    Parameters
    ----------
    top_n : int, optional
        Number of top indicators to display for each class (default: 15).

    Returns
    -------
    None
        Prints the top spam and ham indicator words to the console.
    """
    model = joblib.load("artifacts/logistic_model.pkl")
    vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")

    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = model.coef_[0]

    # Identify top features
    top_spam_words = feature_names[np.argsort(weights)[-top_n:]]
    top_ham_words = feature_names[np.argsort(weights)[:top_n]]

    print("\nTop Spam Indicators:")
    for word in reversed(top_spam_words):
        print(word)

    print("\nTop Ham Indicators:")
    for word in top_ham_words:
        print(word)


if __name__ == "__main__":
    show_top_spam_indicators()

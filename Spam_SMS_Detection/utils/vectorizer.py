import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(texts, save_path: str):
    """
    Build and persist a TF-IDF vectorizer.

    The vectorizer is configured to:
    - Use unigrams and bigrams
    - Remove English stopwords
    - Limit the feature space to the top 3,000 terms

    Parameters
    ----------
    texts : iterable of str
        Collection of preprocessed text documents.
    save_path : str
        File path to save the trained vectorizer.

    Returns
    -------
    tuple
        X : sparse matrix
            TF-IDF feature matrix.
        vectorizer : TfidfVectorizer
            Fitted TF-IDF vectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, save_path)

    return X, vectorizer


def load_vectorizer(path: str) -> TfidfVectorizer:
    """
    Load a previously saved TF-IDF vectorizer.

    Parameters
    ----------
    path : str
        File path to the saved vectorizer.

    Returns
    -------
    TfidfVectorizer
        Loaded TF-IDF vectorizer instance.
    """
    return joblib.load(path)

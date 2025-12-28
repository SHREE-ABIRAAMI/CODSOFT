import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features=60000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.85
):
    """
    Construct a TF-IDF vectorizer optimized for movie plot text.

    The configuration balances vocabulary richness with noise reduction
    to improve downstream classification performance.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True
    )


def fit_transform_text(corpus, save_path=None):
    """
    Fit the TF-IDF vectorizer on the training corpus and transform the text.
    """
    vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(corpus)

    if save_path:
        joblib.dump(vectorizer, save_path)

    return X, vectorizer


def transform_text(corpus, vectorizer_path):
    """
    Transform input text using a previously fitted TF-IDF vectorizer.
    """
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer.transform(corpus)

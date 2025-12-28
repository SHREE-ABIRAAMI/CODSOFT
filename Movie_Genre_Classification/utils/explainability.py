# utils/explainability.py

import numpy as np


def _get_linear_model(model):
    """
    Retrieve the underlying linear estimator from wrapped models.

    Supports direct linear models as well as calibrated classifiers.
    """

    if hasattr(model, "coef_"):
        return model

    if hasattr(model, "estimator_"):
        return model.estimator_

    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator

    raise ValueError("Provided model does not expose linear coefficients")


def explain_prediction(model, vectorizer, text, top_n=10):
    """
    Generate a feature-level explanation for a single prediction.

    Uses linear model coefficients to identify the most influential
    terms contributing to the predicted class.
    """

    linear_model = _get_linear_model(model)

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = linear_model.coef_[list(linear_model.classes_).index(pred)]

    scores = X.toarray()[0] * coef
    top_indices = np.argsort(scores)[-top_n:][::-1]

    explanation = [
        (feature_names[i], round(scores[i], 4))
        for i in top_indices if scores[i] > 0
    ]

    return pred, explanation

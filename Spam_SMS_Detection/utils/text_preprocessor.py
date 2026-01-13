import re
import string


def clean_text(text: str) -> str:
    """
    Clean SMS text for NLP modeling.

    The cleaning process includes:
    - Converting text to lowercase
    - Removing URLs
    - Removing numeric characters
    - Removing punctuation
    - Normalizing whitespace

    Parameters
    ----------
    text : str
        Raw SMS text input.

    Returns
    -------
    str
        Cleaned text suitable for NLP feature extraction.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

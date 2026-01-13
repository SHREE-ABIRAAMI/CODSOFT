import pandas as pd


def load_data(path: str = "data/spam.csv") -> pd.DataFrame:
    """
    Load and clean the Kaggle SMS Spam dataset.

    This function performs the following steps:
    - Reads the dataset using Latin-1 encoding
    - Retains only the relevant columns
    - Renames columns for clarity
    - Encodes labels (ham → 0, spam → 1)

    Parameters
    ----------
    path : str, optional
        Path to the CSV dataset file (default: "data/spam.csv").

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing:
        - label   : int (0 = ham, 1 = spam)
        - message : str (SMS text)
    """
    df = pd.read_csv(path, encoding="latin-1")

    # Retain only required columns
    df = df[["v1", "v2"]]

    # Rename columns for clarity
    df.columns = ["label", "message"]

    # Encode target labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    return df

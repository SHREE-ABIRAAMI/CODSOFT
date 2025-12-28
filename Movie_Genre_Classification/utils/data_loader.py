import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def load_dataset(data_path, multilabel=False):
    """
    Load and preprocess the movie genre classification dataset.

    Parameters:
        data_path (str): Path to the dataset file.
        multilabel (bool): Enable multi-label genre processing.

    Returns:
        plots (list): Movie plot summaries.
        labels (list or ndarray): Genre labels.
        mlb (MultiLabelBinarizer or None): Encoder used for multilabel data.
    """

    records = []

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) != 4:
                continue

            _, title, genre, plot = parts

            if not plot.strip() or not genre.strip():
                continue

            records.append((title, genre.lower(), plot))

    df = pd.DataFrame(records, columns=["title", "genre", "plot"])

    if multilabel:
        # Convert comma-separated genres into multi-hot encoded labels
        df["genre"] = df["genre"].apply(lambda x: x.split(","))
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df["genre"])
        return df["plot"].tolist(), y, mlb

    return df["plot"].tolist(), df["genre"].tolist(), None


def dataset_summary(X, y):
    """Print a brief overview of the dataset."""
    print("Total samples:", len(X))
    print("Sample plot:\n", X[0][:200], "...")

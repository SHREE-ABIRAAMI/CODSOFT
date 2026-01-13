import pandas as pd

from utils.data_loader import load_data


def run_eda() -> None:
    """
    Perform basic exploratory data analysis (EDA) on the SMS spam dataset.

    This analysis includes:
    - Dataset shape
    - Class distribution and percentages
    - Message length statistics by class
    """
    df = load_data("data/spam.csv")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nClass Distribution (0 = Ham, 1 = Spam):")
    print(df["label"].value_counts())

    print("\nPercentage Distribution:")
    print(df["label"].value_counts(normalize=True) * 100)

    # Message length analysis
    df["message_length"] = df["message"].str.len()

    print("\nMessage Length Statistics:")
    print(df.groupby("label")["message_length"].describe())


if __name__ == "__main__":
    run_eda()

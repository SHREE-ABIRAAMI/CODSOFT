import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

"""
Exploratory Data Analysis (EDA) for Movie Genre Classification.

This script loads the raw training dataset and performs basic analysis
to understand genre distribution, plot length characteristics, and
frequently occurring words in movie plots.
"""

print("\n=== EDA SCRIPT STARTED ===\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "train_data.txt")

print("Base Directory   :", BASE_DIR)
print("Project Directory:", PROJECT_DIR)
print("Dataset Path     :", DATA_PATH)

if not os.path.exists(DATA_PATH):
    print("\nERROR: Dataset file not found.")
    sys.exit(1)


"""
Dataset loading and basic preprocessing.

Each record consists of a title, genre, and plot summary.
Genres are normalized to lowercase for consistency.
"""
records = []

with open(DATA_PATH, encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            _, title, genre, plot = parts
            records.append((title, genre.lower(), plot))

df = pd.DataFrame(records, columns=["title", "genre", "plot"])

print("\nDATASET LOADED SUCCESSFULLY")
print("Total Samples :", len(df))
print("Total Genres  :", df["genre"].nunique())


"""
Genre distribution analysis.

Provides insight into class imbalance and dominant genres.
"""
print("\nTop 10 Genres:")
genre_counts = df["genre"].value_counts()
print(genre_counts.head(10))

plt.figure(figsize=(10, 6))
genre_counts.head(15).plot(kind="barh")
plt.title("Top 15 Movie Genres")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()


"""
Plot length analysis.

Analyzes the distribution of word counts in movie plots
to guide vectorization and model design choices.
"""
df["plot_length"] = df["plot"].apply(lambda x: len(x.split()))

print("\nPlot Length Statistics:")
print(df["plot_length"].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["plot_length"], bins=50)
plt.title("Plot Length Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


"""
Common word frequency analysis.

Identifies frequently occurring words across all plots,
useful for understanding vocabulary and noise patterns.
"""
all_words = " ".join(df["plot"]).lower().split()
common_words = Counter(all_words).most_common(20)

print("\nTop 20 Most Common Words in Plots:")
for word, count in common_words:
    print(f"{word}: {count}")

print("\n=== EDA COMPLETED SUCCESSFULLY ===\n")

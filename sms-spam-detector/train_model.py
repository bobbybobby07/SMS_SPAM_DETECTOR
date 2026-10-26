"""Train a Multinomial Naive Bayes SMS spam classifier and save it to model.pkl.

Usage:
    python train_model.py                 # downloads the dataset if needed
    python train_model.py --data my.tsv   # use your own tab-separated file (label<TAB>text)
"""
import argparse
import os
import pickle
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/"
    "pycon-2016-tutorial/master/data/sms.tsv"
)
DEFAULT_DATA = os.path.join("data", "sms.tsv")
MODEL_PATH = "model.pkl"


def load_data(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df = df.dropna().drop_duplicates()
    df["target"] = (df["label"].str.lower() == "spam").astype(int)  # 1 = spam
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=MODEL_PATH)
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} messages ({df['target'].sum()} spam).")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    model = Pipeline(
        [
            ("vectorizer", CountVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["not spam", "spam"]))

    # Retrain on all data for the final model
    model.fit(df["text"], df["target"])
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()

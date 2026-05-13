import os
import pandas as pd
import numpy as np
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    path = kagglehub.dataset_download(
        "johnsmith88/heart-disease-dataset"
    )

    print("Dataset Path:", path)

    files = os.listdir(path)
    print("Files:", files)

    df = pd.read_csv(f"{path}/heart.csv")

    return df


def preprocessing(df):

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train.values

    test_df = pd.DataFrame(X_test)
    test_df["target"] = y_test.values

    return train_df, test_df


def save_dataset(train_df, test_df):

    os.makedirs("dataset_preprocessing", exist_ok=True)

    train_df.to_csv(
        "dataset_preprocessing/train_processed.csv",
        index=False
    )

    test_df.to_csv(
        "dataset_preprocessing/test_processed.csv",
        index=False
    )

    print("Dataset preprocessing berhasil disimpan.")


def main():

    print("Loading dataset...")

    df = load_dataset()

    print("Preprocessing dataset...")

    train_df, test_df = preprocessing(df)

    print("Saving dataset...")

    save_dataset(train_df, test_df)

    print("Selesai.")


if __name__ == "__main__":
    main()

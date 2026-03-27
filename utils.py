import os
import pandas as pd
import kagglehub


def load_dataset():
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")

    file_path = os.path.join(path, "phishing_email.csv")
    df = pd.read_csv(file_path)

    # Rename to standard column name used in the project
    df = df.rename(columns={"text_combined": "text"})

    # Keep only required columns
    df = df[["text", "label"]].copy()

    # Clean missing values
    df.dropna(inplace=True)

    # Make sure types are correct
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df
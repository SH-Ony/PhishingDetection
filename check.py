import os
import pandas as pd
import kagglehub


def find_csv_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def inspect_dataframe(df, file_name):
    print("\n" + "=" * 80)
    print(f"FILE: {file_name}")
    print("=" * 80)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample values from each column:")
    for col in df.columns:
        print(f"\n--- Column: {col} ---")
        print(df[col].dropna().astype(str).head(5).tolist())


def suggest_mapping(df):
    print("\n" + "=" * 80)
    print("SUGGESTED COLUMN CHECK")
    print("=" * 80)

    possible_text_cols = []
    possible_label_cols = []

    for col in df.columns:
        col_lower = col.lower()

        if any(keyword in col_lower for keyword in ["text", "email", "body", "message", "content"]):
            possible_text_cols.append(col)

        if any(keyword in col_lower for keyword in ["label", "type", "class", "category", "target"]):
            possible_label_cols.append(col)

    print("\nPossible text columns:")
    print(possible_text_cols if possible_text_cols else "None found automatically")

    print("\nPossible label columns:")
    print(possible_label_cols if possible_label_cols else "None found automatically")

    for col in possible_label_cols:
        print(f"\nUnique values in label-like column '{col}':")
        try:
            print(df[col].dropna().unique()[:20])
        except Exception as e:
            print(f"Could not inspect unique values: {e}")


def try_basic_standardization(df):
    """
    This does NOT overwrite your main project.
    It only tries to produce a temporary standardized version:
    text, label
    """
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()

        if text_col is None and any(k in col_lower for k in ["text", "email", "body", "message", "content"]):
            text_col = col

        if label_col is None and any(k in col_lower for k in ["label", "type", "class", "category", "target"]):
            label_col = col

    if text_col is None or label_col is None:
        print("\nCould not automatically standardize this file.")
        print("You may need to manually choose the correct columns.")
        return

    temp_df = df[[text_col, label_col]].copy()
    temp_df.columns = ["text", "label"]

    if temp_df["label"].dtype == object:
        temp_df["label"] = temp_df["label"].astype(str).str.strip().str.lower()

        label_map = {
            "phishing": 1,
            "legitimate": 0,
            "legit": 0,
            "ham": 0,
            "spam": 1,
            "safe": 0,
            "malicious": 1
        }

        temp_df["label"] = temp_df["label"].map(label_map)

    print("\n" + "=" * 80)
    print("TEMPORARY STANDARDIZED VERSION")
    print("=" * 80)
    print(temp_df.head())
    print("\nLabel unique values after mapping:")
    print(temp_df["label"].dropna().unique())
    print("\nMissing after mapping:")
    print(temp_df.isnull().sum())


def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    print(f"Dataset downloaded to: {path}")

    csv_files = find_csv_files(path)

    if not csv_files:
        print("No CSV files found in the dataset folder.")
        return

    print("\nCSV files found:")
    for i, file_path in enumerate(csv_files, start=1):
        print(f"{i}. {file_path}")

    for file_path in csv_files:
        try:
            print("\n" + "#" * 100)
            print(f"READING: {file_path}")
            print("#" * 100)

            df = pd.read_csv(file_path)
            inspect_dataframe(df, os.path.basename(file_path))
            suggest_mapping(df)
            try_basic_standardization(df)

        except Exception as e:
            print(f"\nFailed to read {file_path}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
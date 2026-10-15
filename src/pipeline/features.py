"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""


from pathlib import Path
from category_encoders import TargetEncoder
from joblib import dump 
import pandas as pd


PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature functions ----------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    # place after date for readability
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df


#   Creates a frequency encoding.
#   Fit only on train, then applied to test.
def frequency_encode(train: pd.DataFrame, test: pd.DataFrame, col: str):
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    test[f"{col}_freq"] = test[col].map(freq_map).fillna(0)
    return train, test, freq_map


# Uses target encoding (replace category with average of target variable).
# Fitted only on train (prevents leakage).
def target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`, consistently name as <col>_encoded.
    For city_full → city_full_encoded (keeps schema aligned with inference).
    """
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[col], train[target])
    test[encoded_col] = te.transform(test[col])
    return train, test, te



def drop_unused_columns(train: pd.DataFrame, test: pd.DataFrame):
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")
    return train, test

# ---------- pipeline ----------

#   Handles full pipeline: 
#   reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def engineer_features(
    train_path: Path | str | None = None,
    test_path: Path | str | None = None,
    new_data_path: Path | str | None = None,
    output_dir: Path | str = FEATURES_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if train_path is None:
        train_path = PROCESSED_DIR / "processed_train_data.csv"
    if test_path is None:
        test_path = PROCESSED_DIR / "processed_test_data.csv"
    if new_data_path is None:
        new_data_path = PROCESSED_DIR / "processed_new_data.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    new_data_df = pd.read_csv(new_data_path)

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Test date range:", test_df["date"].min(), "to", test_df["date"].max())
    print("New data date range:", new_data_df["date"].min(), "to", new_data_df["date"].max())

    # Date features
    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)
    new_data_df = add_date_features(new_data_df)

    # Frequency encode zipcode (fit on train only)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, test_df, freq_map = frequency_encode(train_df, test_df, "zipcode")
        new_data_df["zipcode_freq"] = new_data_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")   # save mapping

    # Target encode city_full (fit on train only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, test_df, target_encoder = target_encode(train_df, test_df, "city_full", "price")
        new_data_df["city_full_encoded"] = target_encoder.transform(new_data_df["city_full"])
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")  # save encoder

    # Drop leakage / raw categoricals
    train_df, test_df = drop_unused_columns(train_df, test_df)
    new_data_df, _ = drop_unused_columns(new_data_df.copy(), new_data_df.copy())

    # Save engineered data
    out_train_path = output_dir / "featured_train_data.csv"
    out_test_path = output_dir / "featured_test_data.csv"
    out_new_data_path = output_dir / "featured_new_data.csv"
    train_df.to_csv(out_train_path, index=False)
    test_df.to_csv(out_test_path, index=False)
    new_data_df.to_csv(out_new_data_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Test  shape:", test_df.shape)
    print("   New data shape:", new_data_df.shape)
    print("   Encoders saved to models/")

    return train_df, test_df, new_data_df, freq_map, target_encoder


if __name__ == "__main__":
    engineer_features()

"""
Load and time-split the raw dataset.

Production writes to ../data/raw/

"""


import pandas as pd
from pathlib import Path


DATA_DIR = Path("../data/raw")


def load_split_raw_data(path: Path | str = DATA_DIR,
                        output_dir: Path | str = DATA_DIR) -> pd.DataFrame:

    """Load raw housing dataset and save to the output dir."""

    df = pd.read_csv(Path(path))

    # Convert to datetime object
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Timestaamp to split data
    cutoff_date_test = pd.Timestamp("2020-01-01")     # Test starts
    cutoff_date_new = pd.Timestamp("2022-01-01")  # New starts

    # Splits
    train_df = df[df["date"] < cutoff_date_test]
    test_df = df[(df["date"] >= cutoff_date_test) & (df["date"] < cutoff_date_new)]
    new_df = df[df["date"] >= cutoff_date_new]

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)
    new_df.to_csv(outdir / "new.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}, New: {new_df.shape}")

    return train_df, test_df, new_df
    



if __name__ == "__main__":
    load_split_raw_data("../data/raw/housing.csv")

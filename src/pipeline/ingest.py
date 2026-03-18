import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw housing dataset from CSV."""
    return pd.read_csv(Path(path))

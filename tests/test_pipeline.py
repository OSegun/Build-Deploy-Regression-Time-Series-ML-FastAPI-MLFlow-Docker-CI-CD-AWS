import pandas as pd
from src.pipeline.preprocess import preprocess


def test_preprocess_returns_dataframe():
    df = pd.DataFrame({"price": [100000, 200000], "bedrooms": [2, 3]})
    result = preprocess(df)
    assert isinstance(result, pd.DataFrame)

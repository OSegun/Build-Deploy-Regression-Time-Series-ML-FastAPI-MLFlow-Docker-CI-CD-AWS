import pandas as pd
from src.pipeline.features import engineer_features


def test_engineer_features_returns_dataframe():
    df = pd.DataFrame({"price": [100000], "sqft_living": [1500]})
    result = engineer_features(df)
    assert isinstance(result, pd.DataFrame)

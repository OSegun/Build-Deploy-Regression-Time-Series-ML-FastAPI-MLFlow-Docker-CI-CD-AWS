import pandas as pd
import pytest
from pathlib import Path


from src.pipeline.features import (add_date_features, frequency_encode, 
                                   target_encode, drop_unused_columns, engineer_features)      
from src.pipeline.ingest import load_split_raw_data
from src.pipeline.preprocess import (preprocess, preprocess_split, drop_duplicates, 
                                     remove_outliers, clean_and_merge, normalize_city)

# def test_engineer_features_returns_dataframe():
#     df = pd.DataFrame({"price": [100000], "sqft_living": [1500]})
#     result = engineer_features(df)
#     assert isinstance(result, pd.DataFrame)


# =========================
# load.py – unit test
# =========================
# Confirms time-based splitting works.
def test_load_and_split_data_creates_splits(tmp_path):
    dummy_path = tmp_path / "raw.csv"
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        "zipcode": [1000, 2000, 1000, 2000, 3000, 4000],
        "city_full": ["A", "B", "A", "B", "C", "D"],
    })
    df.to_csv(dummy_path, index=False)

    train, test, new = load_split_raw_data(path=str(dummy_path), output_dir=tmp_path)

    assert not train.empty and not test.empty and not new.empty
    assert train["date"].max() < pd.to_datetime("2020-01-01")
    assert test["date"].min() >= pd.to_datetime("2020-01-01")
    assert new["date"].min() >= pd.to_datetime("2022-01-01")
    assert (tmp_path / "train.csv").exists()
    print("✅ Data splitting test passed")


# =========================
# preprocess.py – unit tests
# =========================
# Confirms preprocessing functions behave as intended.
def test_remove_outliers_drops_high_prices():
    df = pd.DataFrame({"median_list_price": [100_000, 500_000, 20_000_000]})
    cleaned = remove_outliers(df)
    assert cleaned["median_list_price"].max() <= 19_000_000
    print("✅ Outlier removal test passed")


def test_drop_duplicates_removes_dupes():
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"],
        "year": [2020, 2020],
        "median_list_price": [100, 100]
    })
    cleaned = drop_duplicates(df)
    assert cleaned.shape[0] == 0
    print("✅ Duplicate removal test passed")


def test_clean_and_merge_skips_when_city_missing():
    df = pd.DataFrame({"date": ["2020-01-01"], "price": [100]})
    result = clean_and_merge(df, metros_path=None)  # should skip gracefully
    assert "date" in result.columns and "price" in result.columns
    print("✅ Clean-and-merge (no city_full) passed")


# =========================
# feature_engineering – unit tests
# =========================
# Confirms feature functions create consistent numeric features and avoid leakage.
def test_add_date_features_extracts_parts():
    df = pd.DataFrame({"date": ["2020-01-15"]})
    df = add_date_features(df)
    assert df.loc[0, "year"] == 2020 and df.loc[0, "month"] == 1 and df.loc[0, "quarter"] == 1
    print("✅ Date feature extraction test passed")


def test_frequency_encode_counts_values():
    train = pd.DataFrame({"zipcode": [1000, 1000, 2000]})
    test = pd.DataFrame({"zipcode": [1000, 3000]})
    train, test, freq_map = frequency_encode(train, test, "zipcode")
    assert train["zipcode_freq"].tolist() == [2, 2, 1]
    assert test["zipcode_freq"].tolist() == [2, 0]
    assert isinstance(freq_map, pd.Series)
    print("✅ Frequency encoding test passed")


def test_target_encode_applies_mapping():
    train = pd.DataFrame({"city_full": ["A", "B", "A"], "price": [100, 200, 300]})
    test = pd.DataFrame({"city_full": ["A", "B"]})
    train, test, te = target_encode(train, test, "city_full", "price")
    assert "city_full_encoded" in train.columns
    assert test["city_full_encoded"].notnull().all()
    assert te is not None
    print("✅ Target encoding test passed")


def test_drop_unused_columns_removes_leakage():
    tr = pd.DataFrame({
        "date": ["2020-01-01"], "city_full": ["A"], "zipcode": [1000],
        "median_sale_price": [200], "price": [300]
    })
    ev = tr.copy()
    tr2, ev2 = drop_unused_columns(tr, ev)
    for col in ["date", "city_full", "zipcode", "median_sale_price"]:
        assert col not in tr2.columns and col not in ev2.columns
    assert "price" in tr2.columns
    print("✅ Drop unused columns test passed")


# =========================
# integration test
# =========================
# Confirms the whole feature pipeline works together.
def test_full_pipeline_integration(tmp_path):
    raw = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        "zipcode": [1000, 2000, 1000, 2000, 1000, 2000],
        "city_full": ["A", "B", "A", "B", "A", "B"],
        "median_list_price": [150_000]*6
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    train, test, new = load_split_raw_data(path=str(raw_path), output_dir=tmp_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(exist_ok=True)
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    new.to_csv(tmp_path / "new_data.csv", index=False)       

    preprocess_split("train", raw_dir=tmp_path, processed_dir=processed_dir, metros_path=None)
    preprocess_split("test", raw_dir=tmp_path, processed_dir=processed_dir, metros_path=None)
    preprocess_split("new_data", raw_dir=tmp_path, processed_dir=processed_dir, metros_path=None)

    out_train, out_test, out_new, freq_map, te = engineer_features(
        train_path=processed_dir / "processed_train_data.csv",
        test_path=processed_dir / "processed_test_data.csv",
        new_data_path=processed_dir / "processed_new_data.csv",
        output_dir=processed_dir,
    )


    assert {"year", "zipcode_freq", "city_full_encoded"}.issubset(out_train.columns)
    assert {"year", "zipcode_freq", "city_full_encoded"}.issubset(out_test.columns)
    assert freq_map is not None
    assert te is not None
    print("✅ Full pipeline integration test passed")
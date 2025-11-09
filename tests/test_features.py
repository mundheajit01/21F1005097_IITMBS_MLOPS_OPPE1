import pandas as pd

def test_rolling_avg_not_null():
    df = pd.read_parquet("data/processed/features.parquet")
    assert df['rolling_avg_10'].isnull().sum() == 0

def test_volume_sum_positive():
    df = pd.read_parquet("data/processed/features.parquet")
    assert (df['volume_sum_10'] >= 0).all()

def test_target_binary():
    df = pd.read_parquet("data/processed/features.parquet")
    assert df['target'].isin([0, 1]).all()

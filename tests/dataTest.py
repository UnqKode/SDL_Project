import numpy as np
import pandas as pd

def test_no_missing_values(sample_data:pd.DataFrame):
    assert not sample_data.isnull().values.any()

def test_features_shape(data:np.ndarray):
    assert data.ndim == 2
    assert data.shape[1] == 128
    assert data.shape[2] == 6

def test_train_test_split(train, test):
    train_ids = set(train["id"])
    test_ids = set(test["id"])
    assert train_ids.isdisjoint(test_ids)


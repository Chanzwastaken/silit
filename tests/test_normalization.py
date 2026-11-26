"""Tests for the StandardScaler-based landmark normalisation.

The scaler must be fitted on the training split only. Fitting on the
combined data would leak validation/test statistics into training and
silently inflate the reported accuracy.
"""

import pickle

import numpy as np
import pytest

from dataset_loader import DatasetLoader


@pytest.fixture
def split():
    """Three feature blocks drawn from one distribution."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(300, 63)).astype(np.float32)
    return data[:200], data[200:250], data[250:]


def test_train_split_is_standardised(split):
    X_train, X_val, X_test = split
    loader = DatasetLoader()

    train_norm, _, _ = loader.normalize_features(X_train, X_val, X_test)

    assert np.allclose(train_norm.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(train_norm.std(axis=0), 1.0, atol=1e-5)


def test_scaler_is_fitted_on_train_only(split):
    """scaler.mean_/scale_ must come from the train split, not all data."""
    X_train, X_val, X_test = split
    loader = DatasetLoader()

    loader.normalize_features(X_train, X_val, X_test)

    assert np.allclose(loader.scaler.mean_, X_train.mean(axis=0))
    assert np.allclose(loader.scaler.scale_, X_train.std(axis=0))


def test_validation_uses_training_statistics(split):
    X_train, X_val, X_test = split
    loader = DatasetLoader()

    _, val_norm, _ = loader.normalize_features(X_train, X_val, X_test)

    expected = (X_val - X_train.mean(axis=0)) / X_train.std(axis=0)
    assert np.allclose(val_norm, expected, atol=1e-4)


def test_shapes_are_preserved(split):
    X_train, X_val, X_test = split
    loader = DatasetLoader()

    train_norm, val_norm, test_norm = loader.normalize_features(
        X_train, X_val, X_test
    )

    assert train_norm.shape == X_train.shape
    assert val_norm.shape == X_val.shape
    assert test_norm.shape == X_test.shape


def test_constant_feature_does_not_divide_by_zero():
    """A zero-variance column must produce zeros, not NaN/inf."""
    X = np.ones((10, 3), dtype=np.float32)
    loader = DatasetLoader()

    train_norm, _, _ = loader.normalize_features(X, X, X)

    assert np.isfinite(train_norm).all()
    assert np.allclose(train_norm, 0.0)


def test_save_scaler_round_trip(tmp_path, split):
    """The scaler pickled for inference must reproduce the same output."""
    X_train, X_val, X_test = split
    loader = DatasetLoader()
    loader.normalize_features(X_train, X_val, X_test)

    path = tmp_path / 'scaler.pkl'
    loader.save_scaler(path)

    with open(path, 'rb') as handle:
        restored = pickle.load(handle)

    assert np.allclose(restored.transform(X_val), loader.scaler.transform(X_val))

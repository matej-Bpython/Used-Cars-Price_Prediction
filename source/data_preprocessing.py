"""Preprocessing pipeline used by the regression models."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

TARGET = "priceUSD"

NUMERIC_FEATURES = [
    "year", "mileage(kilometers)", "volume(cm3)",
    "car_age", "mileage_per_year", "engine_volume_liters",
    "is_newer_car", "is_high_mileage",
]

CATEGORICAL_FEATURES = [
    "make", "model", "condition", "fuel_type", "color",
    "transmission", "drive_unit", "segment", "brand_model",
]


def split_features_target(df: pd.DataFrame):
    """Return X and y after feature engineering has already been applied."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_onehot_preprocessor() -> ColumnTransformer:
    """Preprocessor for linear/tree ensemble models."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_ordinal_preprocessor() -> ColumnTransformer:
    """Preprocessor for HistGradientBoostingRegressor."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

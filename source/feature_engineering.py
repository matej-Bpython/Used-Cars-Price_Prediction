"""Feature engineering for car price prediction."""

from __future__ import annotations

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Create interpretable derived features."""
    data = df.copy()

    data["car_age"] = (reference_year - data["year"]).clip(lower=0)
    data["mileage_per_year"] = (
        data["mileage(kilometers)"] / data["car_age"].replace(0, 1)
    )
    data["engine_volume_liters"] = data["volume(cm3)"] / 1000
    data["is_newer_car"] = (data["year"] >= 2015).astype(int)
    data["is_high_mileage"] = (data["mileage(kilometers)"] >= 300_000).astype(int)
    data["brand_model"] = data["make"].fillna("unknown").astype(str) + "_" + data["model"].fillna("unknown").astype(str)

    return data


if __name__ == "__main__":
    from data_cleaning import load_and_clean
    df = add_features(load_and_clean())
    print("=" * 60)
    print("Feature engineering completed.")
    print("=" * 60)
    print("\nNew features:")
    print([
        "car_age", "mileage_per_year", "engine_volume_liters",
        "is_newer_car", "is_high_mileage", "brand_model"
    ])
    print("=" * 110)
"""Data cleaning utilities for the car price prediction project."""

from __future__ import annotations

import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    "make", "model", "priceUSD", "year", "condition",
    "mileage(kilometers)", "fuel_type", "volume(cm3)", "color",
    "transmission", "drive_unit", "segment",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw car data using transparent, deterministic rules."""
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    column_map = {c.lower(): c for c in EXPECTED_COLUMNS}
    rename_map = {}
    for col in data.columns:
        if col.lower() in column_map:
            rename_map[col] = column_map[col.lower()]
    data = data.rename(columns=rename_map)

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Convert numeric fields safely.
    for col in ["priceUSD", "year", "mileage(kilometers)", "volume(cm3)"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Standardize text fields.
    text_columns = [
        "make", "model", "condition", "fuel_type", "color",
        "transmission", "drive_unit", "segment"
    ]
    for col in text_columns:
        data[col] = data[col].astype("object").apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
        data[col] = data[col].replace({"": np.nan, "nan": np.nan, "none": np.nan, pd.NA: np.nan})

    # Remove rows that cannot represent a meaningful supervised example.
    data = data.dropna(subset=["priceUSD", "year", "mileage(kilometers)"])

    # The dataset contains a small number of implausible observations.
    # These limits are deliberately broad so that valid used cars are retained.
    data = data[data["priceUSD"].between(100, 200_000)]
    data = data[data["year"].between(1980, 2019)]
    data = data[data["mileage(kilometers)"].between(0, 1_000_000)]
    data = data[
        data["volume(cm3)"].isna() | data["volume(cm3)"].between(500, 8_000)
    ]

    return data.reset_index(drop=True)


def load_and_clean(path: str = "data/cars.csv") -> pd.DataFrame:
    """Load the raw CSV file and return the cleaned dataframe."""
    return clean_data(pd.read_csv(path))


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Cleaned dataset shape: {df.shape}")
    print("\nMissing values after cleaning:")
    print(df.isna().sum())

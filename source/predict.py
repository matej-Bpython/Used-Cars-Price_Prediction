"""Make a price prediction for one car using the saved model."""

from __future__ import annotations
import sys
from pathlib import Path
import joblib
import pandas as pd

# This code dynamically locates the project's root directory and inserts it into
# Python's module search path (sys.path), ensuring that script imports work seamlessly regardless of where the file is executed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = "models/car_price_model.joblib"

def predict_price(car: dict) -> float:
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([car])
    from source.feature_engineering import add_features
    row = add_features(row)
    return float(model.predict(row)[0])


if __name__ == "__main__":
    example_car = {
        "make": "volkswagen",
        "model": "passat",
        "year": 2016,
        "condition": "with mileage",
        "mileage(kilometers)": 216000,
        "fuel_type": "diesel",
        "volume(cm3)": 1600,
        "color": "white",
        "transmission": "automatic",
        "drive_unit": "front-wheel drive",
        "segment": "C",
    }

    prediction = predict_price(example_car)
    print("=" * 60)
    print(f"Predicted market price: ${prediction:,.0f}")
    print("=" * 60)

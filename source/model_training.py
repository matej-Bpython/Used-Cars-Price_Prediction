"""Train and save the selected final regression model."""

from __future__ import annotations
import sys
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# This code dynamically locates the project's root directory and inserts it into
# Python's module search path (sys.path), ensuring that script imports work seamlessly regardless of where the file is executed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.data_cleaning import load_and_clean
from source.feature_engineering import add_features
from source.data_preprocessing import split_features_target, build_ordinal_preprocessor


RANDOM_STATE = 42


def train_final_model():
    df = add_features(load_and_clean())
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("preprocessor", build_ordinal_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=60,
            max_depth=18,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/car_price_model.joblib")

    # Save the exact test split for reproducible evaluation.
    test_export = X_test.copy()
    test_export["priceUSD"] = y_test.values
    test_export.to_csv("reports/test_set.csv", index=False)

    print("=" * 60)
    print("Final model training completed.")
    print("=" * 60)
    print("Model saved to: 'models/car_price_model.joblib'")
    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")


if __name__ == "__main__":
    train_final_model()

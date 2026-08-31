"""Compare four regression algorithms on the same train/test split."""

from __future__ import annotations

import sys
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# This code dynamically locates the project's root directory and inserts it into
# Python's module search path (sys.path), ensuring that script imports work seamlessly regardless of where the file is executed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.data_cleaning import load_and_clean
from source.feature_engineering import add_features
from source.data_preprocessing import (
    split_features_target,
    build_onehot_preprocessor,
    build_ordinal_preprocessor,
)

RANDOM_STATE = 42


def compare_models():
    df = add_features(load_and_clean())
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", build_onehot_preprocessor()),
            ("regressor", LinearRegression(n_jobs=-1)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_ordinal_preprocessor()),
            ("regressor", DecisionTreeRegressor(
                max_depth=25, min_samples_leaf=2, random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_ordinal_preprocessor()),
            ("regressor", RandomForestRegressor(
                n_estimators=60, max_depth=18, min_samples_leaf=3,
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", build_ordinal_preprocessor()),
            ("regressor", HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.08, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=RANDOM_STATE
            )),
        ]),
    }

    rows = []
    for name, model in models.items():
        print(f"\nTraining: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": mse ** 0.5,
            "R2": r2_score(y_test, predictions),
            "Training Time (seconds)": time.time() - start,
        })

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/model_comparison.csv", index=False)

    print("\nModel comparison")
    print("=" * 100)
    print(results.round(3).to_string(index=False))
    print("=" * 100)
    print("\nThe model with the lowest MAE is the recommended model.")
    print("=" * 80)

    return results


if __name__ == "__main__":
    compare_models()

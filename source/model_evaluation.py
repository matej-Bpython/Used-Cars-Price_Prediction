"""Evaluate the saved final model."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model_path: str = "models/car_price_model.joblib",
    test_path: str = "reports/test_set.csv",
):
    model = joblib.load(model_path)
    test = pd.read_csv(test_path)

    X_test = test.drop(columns=["priceUSD"])
    y_test = test["priceUSD"]
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame([{
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }])

    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/final_model_metrics.csv", index=False)

    examples = pd.DataFrame({
        "Actual Price (USD)": y_test,
        "Predicted Price (USD)": predictions,
    })
    examples["Absolute Error (USD)"] = (
        examples["Actual Price (USD)"] - examples["Predicted Price (USD)"]
    ).abs()
    examples.head(10).to_csv("reports/prediction_examples.csv", index=False)

    print("=" * 60)
    print("Final model evaluation:")
    print("=" * 60)
    print(f"MAE:  ${mae:,.2f}")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}")
    print("=" * 60)
    print("\nExample predictions:")
    print("=" * 80)
    print(examples.head(10).round(2).to_string(index=False))

    return results, examples


if __name__ == "__main__":
    evaluate_model()

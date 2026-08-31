# Car Price Prediction – Regression Project

Author: Matej Brnjić

Date: 31.08.2026


## 1. Project Overview

This project was developed as a final machine learning project for predicting the market price of used cars.

The goal is to build a **regression model** that predicts the value of the `priceUSD` column using information about the car, such as:

- brand and model;
- production year;
- mileage;
- fuel type;
- engine volume;
- color;
- transmission;
- drive unit;
- vehicle segment;
- condition.

The project follows a complete machine learning workflow:

**EDA → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Evaluation → Model Comparison → Final Model**

---

## 2. Dataset

The project uses the provided `cars.csv` dataset.

The original dataset contains:

- **56,244 rows**
- **12 columns**

### Original columns

| Column | Description |
|---|---|
| `make` | Car brand |
| `model` | Car model |
| `priceUSD` | Target variable – price in USD |
| `year` | Production year |
| `condition` | Vehicle condition |
| `mileage(kilometers)` | Mileage in kilometers |
| `fuel_type` | Fuel type |
| `volume(cm3)` | Engine volume in cm³ |
| `color` | Car color |
| `transmission` | Transmission type |
| `drive_unit` | Drive system |
| `segment` | Vehicle segment |

The target variable is:

```text
priceUSD
```

This makes the problem a **supervised regression problem**.

---

## 3. Project Structure

```text
car-price-prediction/
│
├── data/
│   └── cars.csv
│
├── notebooks/
│   └── used_car_price_prediction.ipynb
│
├── source/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_comparison.py
│   └── predict.py
│
├── models/
│   └── car_price_model.joblib
│
├── reports/
│   ├── model_comparison.csv
│   ├── final_model_metrics.csv
│   ├── prediction_examples.csv
│   └── test_set.csv
│
├── run_project.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Exploratory Data Analysis

The Jupyter notebook contains the complete EDA.

The analysis checks:

- dataset dimensions;
- data types;
- missing values;
- duplicated rows;
- numerical statistics;
- categorical distributions;
- target price distribution;
- relationships between price and year;
- relationships between price and mileage;
- extreme values.

### Important findings

The original dataset contains missing values in:

- `volume(cm3)` – 47 missing values;
- `drive_unit` – 1,905 missing values;
- `segment` – 5,291 missing values.

There are also extreme observations. Examples include:

- production years earlier than 1980;
- mileage above 1,000,000 km;
- prices below $100 or above $200,000;
- engine volumes above 8,000 cm³.

These observations were reviewed as part of data cleaning.

---

## 5. Data Cleaning

Data cleaning is implemented in:

```text
source/data_cleaning.py
```

The cleaning process:

1. standardizes column names;
2. converts numerical columns to numeric data types;
3. standardizes text values;
4. removes rows without a valid target value;
5. removes clearly implausible prices;
6. removes implausible production years;
7. removes extreme mileage values;
8. removes clearly implausible engine volumes.

After cleaning, the dataset contains:

**55,535 rows**

The remaining missing values are intentionally not removed at this stage. They are handled later by the preprocessing pipeline using imputation.

This approach prevents unnecessary loss of valid observations.

---

## 6. Feature Engineering

Feature engineering is implemented in:

```text
source/feature_engineering.py
```

The following features were created:

### `car_age`

```text
reference year - production year
```

The project uses 2026 as the reference year.

### `mileage_per_year`

Approximate mileage accumulated per year:

```text
mileage / car_age
```

### `engine_volume_liters`

Engine volume converted from cm³ to liters:

```text
volume(cm3) / 1000
```

### `is_newer_car`

Binary indicator:

```text
1 = production year >= 2015
0 = otherwise
```

### `is_high_mileage`

Binary indicator:

```text
1 = mileage >= 300,000 km
0 = otherwise
```

### `brand_model`

Combination of:

```text
make + model
```

This allows the model to capture more specific market differences between individual car models.

---

## 7. Data Preprocessing

Preprocessing is implemented in:

```text
source/data_preprocessing.py
```

### Numerical features

Missing numerical values are filled using:

```text
SimpleImputer(strategy="median")
```

For the one-hot encoded models, numerical variables are also standardized using:

```text
StandardScaler
```

### Categorical features

Missing categorical values are filled using:

```text
SimpleImputer(strategy="most_frequent")
```

Categorical values are encoded using:

```text
OneHotEncoder(handle_unknown="ignore")
```

A compact ordinal encoding pipeline is also provided for tree and boosting models. This keeps the training matrix manageable for the 56k-row dataset.

The preprocessing is part of the model pipeline, which helps prevent data leakage.

---

## 8. Train/Test Split

The dataset is divided into:

- **80% training data**
- **20% test data**

The split uses:

```python
random_state=42
```

This makes the experiment reproducible.

The same train/test split is used when comparing the models, which makes the comparison fair.

---

## 9. Models

Four regression approaches were evaluated:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **HistGradientBoosting Regressor**

The models were evaluated using the same held-out test set.

---

## 10. Evaluation Metrics

The project uses four regression metrics.

### MAE – Mean Absolute Error

MAE is the primary metric because it is easy to interpret.

If:

```text
MAE = 1,132.99
```

the model's predictions are on average about **$1,133 away from the actual price**, measured as absolute error.

### MSE – Mean Squared Error

MSE penalizes large errors more strongly because the errors are squared.

### RMSE – Root Mean Squared Error

RMSE is expressed in dollars, which makes it easier to compare with the target value.

### R² – R-squared

R² measures how much of the variation in car prices is explained by the model.

---

## 11. Model Comparison Results

The models produced the following results on the same test set:

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| **Random Forest** | **1,132.99** | 2,828.49 | 0.8799 |
| HistGradientBoosting | 1,252.58 | **2,705.87** | **0.8901** |
| Decision Tree | 1,443.88 | 3,746.08 | 0.7894 |
| Linear Regression | 2,029.87 | 3,763.93 | 0.7874 |

### Why was Random Forest selected?

Random Forest has the **lowest MAE**, which is the primary metric selected for this project.

HistGradientBoosting has a slightly better R² and RMSE, but Random Forest has a lower average absolute dollar error.

Therefore, based on the project's main objective of minimizing the typical price prediction error, **Random Forest was selected as the final model**.

---

## 12. Final Model

The final model is:

```text
RandomForestRegressor
```

The complete preprocessing and model pipeline is saved as:

```text
models/car_price_model.joblib
```

The saved file contains the preprocessing steps together with the trained model, so a new car can be passed to the pipeline without manually encoding the categorical variables.

---

## 13. Final Model Performance

Final test-set performance:

```text
MAE  = $1,132.99
MSE  = 8,000,335.05
RMSE = $2,828.49
R²   = 0.8799
```

### Interpretation

The most important result is:

**MAE = $1,132.99**

This means that the model's average absolute prediction error is approximately **$1,133**.

The R² value of approximately **0.88** indicates that the model explains a large proportion of the variation in the target prices.

However, the model is not expected to predict every car's exact market price. Real prices can depend on information that is not present in the dataset, such as:

- exact equipment level;
- service history;
- geographic location;
- number of previous owners;
- accident history;
- cosmetic condition;
- seller type;
- optional equipment.

---

## 14. Example Prediction

The project includes `source/predict.py`.

Example input:

```python
example_car = {
    "make": "volkswagen",
    "model": "golf",
    "year": 2014,
    "condition": "with mileage",
    "mileage(kilometers)": 180000,
    "fuel_type": "diesel",
    "volume(cm3)": 1600,
    "color": "black",
    "transmission": "mechanics",
    "drive_unit": "front-wheel drive",
    "segment": "C",
}
```

The trained model predicts an approximate price.

The exact prediction is generated by the saved model and can change if the model is retrained with different parameters or data.

---

## 15. How to Run the Project

### Step 1 – Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd car-price-predictions
```

### Step 2 – Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Run the complete workflow

From the project root:

```bash
python run_project.py
```

This will:

1. compare the regression models;
2. train the final Random Forest model;
3. save the model;
4. evaluate the final model;
5. create the evaluation reports.

---

## 16. Running Individual Scripts

All scripts inside the `source/` directory dynamically add the project root to `sys.path`.

This allows you to run any script directly from the terminal or execute the main pipeline without configuration errors:

### Data cleaning
```bash
python source/data_cleaning.py
```

### Feature engineering

```bash
python source/feature_engineering.py
```

### Model comparison

```bash
python source/model_comparison.py
```

### Train final model

```bash
python source/model_training.py
```

### Evaluate final model

```bash
python source/model_evaluation.py
```

### Test a new car prediction

```bash
python source/predict.py
```

---

## 17. Running the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/used_car_price_prediction.ipynb
```

Run the cells from top to bottom.

The notebook contains the complete explanation of:

- EDA;
- data quality;
- cleaning;
- feature engineering;
- preprocessing;
- model comparison;
- evaluation;
- visualization;
- final prediction;
- conclusions.

---

## 18. Generated Reports

After running the project, the following files are created in `reports/`:

### `model_comparison.csv`

Contains the performance of all tested models.

### `final_model_metrics.csv`

Contains the final model metrics.

### `prediction_examples.csv`

Contains examples of actual prices, predicted prices, and absolute errors.

### `test_set.csv`

Contains the held-out test data used for reproducible evaluation.

---

## 19. Conclusion

The project successfully implements a complete machine learning regression workflow for used-car price prediction.

The final Random Forest model achieved an MAE of approximately **$1,133** on the held-out test set.

The project demonstrates the complete process required for a practical machine learning solution:

**data exploration → data cleaning → feature engineering → preprocessing → model training → model comparison → evaluation → final model selection → prediction**

The main conclusion is that the model can provide a useful **approximate market-price estimate**, while the prediction should not be treated as an exact valuation.

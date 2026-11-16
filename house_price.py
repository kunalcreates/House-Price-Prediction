"""House Price Prediction using Linear Regression and Random Forest.

Generates a synthetic housing dataset (no external download needed),
trains two models, compares them, and predicts the price of a new house.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def make_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a synthetic housing dataset with realistic-looking relationships."""
    rng = np.random.default_rng(RANDOM_STATE)

    area = rng.normal(1500, 400, n_samples).clip(400, 4000)  # square feet
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n_samples), 1, 4)
    age = rng.integers(0, 50, n_samples)  # years
    distance = rng.uniform(1, 30, n_samples)  # km from city centre

    # Price in lakhs (INR): larger, newer, closer homes cost more
    price = (
        0.05 * area
        + 3 * bedrooms
        + 2 * bathrooms
        - 0.08 * age
        - 0.3 * distance
        + rng.normal(0, 3, n_samples)
    )

    return pd.DataFrame(
        {
            "area_sqft": area.round(0),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "distance_km": distance.round(1),
            "price_lakhs": price.round(2),
        }
    )


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train both models and return their evaluation metrics."""
    features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
    X = df[features]
    y = df["price_lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }
    return results


def main() -> None:
    df = make_dataset()
    print(f"Dataset: {len(df)} houses\n")

    results = train_and_evaluate(df)
    for name, res in results.items():
        print(f"{name:18s} MAE: {res['mae']:.2f} lakhs   R2: {res['r2']:.3f}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")

    new_house = pd.DataFrame(
        [
            {
                "area_sqft": 1800,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 5,
                "distance_km": 8.0,
            }
        ]
    )
    predicted = best_model.predict(new_house)[0]
    print(f"Predicted price for a 1800 sq ft, 3 BHK, 5-year-old house "
          f"8 km from centre: {predicted:.2f} lakhs")


if __name__ == "__main__":
    main()

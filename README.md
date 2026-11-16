# House Price Prediction

A machine learning project that predicts house prices using Python and scikit-learn.

## What it does

- Generates a synthetic housing dataset (no download needed).
- Trains a **Linear Regression** and a **Random Forest** model.
- Compares them using MAE and R² scores.
- Predicts the price of a sample house using the better model.

## Features

- `area_sqft`, `bedrooms`, `bathrooms`, `age_years`, `distance_km`

Target: `price_lakhs` (price in lakhs INR)

## Setup

```bash
pip install numpy pandas scikit-learn
```

## Run

```bash
python house_price.py
```

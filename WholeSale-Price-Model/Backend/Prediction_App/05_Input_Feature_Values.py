"""
Feature filling script for backend data.
Fills null values in derived columns using the same formulas as training.
Run this BEFORE the prediction script.
"""

import pandas as pd
import numpy as np
import os

# ========== CONFIGURATION ==========
INPUT_PATH = "../Data/Backend_Data.csv"
OUTPUT_PATH = "../Data/Backend_Data.csv"
# ====================================

cities = ['Badulla', 'Hambantota', 'Jaffna', 'Kurunegala',
          'Matale', 'Nuwara_Eliya', 'Ratnapura']
actual_cols = [f"{city}_actual_class" for city in cities]
prob_types = ['prob_drought', 'prob_flood_risk', 'prob_normal']

# All derived columns that may need filling
derived_cols = [
    'price_lag1', 'price_lag2',
    'price_roll_mean_4', 'price_roll_std_4',
    'avg_prob_drought', 'max_prob_drought', 'min_prob_drought', 'std_prob_drought',
    'avg_prob_flood_risk', 'max_prob_flood_risk', 'min_prob_flood_risk', 'std_prob_flood_risk',
    'avg_prob_normal', 'max_prob_normal', 'min_prob_normal', 'std_prob_normal',
    'count_drought', 'count_normal', 'count_flood_risk',
    'usd_x_avg_drought', 'usd_x_avg_flood', 'usd_x_avg_normal',
    'week_x_avg_drought', 'week_x_avg_flood',
    'USD_LKR_avg_sq', 'RateChange_avg_%_sq', 'week_num_sq',
    'week_sin', 'week_cos',
    'veg_2', 'veg_3', 'veg_4', 'veg_5', 'veg_6'
]

# ------------------------------------------------------------
print("=" * 60)
print("FEATURE FILLING FOR BACKEND DATA")
print("=" * 60)

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"File not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# Ensure all derived columns exist
for col in derived_cols:
    if col not in df.columns:
        df[col] = np.nan
        print(f"Added missing column: {col}")

# Count new rows (price is null) – just for info
new_rows = df['price'].isnull().sum()
print(f"Rows with null price (new rows): {new_rows}")

# ------------------------------------------------------------
# 1. Price lags (shift within each vegetable)
# ------------------------------------------------------------
new_lag1 = df.groupby('vegetable')['price'].shift(1)
new_lag2 = df.groupby('vegetable')['price'].shift(2)

df['price_lag1'] = df['price_lag1'].fillna(new_lag1)
df['price_lag2'] = df['price_lag2'].fillna(new_lag2)
print("✓ Lag features filled")

# ------------------------------------------------------------
# 2. Rolling statistics (4‑week, using past prices only)
# ------------------------------------------------------------
# Use shifted price to avoid including current week
shifted_price = df.groupby('vegetable')['price'].shift(1)

new_roll_mean = shifted_price.groupby(df['vegetable']).transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)
new_roll_std = shifted_price.groupby(df['vegetable']).transform(
    lambda x: x.rolling(4, min_periods=1).std()
)

df['price_roll_mean_4'] = df['price_roll_mean_4'].fillna(new_roll_mean)
df['price_roll_std_4']  = df['price_roll_std_4'].fillna(new_roll_std)
print("✓ Rolling features filled")

# ------------------------------------------------------------
# 3. Probability aggregations (averages, max, min, std)
# ------------------------------------------------------------
for prob in prob_types:
    prob_cols = [f"{city}_{prob}" for city in cities if f"{city}_{prob}" in df.columns]
    if prob_cols:
        new_avg = df[prob_cols].mean(axis=1)
        new_max = df[prob_cols].max(axis=1)
        new_min = df[prob_cols].min(axis=1)
        new_std = df[prob_cols].std(axis=1)

        df[f'avg_{prob}'] = df[f'avg_{prob}'].fillna(new_avg)
        df[f'max_{prob}'] = df[f'max_{prob}'].fillna(new_max)
        df[f'min_{prob}'] = df[f'min_{prob}'].fillna(new_min)
        df[f'std_{prob}'] = df[f'std_{prob}'].fillna(new_std)
print("✓ Probability aggregations filled")

# ------------------------------------------------------------
# 4. Count of actual classes (drought, normal, flood)
# ------------------------------------------------------------
for class_val, class_name in [(-1, 'drought'), (0, 'normal'), (1, 'flood_risk')]:
    new_count = (df[actual_cols] == class_val).sum(axis=1)
    df[f'count_{class_name}'] = df[f'count_{class_name}'].fillna(new_count)
print("✓ Count features filled")

# ------------------------------------------------------------
# 5. Interaction terms (USD_LKR and week)
# ------------------------------------------------------------
new_usd_drought = df['USD_LKR_avg'] * df['avg_prob_drought']
new_usd_flood   = df['USD_LKR_avg'] * df['avg_prob_flood_risk']
new_usd_normal  = df['USD_LKR_avg'] * df['avg_prob_normal']

df['usd_x_avg_drought'] = df['usd_x_avg_drought'].fillna(new_usd_drought)
df['usd_x_avg_flood']   = df['usd_x_avg_flood'].fillna(new_usd_flood)
df['usd_x_avg_normal']  = df['usd_x_avg_normal'].fillna(new_usd_normal)

new_week_drought = df['week_num'] * df['avg_prob_drought']
new_week_flood   = df['week_num'] * df['avg_prob_flood_risk']

df['week_x_avg_drought'] = df['week_x_avg_drought'].fillna(new_week_drought)
df['week_x_avg_flood']   = df['week_x_avg_flood'].fillna(new_week_flood)
print("✓ Interaction features filled")

# ------------------------------------------------------------
# 6. Polynomial features (squares)
# ------------------------------------------------------------
new_usd_sq    = df['USD_LKR_avg'] ** 2
new_rate_sq   = df['RateChange_avg'] ** 2    # column name in CSV
new_week_sq   = df['week_num'] ** 2

df['USD_LKR_avg_sq']      = df['USD_LKR_avg_sq'].fillna(new_usd_sq)
df['RateChange_avg_%_sq'] = df['RateChange_avg_%_sq'].fillna(new_rate_sq)
df['week_num_sq']         = df['week_num_sq'].fillna(new_week_sq)
print("✓ Polynomial features filled")

# ------------------------------------------------------------
# 7. Cyclical week encoding
# ------------------------------------------------------------
new_sin = np.sin(2 * np.pi * df['week_num'] / 52)
new_cos = np.cos(2 * np.pi * df['week_num'] / 52)

df['week_sin'] = df['week_sin'].fillna(new_sin)
df['week_cos'] = df['week_cos'].fillna(new_cos)
print("✓ Cyclical week features filled")

# ------------------------------------------------------------
# 8. One‑hot vegetable dummies
# ------------------------------------------------------------
if 'vegetable' in df.columns:
    # Generate dummies for all rows (same as during training)
    veg_dummies = pd.get_dummies(df['vegetable'], prefix='veg', drop_first=True)
    expected_veg = ['veg_2', 'veg_3', 'veg_4', 'veg_5', 'veg_6']

    for col in expected_veg:
        if col in veg_dummies.columns:
            new_dummy = veg_dummies[col]
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].fillna(new_dummy)
        else:
            # If no rows have this vegetable, dummy should be 0
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)

    # Convert to int (same as training)
    for col in expected_veg:
        if col in df.columns:
            df[col] = df[col].astype(int)
    print("✓ One‑hot vegetable columns filled")

# ------------------------------------------------------------
# Final check: any nulls left in derived columns for new rows?
# ------------------------------------------------------------
new_rows_df = df[df['price'].isnull()]
if len(new_rows_df) > 0:
    still_null = new_rows_df[derived_cols].isnull().sum()
    if still_null.sum() > 0:
        print("\n⚠️ Warning: Some derived columns still have nulls in new rows:")
        print(still_null[still_null > 0])
    else:
        print("\n✅ All derived columns are fully filled for new rows.")
else:
    print("\nℹ️ No new rows found – nothing to fill.")

# ------------------------------------------------------------
# Save updated data
# ------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nData saved to {OUTPUT_PATH}")
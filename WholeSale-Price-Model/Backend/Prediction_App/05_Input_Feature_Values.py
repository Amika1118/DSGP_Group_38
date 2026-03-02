import pandas as pd
import numpy as np
import os

# ========== CONFIGURATION ==========
INPUT_PATH = "../Data/Backend_Data.csv"
OUTPUT_PATH = "../Data/Backend_Data.csv"  # Save back to same file
# ====================================

# List of cities (used for probability aggregations)
cities = ['Badulla', 'Hambantota', 'Jaffna', 'Kurunegala',
          'Matale', 'Nuwara_Eliya', 'Ratnapura']

# Actual class columns
actual_cols = [f"{city}_actual_class" for city in cities]

# Probability types
prob_types = ['prob_drought', 'prob_flood_risk', 'prob_normal']

# List of all remaining columns to compute
remaining_columns = [
    'price_lag1',
    'price_lag2',
    'price_roll_mean_4',
    'price_roll_std_4',
    'avg_prob_drought',
    'max_prob_drought',
    'min_prob_drought',
    'std_prob_drought',
    'avg_prob_flood_risk',
    'max_prob_flood_risk',
    'min_prob_flood_risk',
    'std_prob_flood_risk',
    'avg_prob_normal',
    'max_prob_normal',
    'min_prob_normal',
    'std_prob_normal',
    'count_drought',
    'count_normal',
    'count_flood_risk',
    'usd_x_avg_drought',
    'usd_x_avg_flood',
    'usd_x_avg_normal',
    'week_x_avg_drought',
    'week_x_avg_flood',
    'USD_LKR_avg_sq',
    'RateChange_avg_%_sq',
    'week_num_sq',
    'week_sin',
    'week_cos',
    'veg_2',
    'veg_3',
    'veg_4',
    'veg_5',
    'veg_6'
]

# ========== MAIN ==========

# --- Load data ---
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Backend data file not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
print(f"Loaded data shape: {df.shape}")

# --- Identify rows that need computation (any remaining column is null) ---
# First, check which remaining columns exist in the dataframe
existing_remaining = [col for col in remaining_columns if col in df.columns]
missing_remaining = [col for col in remaining_columns if col not in df.columns]

if missing_remaining:
    print(f"Adding missing columns: {missing_remaining}")
    for col in missing_remaining:
        df[col] = np.nan

# Find rows where ANY of the remaining columns are null
null_mask = df[existing_remaining + missing_remaining].isnull().any(axis=1)
rows_to_compute = null_mask.sum()
print(f"Rows needing computation: {rows_to_compute} out of {len(df)}")

if rows_to_compute == 0:
    print("All remaining columns are already filled. No computation needed.")
else:
    # Create a copy of rows to compute to avoid SettingWithCopyWarning
    compute_idx = df[null_mask].index
    df_to_compute = df.loc[compute_idx].copy()

    print(f"Computing features for {len(df_to_compute)} rows...")

    # --- Sort by vegetable and week_num for proper lag calculations ---
    df_to_compute = df_to_compute.sort_values(['vegetable', 'week_num'])

    # --- 1. Price lag features ---
    # Need to use the full dataset for lags to get previous weeks
    df_full_sorted = df.sort_values(['vegetable', 'week_num'])
    df['price_lag1'] = df_full_sorted.groupby('vegetable')['price'].shift(1)
    df['price_lag2'] = df_full_sorted.groupby('vegetable')['price'].shift(2)
    print("Lag features created.")

    # --- 2. Rolling statistics ---
    df['price_roll_mean_4'] = df.groupby('vegetable')['price'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )
    df['price_roll_std_4'] = df.groupby('vegetable')['price'].transform(
        lambda x: x.rolling(4, min_periods=1).std()
    )
    print("Rolling features created.")

    # --- 3. Probability aggregations ---
    for prob in prob_types:
        prob_cols = [f"{city}_{prob}" for city in cities if f"{city}_{prob}" in df.columns]
        if prob_cols:
            df[f'avg_{prob}'] = df[prob_cols].mean(axis=1)
            df[f'max_{prob}'] = df[prob_cols].max(axis=1)
            df[f'min_{prob}'] = df[prob_cols].min(axis=1)
            df[f'std_{prob}'] = df[prob_cols].std(axis=1)
    print("Aggregated probability features added.")

    # --- 4. Count features ---
    for class_val, class_name in [(-1, 'drought'), (0, 'normal'), (1, 'flood_risk')]:
        df[f'count_{class_name}'] = (df[actual_cols] == class_val).sum(axis=1)
    print("Count features added.")

    # --- 5. Interaction features ---
    df['usd_x_avg_drought'] = df['USD_LKR_avg'] * df['avg_prob_drought']
    df['usd_x_avg_flood'] = df['USD_LKR_avg'] * df['avg_prob_flood_risk']
    df['usd_x_avg_normal'] = df['USD_LKR_avg'] * df['avg_prob_normal']

    df['week_x_avg_drought'] = df['week_num'] * df['avg_prob_drought']
    df['week_x_avg_flood'] = df['week_num'] * df['avg_prob_flood_risk']
    print("Interaction features added.")

    # --- 6. Polynomial features ---
    df['USD_LKR_avg_sq'] = df['USD_LKR_avg'] ** 2
    df['RateChange_avg_%_sq'] = df['RateChange_avg'] ** 2
    df['week_num_sq'] = df['week_num'] ** 2
    print("Polynomial features added.")

    # --- 7. Cyclical week features ---
    df['week_sin'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num'] / 52)
    print("Cyclical week features added.")

    # --- 8. One-hot encoded vegetable columns ---
    # Check if vegetable column exists and has the mapping
    if 'vegetable' in df.columns:
        # Create dummies (drop_first=True creates k-1 dummies)
        veg_dummies = pd.get_dummies(df['vegetable'], prefix='veg', drop_first=True)

        # Add dummy columns to dataframe
        for col in veg_dummies.columns:
            df[col] = veg_dummies[col]

        # Ensure all expected veg_X columns exist (fill missing with 0)
        expected_veg_cols = ['veg_2', 'veg_3', 'veg_4', 'veg_5', 'veg_6']
        for col in expected_veg_cols:
            if col not in df.columns:
                df[col] = 0

        # Convert to int
        for col in expected_veg_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        print("One-hot encoded vegetable columns added.")
    else:
        print("Warning: 'vegetable' column not found for one-hot encoding.")

    # --- Verify that all remaining columns are now filled for computed rows ---
    # (Optional: check specific rows)

    print(f"Feature computation complete. Final shape: {df.shape}")

# --- Save updated data ---
df.to_csv(OUTPUT_PATH, index=False)
print(f"Data successfully saved to {OUTPUT_PATH}")

# --- Display summary of the newly computed columns ---
print("\n--- Summary of computed columns ---")
for col in remaining_columns:
    if col in df.columns:
        null_count = df[col].isnull().sum()
        print(f"{col}: {null_count} null values out of {len(df)} rows")
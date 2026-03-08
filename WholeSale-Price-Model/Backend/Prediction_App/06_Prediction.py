"""
Backend prediction script.
Loads the best model, uses the current date to determine week,
and predicts prices for all vegetables of that week.
Run AFTER the feature filling script.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import sys

# ============================================================================
# PATHS
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "..", "models", "best_model", "best_model.pkl")
backend_data_path = os.path.join(script_dir, "..", "Data", "Backend_Data.csv")

print("=" * 60)
print("VEGETABLE PRICE PREDICTION BACKEND")
print("=" * 60)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
print(f"\n📂 Loading best model from: {model_path}")
if not os.path.exists(model_path):
    print(f"❌ Error: Model file not found at {model_path}")
    sys.exit(1)

try:
    best_model = joblib.load(model_path)
    print(f"✅ Model loaded successfully!  Type: {type(best_model).__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Determine current week from date
# ----------------------------------------------------------------------------
def get_week_from_date(date_obj):
    month = date_obj.month
    day = date_obj.day
    week_ranges = [
        (1,1,1,7), (1,8,1,14), (1,15,1,21), (1,22,1,28), (1,29,2,4),
        (2,5,2,11), (2,12,2,18), (2,19,2,25), (2,26,3,4), (3,5,3,11),
        (3,12,3,18), (3,19,3,25), (3,26,4,1), (4,2,4,8), (4,9,4,15),
        (4,16,4,22), (4,23,4,29), (4,30,5,6), (5,7,5,13), (5,14,5,20),
        (5,21,5,27), (5,28,6,3), (6,4,6,10), (6,11,6,17), (6,18,6,24),
        (6,25,7,1), (7,2,7,8), (7,9,7,15), (7,16,7,22), (7,23,7,29),
        (7,30,8,5), (8,6,8,12), (8,13,8,19), (8,20,8,26), (8,27,9,2),
        (9,3,9,9), (9,10,9,16), (9,17,9,23), (9,24,9,30), (10,1,10,7),
        (10,8,10,14), (10,15,10,21), (10,22,10,28), (10,29,11,4),
        (11,5,11,11), (11,12,11,18), (11,19,11,25), (11,26,12,2),
        (12,3,12,9), (12,10,12,16), (12,17,12,23), (12,24,12,30)
    ]
    for week_num, (sm, sd, em, ed) in enumerate(week_ranges, 1):
        start = datetime(date_obj.year, sm, sd)
        end   = datetime(date_obj.year, em, ed)
        if start <= date_obj <= end:
            return week_num
    if month == 12 and day == 31:
        return 1
    return None

current_date = datetime.now()
print(f"\n📅 Current device date: {current_date.strftime('%Y-%m-%d (%A)')}")
current_week = get_week_from_date(current_date)
current_year = current_date.year
print(f"📊 Determined week: Week {current_week}, {current_year}")

# ----------------------------------------------------------------------------
# Load backend data
# ----------------------------------------------------------------------------
print(f"\n📂 Loading backend data from: {backend_data_path}")
if not os.path.exists(backend_data_path):
    print(f"❌ Error: File not found")
    sys.exit(1)

backend_df = pd.read_csv(backend_data_path)
print(f"✅ Loaded {len(backend_df)} rows, columns: {list(backend_df.columns)}")

# Filter for current year and week
filtered_df = backend_df[(backend_df['year'] == current_year) &
                         (backend_df['week_num'] == current_week)]
if len(filtered_df) == 0:
    print(f"\n⚠️ No data for Year {current_year}, Week {current_week}")
    print(f"Available years: {backend_df['year'].unique()}")
    sys.exit(1)

print(f"\n✅ Found {len(filtered_df)} rows for current week.")

# ============================================================================
# DEBUG: Check lag features for new week
# ============================================================================
print("\n" + "=" * 60)
print("DEBUG: Lag features for new week")
print("=" * 60)
veg_names = {1:"Bitter Gourd", 2:"Brinjals", 3:"Cabbage",
             4:"Carrot", 5:"Pumpkin", 6:"Tomatoes"}

for _, row in filtered_df.iterrows():
    veg = int(row['vegetable'])
    name = veg_names.get(veg, f"Unknown({veg})")
    print(f"\n{name}:")
    print(f"  price_lag1          = {row['price_lag1']:.2f}")
    print(f"  price_lag2          = {row['price_lag2']:.2f}")
    print(f"  price_roll_mean_4   = {row['price_roll_mean_4']:.2f}")
    print(f"  price_roll_std_4    = {row['price_roll_std_4']:.2f}")

# ----------------------------------------------------------------------------
# Prepare features (all columns except year and price)
# ----------------------------------------------------------------------------
feature_cols = [c for c in backend_df.columns if c not in ['year', 'price']]
print(f"\n🔍 Using {len(feature_cols)} feature columns for prediction")

X_pred = filtered_df[feature_cols].copy()

# If any missing columns (should not happen after feature filling), fill with 0
missing = [c for c in feature_cols if c not in X_pred.columns]
if missing:
    print(f"⚠️ Missing columns (filling with 0): {missing}")
    for m in missing:
        X_pred[m] = 0

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

try:
    predictions = best_model.predict(X_pred)
    # Clip negative predictions to 0
    neg_count = (predictions < 0).sum()
    if neg_count > 0:
        print(f"⚠️ {neg_count} negative predictions clipped to 0.")
        predictions = np.maximum(predictions, 0)

    filtered_df = filtered_df.copy()
    filtered_df['predicted_price'] = predictions
    print(f"✅ Predicted {len(predictions)} vegetables.")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PREDICTED PRICES BY VEGETABLE")
print("=" * 60)

results = []
for _, row in filtered_df.iterrows():
    veg = int(row['vegetable'])
    name = veg_names.get(veg, f"Unknown({veg})")
    results.append({
        'Vegetable': name,
        'Vegetable Code': veg,
        'Predicted Price': round(row['predicted_price'], 2)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ----------------------------------------------------------------------------
# Update backend CSV with predicted prices
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("UPDATING BACKEND DATA")
print("=" * 60)

mask = (backend_df['year'] == current_year) & (backend_df['week_num'] == current_week)
backend_df.loc[mask, 'price'] = filtered_df['predicted_price'].values
backend_df.to_csv(backend_data_path, index=False)
print(f"✅ Updated price column for Week {current_week}, {current_year}")

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"📅 Date: {current_date.strftime('%Y-%m-%d')}")
print(f"📊 Week: {current_week}, {current_year}")
print(f"🥕 Vegetables processed: {len(results)}")
print(f"💰 Average predicted price: {results_df['Predicted Price'].mean():.2f}")
print(f"📈 Min: {results_df['Predicted Price'].min():.2f}")
print(f"📉 Max: {results_df['Predicted Price'].max():.2f}")
print("\n✅ Backend processing completed successfully!")

# Optional: save detailed results
out_file = os.path.join(script_dir, "..", "Data", "Predictions.csv")
results_df.to_csv(out_file, index=False)
print(f"\n📁 Detailed predictions saved to: {out_file}")
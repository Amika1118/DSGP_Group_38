"""
Script to run the USD/LKR scraper, retrieve the predicted rate,
and append/update the historical CSV with the correct date and rate change.
The date used is the first day of the current week based on a fixed weekly grid:
- Week 1 starts on Jan 1
- Week n starts on Jan 1 + (n-1)*7 days
- Week numbers are capped at 52 (the last week may be longer to include Dec 31)
"""

import subprocess
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# ----------------------------------------------------------------------
# Step 1: Define paths relative to this script's location
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the scraper script
scraper_path = os.path.join(script_dir, "..", "USD-LKR_Scraper", "Scraper.py")

# Path to the forecast JSON file
forecast_json_path = os.path.join(script_dir, "..", "USD-LKR_Scraper", "usd_lkr_forecast.json")

# Path to the historical CSV file
csv_path = os.path.join(script_dir, "..", "..", "data", "raw", "USD_LKR Historical Data.csv")

# ----------------------------------------------------------------------
# Step 2: Run the scraper
# ----------------------------------------------------------------------
print("Running USD/LKR scraper...")
try:
    result = subprocess.run(
        [sys.executable, scraper_path],
        capture_output=True,
        text=True,
        check=True
    )
    print("Scraper executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error running scraper: {e}")
    print(f"stderr: {e.stderr}")
    sys.exit(1)

# ----------------------------------------------------------------------
# Step 3: Load the forecast JSON
# ----------------------------------------------------------------------
if not os.path.exists(forecast_json_path):
    print(f"Forecast JSON file not found: {forecast_json_path}")
    sys.exit(1)

with open(forecast_json_path, "r") as f:
    forecast_data = json.load(f)

predicted_rate = forecast_data.get("predicted_rate")
if predicted_rate is None:
    print("No predicted_rate found in JSON.")
    sys.exit(1)

print(f"Predicted rate: {predicted_rate}")

# ----------------------------------------------------------------------
# Step 4: Load the historical CSV
# ----------------------------------------------------------------------
if not os.path.exists(csv_path):
    print(f"Historical CSV not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Ensure columns are present
required_cols = ["Date", "DollarRate", "RateChange %"]
if not all(col in df.columns for col in required_cols):
    print("CSV does not have required columns.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Step 5: Determine target date based on current week definition
# ----------------------------------------------------------------------
today = datetime.now().date()
year = today.year
jan1 = datetime(year, 1, 1).date()
delta_days = (today - jan1).days
week_num = delta_days // 7 + 1
if week_num > 52:
    week_num = 52  # last week absorbs remaining days
target_date = jan1 + timedelta(days=(week_num - 1) * 7)
target_date_str = target_date.strftime("%m/%d/%Y")

print(f"Today's date: {today.strftime('%m/%d/%Y')}")
print(f"Target date (first day of current week): {target_date_str}")

# ----------------------------------------------------------------------
# Step 6: Prepare dataframe with datetime column for comparison
# ----------------------------------------------------------------------
df["Date_dt"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date_dt").reset_index(drop=True)

# Helper to find the most recent row before a given date (exclusive)
def get_previous_rate(before_date):
    mask = df["Date_dt"] < pd.Timestamp(before_date)
    if mask.any():
        return df.loc[mask, "DollarRate"].iloc[-1]
    return None

# ----------------------------------------------------------------------
# Step 7: Check if target date already exists
# ----------------------------------------------------------------------
mask_target = df["Date_dt"] == pd.Timestamp(target_date)
if mask_target.any():
    # Update existing row
    idx = df[mask_target].index[0]
    print(f"Row for {target_date_str} already exists. Updating...")

    # Get rate from exactly 7 days prior (previous week's start)
    prev_week_date = target_date - timedelta(days=7)
    prev_mask = df["Date_dt"] == pd.Timestamp(prev_week_date)
    if prev_mask.any():
        prev_rate = df.loc[prev_mask, "DollarRate"].values[0]
    else:
        # Fallback to most recent row before target date
        prev_rate = get_previous_rate(target_date)
        if prev_rate is None:
            prev_rate = predicted_rate  # no previous data, change 0
            print("Warning: No previous rate found. Setting change to 0.")
        else:
            print(f"Previous week's date {prev_week_date.strftime('%m/%d/%Y')} not found; using last available rate from {df.loc[df['Date_dt'] < pd.Timestamp(target_date), 'Date_dt'].iloc[-1].strftime('%m/%d/%Y')}")

    rate_change_pct = ((predicted_rate - prev_rate) / prev_rate) * 100 if prev_rate != 0 else 0.0
    rate_change_str = f"{rate_change_pct:.2f}%"

    # Update the row
    df.at[idx, "DollarRate"] = predicted_rate
    df.at[idx, "RateChange %"] = rate_change_str
    print(f"Updated rate to {predicted_rate} with change {rate_change_str}")
else:
    # Append new row
    print(f"No row for {target_date_str} found. Appending new row.")

    # Get rate from previous week's start
    prev_week_date = target_date - timedelta(days=7)
    prev_mask = df["Date_dt"] == pd.Timestamp(prev_week_date)
    if prev_mask.any():
        prev_rate = df.loc[prev_mask, "DollarRate"].values[0]
    else:
        # Fallback to last row overall
        if not df.empty:
            prev_rate = df.iloc[-1]["DollarRate"]
            print(f"Previous week's date {prev_week_date.strftime('%m/%d/%Y')} not found; using last available rate from {df.iloc[-1]['Date']}")
        else:
            prev_rate = predicted_rate  # empty CSV, change 0

    rate_change_pct = ((predicted_rate - prev_rate) / prev_rate) * 100 if prev_rate != 0 else 0.0
    rate_change_str = f"{rate_change_pct:.2f}%"

    new_row = {
        "Date": target_date_str,
        "DollarRate": predicted_rate,
        "RateChange %": rate_change_str
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Appended new row with rate {predicted_rate} and change {rate_change_str}")

# ----------------------------------------------------------------------
# Step 8: Drop temporary column and save
# ----------------------------------------------------------------------
df = df.drop(columns=["Date_dt"])
df.to_csv(csv_path, index=False)
print(f"Updated CSV saved to {csv_path}")
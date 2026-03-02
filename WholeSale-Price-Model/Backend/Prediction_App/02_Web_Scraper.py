"""
Script to run the USD/LKR scraper, retrieve the predicted rate,
and append it to the historical CSV with the correct date and rate change.
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
    # Run the scraper script (assumes it's a Python script)
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
# Step 5: Determine the last date and compute new date
# ----------------------------------------------------------------------
# Convert Date column to datetime (assuming mm/dd/yyyy format)
df["Date_dt"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Sort by date just in case
df = df.sort_values("Date_dt").reset_index(drop=True)

last_row = df.iloc[-1]
last_date = last_row["Date_dt"]
last_rate = last_row["DollarRate"]

# New date = last date + 7 days
new_date = last_date + timedelta(days=7)
new_date_str = new_date.strftime("%m/%d/%Y")  # back to mm/dd/yyyy

print(f"Last date: {last_date.strftime('%m/%d/%Y')} with rate {last_rate}")
print(f"New date: {new_date_str}")

# ----------------------------------------------------------------------
# Step 6: Calculate the rate change percentage
# ----------------------------------------------------------------------
rate_change_pct = ((predicted_rate - last_rate) / last_rate) * 100
# Format to 2 decimal places (like -0.08%)
rate_change_str = f"{rate_change_pct:.2f}%"

print(f"Rate change: {rate_change_str}")

# ----------------------------------------------------------------------
# Step 7: Append the new row
# ----------------------------------------------------------------------
new_row = {
    "Date": new_date_str,
    "DollarRate": predicted_rate,
    "RateChange %": rate_change_str
}

# Use pd.concat instead of append (future proof)
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Drop the temporary datetime column before saving
df = df.drop(columns=["Date_dt"])

# ----------------------------------------------------------------------
# Step 8: Save the updated CSV
# ----------------------------------------------------------------------
df.to_csv(csv_path, index=False)
print(f"Updated CSV saved to {csv_path}")
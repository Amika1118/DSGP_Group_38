import pandas as pd
from datetime import datetime, timedelta
import os

# ========== CONFIGURATION ==========
INPUT_PATH = "../../data/raw/Vegetable_prices_weekly.csv"
OUTPUT_PATH = "../Data/Backend_Data.csv"

# Set True for current week, False for next week
USE_CURRENT_WEEK = True  # Change to False if you need next week's data
# ====================================

# Mapping dictionaries (keys must match the raw data exactly after cleaning)
class_mapping = {'normal': 0, 'flood_risk': 1, 'drought': -1}
veg_mapping = {
    'Bitter Gourd': 1,
    'Brinjals': 2,
    'Cabbage': 3,
    'Carrot': 4,
    'Pumpkin': 5,
    'Tomatoes': 6
}

# All columns that must appear in the output (including year)
required_columns = [
    'year',
    'vegetable',
    'Badulla_actual_class', 'Hambantota_actual_class', 'Jaffna_actual_class',
    'Kurunegala_actual_class', 'Matale_actual_class', 'Nuwara_Eliya_actual_class',
    'Ratnapura_actual_class',
    'Badulla_prob_drought', 'Hambantota_prob_drought', 'Jaffna_prob_drought',
    'Kurunegala_prob_drought', 'Matale_prob_drought', 'Nuwara_Eliya_prob_drought',
    'Ratnapura_prob_drought',
    'Badulla_prob_flood_risk', 'Hambantota_prob_flood_risk', 'Jaffna_prob_flood_risk',
    'Kurunegala_prob_flood_risk', 'Matale_prob_flood_risk', 'Nuwara_Eliya_prob_flood_risk',
    'Ratnapura_prob_flood_risk',
    'Badulla_prob_normal', 'Hambantota_prob_normal', 'Jaffna_prob_normal',
    'Kurunegala_prob_normal', 'Matale_prob_normal', 'Nuwara_Eliya_prob_normal',
    'Ratnapura_prob_normal',
    'USD_LKR_avg', 'RateChange_avg', 'week_num'
]


def get_week_from_date(date):
    """
    Determine the custom week number for a given date based on the defined week boundaries.
    Week 1: Jan 1-7
    Week 2: Jan 8-14
    Week 3: Jan 15-21
    Week 4: Jan 22-28
    Week 5: Jan 29-Feb 4
    ... and so on with 7-day intervals
    """
    year = date.year
    # Create Jan 1 of the given year
    jan1 = datetime(year, 1, 1)

    # Calculate days since Jan 1
    days_diff = (date - jan1).days

    # Week number is floor(days_diff / 7) + 1
    # But careful: Jan 1 (days_diff=0) should be week 1
    week_num = (days_diff // 7) + 1

    # Handle dates that might fall into next year's weeks
    if week_num > 52:
        # If we're near year end, this might be week 52 or 53
        # Cap at 52 for simplicity (or adjust based on your needs)
        week_num = min(week_num, 52)

    return week_num


def get_week_range(week_num, year):
    """
    Get the start and end dates for a given week number in the custom system.
    Returns (start_date, end_date) as datetime objects.
    """
    jan1 = datetime(year, 1, 1)
    start_date = jan1 + timedelta(days=(week_num - 1) * 7)
    end_date = start_date + timedelta(days=6)
    return start_date, end_date


def get_target_week_and_year(today, use_current):
    """Determine target week number and year based on current date and flag."""
    current_week = get_week_from_date(today)
    current_year = today.year

    # Get the date ranges for debugging/information
    week_start, week_end = get_week_range(current_week, current_year)
    print(f"Current date {today.strftime('%Y-%m-%d')} falls in:")
    print(f"Week {current_week}: {week_start.strftime('%b %d')} – {week_end.strftime('%b %d')}")

    if use_current:
        target_week = current_week
        target_year = current_year
    else:
        target_week = current_week + 1
        target_year = current_year

        # Get next week's date range to check if it's in next year
        next_week_start, next_week_end = get_week_range(target_week, target_year)

        # If next week's start date is in the next year, adjust year
        if next_week_start.year > current_year:
            target_year = next_week_start.year
            # Recalculate week number for the new year
            target_week = get_week_from_date(next_week_start)

        # Show next week's range
        print(f"\nNext week (target):")
        print(
            f"Week {target_week}: {next_week_start.strftime('%b %d')} – {next_week_end.strftime('%b %d')} ({target_year})")

    return target_week, target_year


def week_to_int(week_series):
    """
    Convert week column to integer.
    Handles strings like 'W1', 'w1', '1' or already numeric.
    """
    if pd.api.types.is_numeric_dtype(week_series):
        return week_series.astype(int)
    # Assume string; remove any non-digit characters and convert
    return week_series.astype(str).str.replace(r'\D', '', regex=True).astype(int)


# ========== MAIN ==========
today = datetime.now()
target_week_num, target_year = get_target_week_and_year(today, USE_CURRENT_WEEK)
target_week_str = f"W{target_week_num}"

print(f"\nTarget week: {target_week_str} of {target_year}")

# --- Load raw data ---
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Raw data file not found: {INPUT_PATH}")

df_raw = pd.read_csv(INPUT_PATH)
print(f"Raw data shape: {df_raw.shape}")

# --- Filter for target year and week ---
df_new = df_raw[(df_raw['year'] == target_year) & (df_raw['week'] == target_week_str)].copy()

if df_new.empty:
    print(f"No data found for {target_year} week {target_week_str}. No new rows to append.")
else:
    print(f"Found {len(df_new)} rows for {target_year} week {target_week_str}.")

    # --- Debug: show unique values in actual_class columns BEFORE mapping ---
    cities = ['Badulla', 'Hambantota', 'Jaffna', 'Kurunegala',
              'Matale', 'Nuwara_Eliya', 'Ratnapura']
    actual_cols = [f"{city}_actual_class" for city in cities]
    print("\nUnique values in actual_class columns (before cleaning):")
    for col in actual_cols:
        if col in df_new.columns:
            uniq = df_new[col].dropna().unique()
            print(f"{col}: {uniq}")

    # --- Clean text in actual_class columns (strip, lowercase) ---
    for col in actual_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str).str.strip().str.lower()
            # Replace any string 'nan' with actual NaN
            df_new[col] = df_new[col].replace('nan', pd.NA)

    # --- Apply class mapping ---
    for col in actual_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].map(class_mapping)
        else:
            print(f"Warning: Column '{col}' missing, skipping mapping.")

    # --- Apply vegetable mapping ---
    if 'vegetable' in df_new.columns:
        # Clean vegetable names just in case
        df_new['vegetable'] = df_new['vegetable'].astype(str).str.strip()
        df_new['vegetable'] = df_new['vegetable'].map(veg_mapping)
        print("Vegetable codes applied.")
    else:
        print("Warning: 'vegetable' column not found.")

    # --- Convert week column from "W9" to integer and rename to week_num ---
    df_new['week_num'] = week_to_int(df_new['week'])
    # Drop the original 'week' column if it exists
    if 'week' in df_new.columns:
        df_new.drop(columns=['week'], inplace=True)

    # --- Ensure all required columns are present ---
    # (If any column is missing, it will be added with NaN values)
    for col in required_columns:
        if col not in df_new.columns:
            df_new[col] = pd.NA

    # --- Reorder columns to match the required order ---
    df_new = df_new[required_columns]

    # --- Append to existing Backend_Data.csv ---
    if os.path.exists(OUTPUT_PATH):
        df_existing = pd.read_csv(OUTPUT_PATH)
        print(f"Existing backend data shape: {df_existing.shape}")

        # Ensure week_num in existing data is integer
        if 'week_num' in df_existing.columns:
            df_existing['week_num'] = week_to_int(df_existing['week_num'])

        # Combine (new rows at the bottom)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print("Backend_Data.csv does not exist. Creating new file.")
        df_combined = df_new

    # --- Save ---
    df_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Data successfully written to {OUTPUT_PATH}. Total rows: {len(df_combined)}")
import pandas as pd
import os

# Define file paths
upcoming_file = "../../data/raw/Forcast_Weather/upcoming_7_days.csv"
predictions_file = "../../data/raw/predictions_2010_2025_for_arosha.csv"

# Read the upcoming 7 days data
upcoming_df = pd.read_csv(upcoming_file)

# Read the existing predictions data
predictions_df = pd.read_csv(predictions_file)

# Expected structure
expected_columns = ['date', 'city', 'actual_class', 'prob_drought', 'prob_flood_risk', 'prob_normal']

if list(predictions_df.columns) != expected_columns:
    print("Warning: Predictions file columns don't match expected format.")
    print(f"Expected: {expected_columns}")
    print(f"Found: {list(predictions_df.columns)}")

# Rename column if needed
if 'predicted_class' in upcoming_df.columns and 'actual_class' not in upcoming_df.columns:
    upcoming_df = upcoming_df.rename(columns={'predicted_class': 'actual_class'})

# Ensure correct column order
upcoming_df = upcoming_df[['date', 'city', 'actual_class', 'prob_drought', 'prob_flood_risk', 'prob_normal']]

# Combine data
combined_df = pd.concat([predictions_df, upcoming_df], ignore_index=True)

# Convert date to datetime for proper comparison
combined_df['date'] = pd.to_datetime(combined_df['date'])

# REMOVE DUPLICATES based on city + date
combined_df = combined_df.drop_duplicates(subset=['city', 'date'], keep='last')

# Sort for readability
combined_df = combined_df.sort_values(['city', 'date'])

# Convert date back to string format
combined_df['date'] = combined_df['date'].dt.strftime('%m/%d/%Y')

# Save back to file
combined_df.to_csv(predictions_file, index=False)

print(f"Successfully processed file: {predictions_file}")
print(f"Total rows now: {len(combined_df)}")

# Verify Nuwara Eliya
nuwara_eliya_data = combined_df[combined_df['city'] == 'Nuwara_Eliya'].tail(10)
print("\nLast 10 rows for Nuwara_Eliya after update:")
print(nuwara_eliya_data.to_string(index=False))
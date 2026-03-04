import pandas as pd
import os

# Define file paths
upcoming_file = "../../data/raw/Forcast_Weather/upcoming_7_days.csv"
predictions_file = "../../data/raw/predictions_2010_2025_for_arosha.csv"

# Read the upcoming 7 days data
upcoming_df = pd.read_csv(upcoming_file)

# Read the existing predictions data
predictions_df = pd.read_csv(predictions_file)

# Ensure both dataframes have the same structure
# The predictions file might have 'actual_class' while upcoming has 'predicted_class'
# We need to map 'predicted_class' to 'actual_class' for consistency

# Check if the predictions file has the expected columns
expected_columns = ['date', 'city', 'actual_class', 'prob_drought', 'prob_flood_risk', 'prob_normal']
if list(predictions_df.columns) != expected_columns:
    print(f"Warning: Predictions file columns don't match expected format.")
    print(f"Expected: {expected_columns}")
    print(f"Found: {list(predictions_df.columns)}")

# Prepare upcoming data to match predictions file structure
# Rename 'predicted_class' to 'actual_class' if needed
if 'predicted_class' in upcoming_df.columns and 'actual_class' not in upcoming_df.columns:
    upcoming_df = upcoming_df.rename(columns={'predicted_class': 'actual_class'})

# Ensure column order matches
upcoming_df = upcoming_df[['date', 'city', 'actual_class', 'prob_drought', 'prob_flood_risk', 'prob_normal']]

# Append the upcoming data to the predictions data
combined_df = pd.concat([predictions_df, upcoming_df], ignore_index=True)

# Sort by date and city for better organization
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df = combined_df.sort_values(['city', 'date'])
combined_df['date'] = combined_df['date'].dt.strftime('%m/%d/%Y')

# Save the combined dataframe back to the predictions file
combined_df.to_csv(predictions_file, index=False)

print(f"Successfully appended {len(upcoming_df)} rows to {predictions_file}")
print(f"Total rows now: {len(combined_df)}")

# Verify the append for Nuwara Eliya (as per your example)
nuwara_eliya_data = combined_df[combined_df['city'] == 'Nuwara_Eliya'].tail(10)
print("\nLast 10 rows for Nuwara_Eliya after append:")
print(nuwara_eliya_data.to_string(index=False))
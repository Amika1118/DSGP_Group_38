import pytest
import pandas as pd
from pathlib import Path
import importlib.util
import sys

SCRIPT_PATH = Path(__file__).parent.parent / "Prediction_App" / "01_Combine_Weather_Data.py"


def test_combine_weather_data(tmp_path):
    # Prepare test data
    upcoming_data = pd.DataFrame({
        'date': ['04/01/2026', '04/02/2026'],
        'city': ['Nuwara_Eliya', 'Badulla'],
        'predicted_class': ['normal', 'flood_risk'],
        'prob_drought': [0.1, 0.2],
        'prob_flood_risk': [0.2, 0.7],
        'prob_normal': [0.7, 0.1]
    })

    existing_data = pd.DataFrame({
        'date': ['03/01/2026', '03/02/2026'],
        'city': ['Nuwara_Eliya', 'Badulla'],
        'actual_class': ['normal', 'drought'],
        'prob_drought': [0.1, 0.8],
        'prob_flood_risk': [0.2, 0.1],
        'prob_normal': [0.7, 0.1]
    })

    # Create temporary input files
    upcoming_file = tmp_path / "upcoming_7_days.csv"
    predictions_file = tmp_path / "predictions_2010_2025_for_arosha.csv"

    upcoming_data.to_csv(upcoming_file, index=False)
    existing_data.to_csv(predictions_file, index=False)

    # Prepare globals (override file paths)
    script_globals = {
        'pd': pd,
        'os': __import__('os'),
        '__name__': '__main__',
        'upcoming_file': str(upcoming_file),
        'predictions_file': str(predictions_file),
    }

    # Execute script
    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        script_code = f.read()
    exec(script_code, script_globals)

    # Read output file
    result_df = pd.read_csv(predictions_file)

    # Assertions
    assert len(result_df) == 4, "Expected 4 rows (2 existing + 2 new)"
    expected_columns = ['date', 'city', 'actual_class', 'prob_drought', 'prob_flood_risk', 'prob_normal']
    assert list(result_df.columns) == expected_columns
    # Check sorting (city then date): first row should be Badulla with earliest date (03/02/2026)
    assert result_df['date'].iloc[0] == '03/02/2026'
    assert result_df['date'].iloc[-1] == '04/01/2026'
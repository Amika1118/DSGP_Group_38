import pytest
import pandas as pd
from pathlib import Path
import importlib.util
import sys
from datetime import datetime

# Adjust the filename if there's a typo (e.g., "03_Data_Prepareration.py")
SCRIPT_PATH = Path(__file__).parent.parent / "Prediction_App" / "03_Data_Preparation.py"


def test_data_preparation_appends_new_week(mocker, tmp_path):
    # Mock raw input data
    raw_data = pd.DataFrame({
        'year': [2026, 2026],
        'week': ['W3', 'W3'],
        'vegetable': ['Cabbage', 'Carrot'],
        'Badulla_actual_class': ['normal', 'flood_risk'],
        'Hambantota_actual_class': ['drought', 'normal'],
        'USD_LKR_avg': [310.0, 310.0],
        'RateChange_avg': [3.33, 3.33],
    })
    cities = ['Badulla', 'Hambantota', 'Jaffna', 'Kurunegala', 'Matale', 'Nuwara_Eliya', 'Ratnapura']
    probs = ['prob_drought', 'prob_flood_risk', 'prob_normal']
    for city in cities:
        for prob in probs:
            raw_data[f'{city}_{prob}'] = 0.1
    for city in cities:
        raw_data[f'{city}_actual_class'] = raw_data['Badulla_actual_class']

    # Write raw file
    raw_file = tmp_path / "Vegetable_prices_weekly.csv"
    raw_data.to_csv(raw_file, index=False)

    # Create existing backend file
    backend_existing = pd.DataFrame({
        'year': [2026, 2026],
        'week_num': [1, 2],
        'vegetable': [3, 3],
        'Badulla_actual_class': [0, 0],
    })
    backend_file = tmp_path / "Backend_Data.csv"
    backend_existing.to_csv(backend_file, index=False)

    # Mock datetime.now inside the script's module
    fake_now = datetime(2026, 1, 20)  # week 3
    mocker.patch("data_prep.datetime.now", return_value=fake_now)

    # Prepare globals
    script_globals = {
        'pd': pd,
        'os': __import__('os'),
        'datetime': __import__('datetime'),
        '__name__': '__main__',
        'INPUT_PATH': str(raw_file),
        'OUTPUT_PATH': str(backend_file),
        'USE_CURRENT_WEEK': True,
    }

    # Execute script
    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        script_code = f.read()
    exec(script_code, script_globals)

    # Read updated backend
    result_df = pd.read_csv(backend_file)
    assert len(result_df) == 4  # 2 existing + 2 new
    new_rows = result_df[result_df['week_num'] == 3]
    assert len(new_rows) == 2
    # Check mapping: normal→0, flood_risk→1
    cabbage_row = new_rows[new_rows['vegetable'] == 3]
    assert cabbage_row['Badulla_actual_class'].iloc[0] == 0
    carrot_row = new_rows[new_rows['vegetable'] == 4]
    assert carrot_row['Badulla_actual_class'].iloc[0] == 1
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import sys
from datetime import datetime

SCRIPT_PATH = Path(__file__).parent.parent / "Prediction_App" / "04_Data_Collector.py"


def test_data_collector_fills_derived_features(mocker, tmp_path):
    cities = ['Badulla', 'Hambantota', 'Jaffna', 'Kurunegala', 'Matale', 'Nuwara_Eliya', 'Ratnapura']
    probs = ['prob_drought', 'prob_flood_risk', 'prob_normal']

    # Historical rows
    hist_data = {
        'year': [2026, 2026],
        'week': ['W1', 'W2'],
        'week_num': [1, 2],
        'vegetable': [3, 3],
        'price': [100.0, 110.0],
        'USD_LKR_avg': [300.0, 305.0],
        'RateChange_avg': [1.0, 1.67],
    }
    # New rows (price null)
    new_data = {
        'year': [2026, 2026],
        'week': ['W3', 'W3'],
        'week_num': [3, 3],
        'vegetable': [3, 4],
        'price': [np.nan, np.nan],
        'USD_LKR_avg': [310.0, 310.0],
        'RateChange_avg': [3.33, 3.33],
    }

    for city in cities:
        for prob in probs:
            hist_data[f'{city}_{prob}'] = [0.1, 0.1]
            new_data[f'{city}_{prob}'] = [0.1, 0.2]
    for city in cities:
        hist_data[f'{city}_actual_class'] = [0, 0]
        new_data[f'{city}_actual_class'] = [0, 1]

    full_data = pd.concat([pd.DataFrame(hist_data), pd.DataFrame(new_data)], ignore_index=True)

    # Write to temporary backend file
    backend_file = tmp_path / "Backend_Data.csv"
    full_data.to_csv(backend_file, index=False)

    # Mock datetime.now inside the script's module
    fake_now = datetime(2026, 1, 20)  # week 3
    mocker.patch("data_collector.datetime.now", return_value=fake_now)

    # Prepare globals
    script_globals = {
        'pd': pd,
        'np': np,
        'os': __import__('os'),
        '__name__': '__main__',
        'INPUT_PATH': str(backend_file),
        'OUTPUT_PATH': str(backend_file),
    }

    # Execute script
    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        script_code = f.read()
    exec(script_code, script_globals)

    # Read updated backend
    result_df = pd.read_csv(backend_file)

    # Check new rows
    new_rows = result_df[result_df['price'].isna()]
    assert len(new_rows) == 2

    # Cabbage (veg=3) at week 3
    cabbage = new_rows[new_rows['vegetable'] == 3].iloc[0]
    assert cabbage['price_lag1'] == 110.0
    assert cabbage['price_lag2'] == 100.0
    assert cabbage['price_roll_mean_4'] == 105.0
    assert cabbage['avg_prob_drought'] == pytest.approx(0.1)
    assert cabbage['count_drought'] == 0
    assert cabbage['usd_x_avg_drought'] == cabbage['USD_LKR_avg'] * cabbage['avg_prob_drought']
    assert not pd.isna(cabbage['USD_LKR_avg_sq'])
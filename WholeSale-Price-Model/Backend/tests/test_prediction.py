import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import sys
from datetime import datetime

SCRIPT_PATH = Path(__file__).parent.parent / "Prediction_App" / "05_Input_Feature_Values.py"


def test_prediction_updates_prices(mocker, tmp_path):
    # Dummy model
    class DummyModel:
        def predict(self, X):
            return np.arange(200, 200 + len(X))

    mocker.patch("joblib.load", return_value=DummyModel())

    cities = ['Badulla', 'Hambantota', 'Jaffna', 'Kurunegala', 'Matale', 'Nuwara_Eliya', 'Ratnapura']
    probs = ['prob_drought', 'prob_flood_risk', 'prob_normal']

    data = {
        'year': [2026, 2026, 2026],
        'week_num': [3, 3, 3],
        'vegetable': [3, 4, 5],
        'price': [np.nan, np.nan, np.nan],
        'USD_LKR_avg': [310.0, 310.0, 310.0],
        'RateChange_avg': [3.33, 3.33, 3.33],
    }
    derived_cols = ['price_lag1', 'price_lag2', 'price_roll_mean_4', 'price_roll_std_4',
                    'avg_prob_drought', 'avg_prob_flood_risk', 'avg_prob_normal',
                    'max_prob_drought', 'min_prob_drought', 'std_prob_drought',
                    'usd_x_avg_drought', 'usd_x_avg_flood', 'usd_x_avg_normal',
                    'week_x_avg_drought', 'week_x_avg_flood',
                    'USD_LKR_avg_sq', 'RateChange_avg_%_sq', 'week_num_sq',
                    'week_sin', 'week_cos', 'veg_2', 'veg_3', 'veg_4', 'veg_5', 'veg_6',
                    'count_drought', 'count_normal', 'count_flood_risk']
    for col in derived_cols:
        data[col] = [0.0] * 3

    for city in cities:
        for prob in probs:
            data[f'{city}_{prob}'] = [0.1, 0.2, 0.15]
    for city in cities:
        data[f'{city}_actual_class'] = [0, 1, 0]

    df = pd.DataFrame(data)

    # Write backend file
    backend_file = tmp_path / "Backend_Data.csv"
    df.to_csv(backend_file, index=False)

    # Mock datetime.now
    fake_now = datetime(2026, 1, 20)  # week 3
    mocker.patch("prediction.datetime.now", return_value=fake_now)

    predictions_csv = tmp_path / "Predictions.csv"

    script_globals = {
        'pd': pd,
        'np': np,
        'joblib': __import__('joblib'),
        'os': __import__('os'),
        'sys': sys,
        'datetime': __import__('datetime'),
        '__name__': '__main__',
        'model_path': "dummy_path",
        'backend_data_path': str(backend_file),
        'out_file': str(predictions_csv),
    }

    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        script_code = f.read()
    exec(script_code, script_globals)

    # Read updated backend
    result_df = pd.read_csv(backend_file)
    updated = result_df[(result_df['year'] == 2026) & (result_df['week_num'] == 3)]
    assert not updated['price'].isna().any()
    expected_prices = [200, 201, 202]
    assert list(updated.sort_values('vegetable')['price'].values) == expected_prices

    # Check predictions CSV
    pred_df = pd.read_csv(predictions_csv)
    assert len(pred_df) == 3
    assert list(pred_df['Predicted Price'].values) == expected_prices
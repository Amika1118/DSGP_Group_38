import pytest
import pandas as pd
import json
from pathlib import Path
import importlib.util
import sys
from datetime import datetime

SCRIPT_PATH = Path(__file__).parent.parent / "Prediction_App" / "02_Web_Scraper.py"


def test_web_scraper_updates_csv(mocker, tmp_path):
    # Mock subprocess.run
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    # Create temporary forecast JSON
    forecast_data = {"predicted_rate": 350.5}
    forecast_json_path = tmp_path / "usd_lkr_forecast.json"
    with open(forecast_json_path, 'w') as f:
        json.dump(forecast_data, f)

    # Create temporary historical CSV
    hist_data = pd.DataFrame({
        'Date': ['01/01/2026', '01/08/2026'],
        'DollarRate': [300.0, 310.0],
        'RateChange %': ['1.0%', '3.33%']
    })
    csv_path = tmp_path / "USD_LKR_Historical_Data.csv"
    hist_data.to_csv(csv_path, index=False)

    # Mock datetime.now inside the script's module
    fake_now = datetime(2026, 1, 15)  # should be week 3 (Jan 15–21)
    mocker.patch("web_scraper.datetime.now", return_value=fake_now)

    # Prepare globals with overridden paths
    script_globals = {
        'pd': pd,
        'subprocess': __import__('subprocess'),
        'json': json,
        'datetime': __import__('datetime'),
        'os': __import__('os'),
        'sys': sys,
        '__name__': '__main__',
        'scraper_path': "dummy_path",
        'forecast_json_path': str(forecast_json_path),
        'csv_path': str(csv_path),
    }

    # Execute script
    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        script_code = f.read()
    exec(script_code, script_globals)

    # Read updated CSV
    result_df = pd.read_csv(csv_path)

    target_date = '01/15/2026'
    assert target_date in result_df['Date'].values
    row = result_df[result_df['Date'] == target_date].iloc[0]
    assert row['DollarRate'] == 350.5
    expected_change = ((350.5 - 310.0) / 310.0) * 100
    assert float(row['RateChange %'].rstrip('%')) == pytest.approx(expected_change, rel=1e-2)
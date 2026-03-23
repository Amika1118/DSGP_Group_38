import pytest
import json
from pathlib import Path
import importlib.util
import sys

SCRIPT_PATH = Path(__file__).parent.parent / "USD-LKR_Scraper" / "Scraper.py"

spec = importlib.util.spec_from_file_location("scraper", SCRIPT_PATH)
scraper = importlib.util.module_from_spec(spec)
sys.modules["scraper"] = scraper
spec.loader.exec_module(scraper)


def test_scraper_parses_rate_correctly(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = b"""
    <html>
        <body>
            <h3>USD to LKR Forecast for This Week</h3>
            <p>The forecast is Rs 350.50 and we expect stability.</p>
        </body>
    </html>
    """
    mocker.patch("requests.get", return_value=mock_response)

    result = scraper.scrape_this_week_forecast("http://fake.url")
    assert result["predicted_rate"] == 350.5
    assert "Rs 350.50" in result["forecast_text"]


def test_scraper_handles_missing_heading(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = b"<html><body><p>No heading</p></body></html>"
    mocker.patch("requests.get", return_value=mock_response)

    result = scraper.scrape_this_week_forecast("http://fake.url")
    assert "error" in result
    assert "Could not find the H3 heading" in result["error"]


def test_save_to_json(mocker, tmp_path):
    file_path = tmp_path / "subdir" / "test.json"
    assert not file_path.parent.exists()

    mock_makedirs = mocker.patch("os.makedirs")
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    data = {"predicted_rate": 350.5, "forecast_text": "Rs 350.50"}
    scraper.save_to_json(data, str(file_path))

    mock_makedirs.assert_called_once_with(str(file_path.parent))
    mock_open.assert_called_once_with(str(file_path), 'w', encoding='utf-8')
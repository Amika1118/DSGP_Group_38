import pytest
from pathlib import Path
import importlib.util
import sys
from unittest.mock import call

SCRIPT_PATH = Path(__file__).parent.parent / "Run" / "main.py"

spec = importlib.util.spec_from_file_location("main", SCRIPT_PATH)
main_module = importlib.util.module_from_spec(spec)
sys.modules["main"] = main_module


def test_main_runs_all_scripts(mocker):
    mock_run = mocker.patch("subprocess.run")
    spec.loader.exec_module(main_module)

    expected_scripts = [
        "01_Combine_Weather_Data.py",
        "02_Web_Scraper.py",
        "03_Data_Prepareration.py",   # match actual filename (with typo if present)
        "04_Data_Collector.py",
        "05_Input_Feature_Values.py",
        "06_Prediction.py"
    ]

    assert mock_run.call_count == 6
    calls = mock_run.call_args_list
    base_dir = Path(__file__).parent.parent / "Prediction_App"

    for i, script in enumerate(expected_scripts):
        expected_path = str((base_dir / script).resolve())
        args = calls[i][0][0]
        actual_path = str(Path(args[1]).resolve())
        assert args[0] == "python"
        assert actual_path == expected_path
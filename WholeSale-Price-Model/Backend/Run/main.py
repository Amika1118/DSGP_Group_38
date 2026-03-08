import subprocess
import os

# path to Prediction_App folder
base_dir = os.path.join(os.path.dirname(__file__), "..", "Prediction_App")

scripts = [
    "01_Combine_Weather_Data.py",
    "02_Web_Scraper.py",
    "03_Data_Prepareration.py",
    "04_Data_Collector.py",
    "05_Input_Feature_Values.py",
    "06_Prediction.py"
]

for script in scripts:
    script_path = os.path.join(base_dir, script)
    print(f"Running {script}...")
    subprocess.run(["python", script_path])
import sys
import asyncio
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Set asyncio policy for Windows to avoid zmq "ProactorEventLoop" warning
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def run_notebook(notebook_path: Path):
    """Execute a Jupyter notebook in-memory (no output file saved)."""
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)

    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Determine kernel name (default to python3)
    kernel_name = nb.metadata.get('kernelspec', {}).get('name', 'python3')

    # Create an executor that runs all cells
    executor = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)

    try:
        # Execute the notebook. The 'path' metadata tells the kernel where to look for files.
        executor.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        print(f"✅ Notebook '{notebook_path.name}' executed successfully.")
    except Exception as e:
        print(f"❌ Error during notebook execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Build the absolute path to the notebook relative to this script
    script_dir = Path(__file__).resolve().parent
    # Adjust the relative path: ..\..\Notebooks\01_Data_Preparation.ipynb
    notebook_relative = Path("..") / ".." / "Notebooks" / "01_Data_Preparation.ipynb"
    notebook_path = (script_dir / notebook_relative).resolve()

    run_notebook(notebook_path)
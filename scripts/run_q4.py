"""Canonical entrypoint for q4-style aggregation workflow.

Wrapper for compatibility with legacy file `q4.py`.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).resolve().parent.parent / "q4.py"
    runpy.run_path(str(legacy_script), run_name="__main__")

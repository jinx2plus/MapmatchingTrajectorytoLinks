"""Canonical entrypoint for q3-style aggregation workflow.

Wrapper for compatibility with legacy file `q3.py`.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).resolve().parent.parent / "q3.py"
    runpy.run_path(str(legacy_script), run_name="__main__")

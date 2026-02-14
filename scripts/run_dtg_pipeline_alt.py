"""Canonical entrypoint for the alternative DTG processing flow.

This is the clean wrapper for `untitled1.py`.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).resolve().parent.parent / "untitled1.py"
    runpy.run_path(str(legacy_script), run_name="__main__")

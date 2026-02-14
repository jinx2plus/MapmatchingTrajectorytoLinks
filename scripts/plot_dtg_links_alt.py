"""Canonical entrypoint for the secondary plotting workflow.

Wrapper for legacy file `plot6.py`.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).resolve().parent.parent / "plot6.py"
    runpy.run_path(str(legacy_script), run_name="__main__")

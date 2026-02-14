"""Canonical entrypoint for the main DTG processing flow.

This repository keeps legacy scripts at the repository root for historical reasons.
Run this file instead for a cleaner project entrypoint.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).resolve().parent.parent / "untitled2.py"
    runpy.run_path(str(legacy_script), run_name="__main__")

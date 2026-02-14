from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def run_script(name: str) -> None:
    script = Path(__file__).resolve().parent.parent / name
    runpy.run_path(str(script), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="JB DTG processing launcher")
    parser.add_argument(
        "mode",
        choices=[
            "pipeline",
            "pipeline-alt",
            "q3",
            "q4",
            "plot",
            "plot-alt",
        ],
        help="Select workflow to run",
    )
    args = parser.parse_args()

    mapping = {
        "pipeline": "../untitled2.py",
        "pipeline-alt": "../untitled1.py",
        "q3": "../q3.py",
        "q4": "../q4.py",
        "plot": "../plot5.py",
        "plot-alt": "../plot6.py",
    }

    target = Path(__file__).resolve().parent / mapping[args.mode]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

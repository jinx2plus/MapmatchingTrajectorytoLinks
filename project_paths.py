from pathlib import Path
from functools import lru_cache


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIRS = [
    REPO_ROOT / "data",
    REPO_ROOT / "data" / "region",
    REPO_ROOT / "data" / "roi",
    REPO_ROOT / "tools",
]


def get_candidate_paths(relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        yield path
    else:
        yield REPO_ROOT / path
        for data_dir in DATA_DIRS:
            yield data_dir / path


@lru_cache(maxsize=None)
def resolve_repo_file(relative_path, *, required=True):
    for candidate in get_candidate_paths(relative_path):
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Required file not found: {relative_path}. Searched in repository root and data/tools folders."
        )
    return REPO_ROOT / relative_path

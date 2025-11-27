import os
import sys
import json
from pathlib import Path
from typing import Optional, List
import pytest

# Ensure project root is importable (v2/)
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _find_latest_segments_jsonl(root: Path) -> Optional[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return None
    candidates: List[Path] = list(runs_dir.glob("**/segments.jsonl"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def pytest_addoption(parser):
    parser.addoption(
        "--segments",
        action="store",
        default=None,
        help="Path to segments.jsonl to drive retrieval test",
    )


@pytest.fixture(scope="session")
def segments_path(pytestconfig) -> Optional[Path]:
    # Priority: CLI option > ENV var > auto-discover latest under runs/**/segments.jsonl
    cli = pytestconfig.getoption("--segments")
    if cli:
        p = Path(cli).expanduser().resolve()
        return p if p.exists() else None

    env = os.getenv("SEGMENTS_JSONL")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.exists() else None

    return _find_latest_segments_jsonl(PROJECT_ROOT)


@pytest.fixture(scope="session")
def segments(segments_path) -> list:
    if not segments_path:
        pytest.skip("No segments.jsonl found. Provide one via --segments or SEGMENTS_JSONL.")
    items = []
    with segments_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

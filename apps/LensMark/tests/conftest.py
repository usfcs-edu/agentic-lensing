import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "nine"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden"


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="rewrite tests/golden/sha256.json from the current renderer output")


@pytest.fixture
def update_golden(request) -> bool:
    return bool(request.config.getoption("--update-golden"))


def _copy_examples(dst: Path, *, with_json: bool) -> Path:
    assert (EXAMPLES / "deck-01.png").exists(), "run `lensmark examples build` first"
    dst.mkdir()
    for p in EXAMPLES.iterdir():
        if p.suffix == ".png" and ".annot" not in p.name and ".mask" not in p.name:
            shutil.copy(p, dst / p.name)
        elif with_json and p.name.endswith(".lensmark.json"):
            shutil.copy(p, dst / p.name)
    shutil.copy(EXAMPLES / "lensmark.config.json", dst / "lensmark.config.json")
    return dst


@pytest.fixture
def nine(tmp_path: Path) -> Path:
    """A throw-away copy of examples/nine: images + lensmark.config.json only (no annotation JSON)."""
    return _copy_examples(tmp_path / "nine", with_json=False)


@pytest.fixture
def nine_full(tmp_path: Path) -> Path:
    """A throw-away copy of examples/nine including the hand-authored <id>.lensmark.json files."""
    return _copy_examples(tmp_path / "nine", with_json=True)

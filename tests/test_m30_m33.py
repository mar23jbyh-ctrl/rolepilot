"""Regression checks for removed obsolete application symbols."""

import glob
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize(
    "symbol",
    ["volatility", "temperature_mode", "active_question", "_run_is_volatile", "average_first_three"],
)
def test_m33_dead_symbols_are_gone(symbol):
    hits = []
    for path in glob.glob(str(PROJECT_ROOT / "app" / "**" / "*.py"), recursive=True):
        text = Path(path).read_text(encoding="utf-8")
        if symbol in text:
            hits.append(Path(path).name)
    assert hits == [], f"{symbol} 应已清理，仍出现在 {hits}"

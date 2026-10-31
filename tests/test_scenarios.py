from pathlib import Path

import pytest

from app.listening.policy import Policy
from app.replay import find_scenarios, run_scenario

ROOT = Path(__file__).resolve().parents[1]
POLICY = Policy.load(ROOT / "policy.toml")
SCENARIOS = find_scenarios([str(ROOT / "scenarios")])


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario(path):
    result = run_scenario(path, POLICY)
    assert result.passed, "\n" + result.report()

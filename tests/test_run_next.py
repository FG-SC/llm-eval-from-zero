"""`python -m src.run --next` picks the right run. The quota check is stubbed: these tests never call the API."""
import json

import pytest

from src import run as runner
from src.settings import CONFIG_DIR, RESULTS_DIR, load_yaml

RUNS = load_yaml(CONFIG_DIR / "runs.yaml")
MODELS = load_yaml(CONFIG_DIR / "models.yaml")


def test_every_full_run_names_a_smoke_test_with_the_same_task():
    for name, cfg in RUNS.items():
        if cfg["limit"] is None:
            smoke = RUNS[cfg["smoke"]]
            assert smoke["task"] == cfg["task"] and smoke["limit"] == 10, name


def test_pending_runs_follow_priority_and_skip_finished_ones():
    pending = runner.pending_full_runs(RUNS)
    assert [RUNS[n]["priority"] for n in pending] == sorted(RUNS[n]["priority"] for n in pending)
    assert all(not (RESULTS_DIR / n / "results.json").exists() for n in pending)


def test_saved_smoke_tests_pass():
    for name, cfg in RUNS.items():
        if cfg["limit"] is None and (RESULTS_DIR / cfg["smoke"] / "run_summary.json").exists():
            assert runner.smoke_check(cfg["smoke"])[0], cfg["smoke"]


def test_failed_or_missing_smoke_test_blocks_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    assert runner.smoke_check("smoke_x") == (False, "smoke test smoke_x has not run yet")
    (tmp_path / "smoke_x").mkdir()
    (tmp_path / "smoke_x" / "run_summary.json").write_text(json.dumps(
        {"unusable_answers": {"rate": 0.1}, "metrics": {}, "extraction_rate": {"strict-match": 1.0}}), encoding="utf-8")
    assert runner.smoke_check("smoke_x")[0] is False


def test_next_run_is_the_first_pending_run_that_fits(monkeypatch):
    pending = runner.pending_full_runs(RUNS)
    if len(pending) < 2:
        pytest.skip("fewer than two pending runs")
    monkeypatch.setattr(runner, "smoke_check", lambda smoke: (True, "stub"))
    asked = iter(pending)  # every smoke passes, so the quota is checked run by run, in priority order
    monkeypatch.setattr(runner.budget, "check", lambda *a, **k: (next(asked) == pending[1], "stub"))
    assert runner.next_run(RUNS, MODELS, 0.5) == pending[1]


def test_nothing_runs_when_nothing_fits(monkeypatch):
    monkeypatch.setattr(runner, "smoke_check", lambda smoke: (True, "stub"))
    monkeypatch.setattr(runner.budget, "check", lambda *a, **k: (False, "stub"))
    assert runner.next_run(RUNS, MODELS, 0.5) is None

"""`python -m src.run --next` picks the right run. The quota check is stubbed: these tests never call the API.
Every run in this repo has finished, so the tests point the runner at an empty results folder."""
import json

from src import run as runner
from src.settings import CONFIG_DIR, RESULTS_DIR, load_yaml

RUNS = load_yaml(CONFIG_DIR / "runs.yaml")
MODELS = load_yaml(CONFIG_DIR / "models.yaml")
FULL = sorted((n for n, c in RUNS.items() if c["limit"] is None), key=lambda n: RUNS[n]["priority"])


def _empty_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "cached_answers", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "smoke_check", lambda smoke: (True, "stub"))


def test_every_full_run_names_a_smoke_test_with_the_same_task():
    for name, cfg in RUNS.items():
        if cfg["limit"] is None:
            smoke = RUNS[cfg["smoke"]]
            assert smoke["task"] == cfg["task"] and smoke["limit"] == 10, name


def test_pending_runs_follow_priority_and_skip_finished_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    assert runner.pending_full_runs(RUNS) == FULL
    (tmp_path / FULL[0]).mkdir()
    (tmp_path / FULL[0] / "results.json").write_text("{}", encoding="utf-8")
    assert runner.pending_full_runs(RUNS) == FULL[1:]


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


def test_next_run_is_the_first_pending_run_that_fits(tmp_path, monkeypatch):
    _empty_results(tmp_path, monkeypatch)
    asked = iter(FULL)  # every smoke passes, so the quota is checked run by run, in priority order
    monkeypatch.setattr(runner.budget, "check", lambda *a, **k: (next(asked) == FULL[1], "stub"))
    assert runner.next_run(RUNS, MODELS, 0.5) == FULL[1]


def test_nothing_runs_when_nothing_fits(tmp_path, monkeypatch):
    _empty_results(tmp_path, monkeypatch)
    checked = []
    monkeypatch.setattr(runner.budget, "check", lambda *a, **k: (checked.append(1), (False, "stub"))[1])
    assert runner.next_run(RUNS, MODELS, 0.5) is None
    assert len(checked) == len(FULL)

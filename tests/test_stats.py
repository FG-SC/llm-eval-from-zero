"""Known values for the statistics."""
import pytest

from src import stats


def test_wilson_known_value():
    lo, hi = stats.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=0.001) and hi == pytest.approx(0.596, abs=0.001)


def test_wilson_halfwidth_matches_interval():
    lo, hi = stats.wilson(50, 100)
    assert stats.wilson_halfwidth(0.5, 100) == pytest.approx((hi - lo) / 2, abs=1e-9)


def test_halfwidth_shrinks_with_n():
    assert stats.wilson_halfwidth(0.9, 1000) < stats.wilson_halfwidth(0.9, 100) < stats.wilson_halfwidth(0.9, 10)


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 6) == pytest.approx(0.03125)  # 2 * 0.5**6
    assert stats.mcnemar_exact(3, 3) == pytest.approx(1.0)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    a = [1] * 90 + [0] * 10
    b = [1] * 80 + [0] * 20
    first = stats.bootstrap_paired_diff(a, b, reps=2000)
    assert first == stats.bootstrap_paired_diff(a, b, reps=2000)
    diff, lo, hi = first
    assert diff == pytest.approx(-0.10) and lo <= diff <= hi

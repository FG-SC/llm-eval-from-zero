"""Statistics used in the notebook. Closed-form or seeded, so every number is reproducible."""
import math
import random

from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

Z95 = 1.959963984540054
BOOTSTRAP_SEED = 2026


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (95% by default)."""
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def wilson_halfwidth(p: float, n: int, z: float = Z95) -> float:
    """Half-width of the Wilson interval for an accuracy p measured on n questions (closed form)."""
    denom = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs (b: only A right, c: only B right)."""
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def bootstrap_paired_diff(a: list[int], b: list[int], reps: int = 10_000, seed: int = BOOTSTRAP_SEED,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean(b) - mean(a) on paired 0/1 outcomes, with a percentile bootstrap interval."""
    if len(a) != len(b) or not a:
        raise ValueError("paired outcomes must have the same, non-zero length")
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(reps * alpha / 2)]
    hi = diffs[int(reps * (1 - alpha / 2)) - 1]
    return (sum(b) - sum(a)) / n, lo, hi

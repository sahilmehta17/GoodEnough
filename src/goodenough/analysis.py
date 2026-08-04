"""
Statistics for the map. Pure standard library: no numpy, scipy, or pandas, so
it runs on machines where compiled binaries are blocked.

The core question is a PAIRED comparison: both models answer the same items, so
we compare them item by item, not as two separate averages. For each item we
have local_correct and hosted_correct, each 0 or 1.

We report the accuracy difference delta = p_local - p_hosted and a confidence
interval for it, then classify each slice as non-inferior, below margin, or
inconclusive at the predeclared margin (PREREGISTRATION.md sections 5, 8, 11).

Primary interval: an exact conditional interval on the discordant pairs
(McNemar-style Clopper-Pearson), the "exact matched-binary interval" named in
the pre-registration. Secondary: a paired bootstrap as a cross-check.
"""

from __future__ import annotations

import math
import random

# --------------------------------------------------------------------------
# Regularized incomplete beta and its inverse (for exact Clopper-Pearson).
# Continued-fraction method, standard numerical recipe. No dependencies.
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for mi in range(1, MAXIT + 1):
        m2 = 2 * mi
        aa = mi * (b - mi) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + mi) * (qab + mi) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of betainc in x, by bisection."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_lower(k: int, n: int, tail: float) -> float:
    """Clopper-Pearson one-sided lower bound for a binomial proportion."""
    if k == 0:
        return 0.0
    return beta_quantile(tail, k, n - k + 1)


def cp_upper(k: int, n: int, tail: float) -> float:
    """Clopper-Pearson one-sided upper bound for a binomial proportion."""
    if k == n:
        return 1.0
    return beta_quantile(1.0 - tail, k + 1, n - k)


# --------------------------------------------------------------------------
# Paired binary comparison
# --------------------------------------------------------------------------

def accuracy(vals: list[int]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def discordance(local: list[int], hosted: list[int]) -> dict:
    """How often the two models disagree. Drives whether n is adequate."""
    n = len(local)
    dis = sum(1 for l, h in zip(local, hosted) if l != h)
    return {"n": n, "discordant": dis, "rate": (dis / n if n else 0.0)}


def paired_binary_interval(local: list[int], hosted: list[int], tail: float = 0.05) -> dict:
    """
    Exact conditional interval for delta = p_local - p_hosted.

    Conditions on the discordant pairs (McNemar). b = local right / hosted wrong,
    c = local wrong / hosted right. With tail=0.05 the returned lower and upper
    are the one-sided 95% bounds (together a two-sided 90% interval).
    """
    n = len(local)
    b = sum(1 for l, h in zip(local, hosted) if l == 1 and h == 0)
    c = sum(1 for l, h in zip(local, hosted) if l == 0 and h == 1)
    m = b + c
    delta = (b - c) / n if n else 0.0
    if m == 0:
        # Models never disagree: the accuracy difference is exactly 0 here.
        return {"delta": 0.0, "lower": 0.0, "upper": 0.0, "b": b, "c": c, "m": m, "n": n}
    pi_lo = cp_lower(b, m, tail)
    pi_hi = cp_upper(b, m, tail)
    lower = (2.0 * pi_lo - 1.0) * (m / n)
    upper = (2.0 * pi_hi - 1.0) * (m / n)
    return {"delta": delta, "lower": lower, "upper": upper, "b": b, "c": c, "m": m, "n": n}


def bootstrap_paired(local: list[int], hosted: list[int], iters: int = 10000,
                     seed: int = 42, tail: float = 0.05) -> dict:
    """Paired bootstrap cross-check: resample items, recompute delta."""
    n = len(local)
    if n == 0:
        return {"lower": 0.0, "upper": 0.0}
    diffs = [l - h for l, h in zip(local, hosted)]
    rng = random.Random(seed)
    deltas = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        deltas.append(s / n)
    deltas.sort()
    lo = deltas[max(0, int(tail * iters) - 1)]
    hi = deltas[min(iters - 1, int((1.0 - tail) * iters))]
    return {"lower": lo, "upper": hi}


def verdict(lower: float, upper: float, margin: float) -> str:
    """
    Three-way classification at a non-inferiority margin (e.g. 0.10).
    lower/upper are the one-sided 95% bounds on delta = p_local - p_hosted.
    """
    if lower > -margin:
        return "non_inferior"
    if upper < -margin:
        return "below_margin"
    return "inconclusive"


def slice_result(local: list[int], hosted: list[int], margin: float,
                 with_bootstrap: bool = True) -> dict:
    """Full per-slice result: accuracies, the exact interval, verdict, and a check."""
    exact = paired_binary_interval(local, hosted)
    out = {
        "n": len(local),
        "acc_local": accuracy(local),
        "acc_hosted": accuracy(hosted),
        "delta": exact["delta"],
        "ci_lower": exact["lower"],
        "ci_upper": exact["upper"],
        "b_local_right_hosted_wrong": exact["b"],
        "c_local_wrong_hosted_right": exact["c"],
        "discordant": exact["m"],
        "verdict": verdict(exact["lower"], exact["upper"], margin),
        "margin": margin,
    }
    if with_bootstrap:
        boot = bootstrap_paired(local, hosted)
        out["bootstrap_lower"] = boot["lower"]
        out["bootstrap_upper"] = boot["upper"]
        out["bootstrap_verdict"] = verdict(boot["lower"], boot["upper"], margin)
    return out

# dev-only. scipy is not a runtime dependency of this project (see
# src/goodenough/analysis.py's module docstring); this file only checks that
# the hand-rolled implementation agrees with a trusted reference. It must
# skip cleanly on a bare install so the core test suite still passes without
# scipy.
import math
import unittest

try:
    from scipy import optimize as sp_optimize
    from scipy.special import betainc as sp_betainc
    from scipy.stats import beta as sp_beta
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.goodenough import analysis

BETAINC_TOLERANCE = 1e-10
CP_TOLERANCE = 1e-9
LOGISTIC_SLOPE_TOLERANCE = 1e-3  # iterative fit vs iterative fit, looser by design


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy not installed; dev-only cross-check")
class BetaincVsScipyTests(unittest.TestCase):
    """
    analysis.betainc is a continued-fraction regularized incomplete beta,
    the numerical core every Clopper-Pearson bound in this project rests on.
    scipy.special.betainc is the trusted reference.
    """

    def test_grid_of_a_b_x(self):
        max_dev = 0.0
        a_values = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
        b_values = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
        x_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for a in a_values:
            for b in b_values:
                for x in x_values:
                    ours = analysis.betainc(a, b, x)
                    theirs = float(sp_betainc(a, b, x))
                    dev = abs(ours - theirs)
                    max_dev = max(max_dev, dev)
                    self.assertLess(
                        dev, BETAINC_TOLERANCE,
                        f"betainc(a={a}, b={b}, x={x}): ours={ours!r} scipy={theirs!r} "
                        f"deviation={dev!r}",
                    )
        print(f"\n[betainc] max deviation vs scipy over "
             f"{len(a_values)*len(b_values)*len(x_values)} grid points: {max_dev:.3e}")

    def test_boundary_x_values(self):
        for a, b in [(1.0, 1.0), (0.5, 5.0), (50.0, 2.0)]:
            self.assertEqual(analysis.betainc(a, b, 0.0), 0.0)
            self.assertEqual(analysis.betainc(a, b, 1.0), 1.0)


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy not installed; dev-only cross-check")
class ClopperPearsonVsScipyTests(unittest.TestCase):
    """
    analysis.cp_lower / cp_upper are the one-sided exact Clopper-Pearson
    bounds the primary non-inferiority interval is built from
    (PREREGISTRATION.md section 8). scipy.stats.beta.ppf is the reference:
    the standard textbook identity is
        CP_lower(k, n, tail) = Beta.ppf(tail, k, n - k + 1)
        CP_upper(k, n, tail) = Beta.ppf(1 - tail, k + 1, n - k)
    """

    def test_grid_of_k_n_including_edges(self):
        max_dev_lower = 0.0
        max_dev_upper = 0.0
        tail = 0.05
        n_values = [1, 2, 5, 10, 20, 50, 100, 150]
        for n in n_values:
            for k in range(0, n + 1):  # includes k=0 and k=n, the two edge cases
                ours_lo = analysis.cp_lower(k, n, tail)
                ours_hi = analysis.cp_upper(k, n, tail)

                if k == 0:
                    theirs_lo = 0.0
                else:
                    theirs_lo = float(sp_beta.ppf(tail, k, n - k + 1))
                if k == n:
                    theirs_hi = 1.0
                else:
                    theirs_hi = float(sp_beta.ppf(1.0 - tail, k + 1, n - k))

                dev_lo = abs(ours_lo - theirs_lo)
                dev_hi = abs(ours_hi - theirs_hi)
                max_dev_lower = max(max_dev_lower, dev_lo)
                max_dev_upper = max(max_dev_upper, dev_hi)

                self.assertLess(dev_lo, CP_TOLERANCE,
                                f"cp_lower(k={k}, n={n}): ours={ours_lo!r} "
                                f"scipy={theirs_lo!r}")
                self.assertLess(dev_hi, CP_TOLERANCE,
                                f"cp_upper(k={k}, n={n}): ours={ours_hi!r} "
                                f"scipy={theirs_hi!r}")

        total_points = sum(n + 1 for n in n_values)
        print(f"\n[cp_lower] max deviation vs scipy over {total_points} (k, n) "
             f"points: {max_dev_lower:.3e}")
        print(f"[cp_upper] max deviation vs scipy over {total_points} (k, n) "
             f"points: {max_dev_upper:.3e}")

    def test_very_small_n(self):
        # n=1 is the smallest possible non-inferiority sample; both k=0 and
        # k=1 must still agree with scipy exactly.
        for k in (0, 1):
            self.assertLess(abs(analysis.cp_lower(k, 1, 0.05) -
                               (0.0 if k == 0 else sp_beta.ppf(0.05, k, 1 - k + 1))),
                           CP_TOLERANCE)


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy not installed; dev-only cross-check")
class LogisticFitVsScipyTests(unittest.TestCase):
    """
    analysis.logistic_fit is a hand-rolled Newton-Raphson / IRLS fit for the
    GSM8K difficulty slope. This checks it against an independent iterative
    fit (scipy.optimize) rather than a closed form, so the tolerance is
    looser by design: both are numerical optimizers, not two evaluations of
    the same formula.
    """

    def _fit_with_scipy(self, xs, ys):
        def neg_log_likelihood(params):
            b0, b1 = params
            ll = 0.0
            for x, y in zip(xs, ys):
                z = b0 + b1 * x
                # log-sum-exp form, numerically stable for large |z|
                ll += y * z - math.log1p(math.exp(z)) if z < 0 else \
                    y * z - z - math.log1p(math.exp(-z))
            return -ll

        result = sp_optimize.minimize(neg_log_likelihood, x0=[0.0, 0.0], method="BFGS")
        return result.x[0], result.x[1]

    def test_slope_matches_scipy_optimize_on_a_synthetic_dataset(self):
        # Deterministic synthetic data: accuracy clearly falls as steps rise.
        rng_xs = list(range(1, 9)) * 10
        ys = []
        for x in rng_xs:
            # Higher step count -> lower probability of a correct answer.
            p_correct = max(0.05, 1.0 - 0.11 * x)
            ys.append(1 if (hash((x, len(ys))) % 100) / 100.0 < p_correct else 0)

        our_b0, our_b1 = analysis.logistic_fit(rng_xs, ys)
        sp_b0, sp_b1 = self._fit_with_scipy(rng_xs, ys)

        dev_intercept = abs(our_b0 - sp_b0)
        dev_slope = abs(our_b1 - sp_b1)
        print(f"\n[logistic_fit] deviation vs scipy.optimize: "
             f"intercept={dev_intercept:.3e} slope={dev_slope:.3e}")
        self.assertLess(dev_slope, LOGISTIC_SLOPE_TOLERANCE)
        self.assertLess(dev_intercept, LOGISTIC_SLOPE_TOLERANCE)


if __name__ == "__main__":
    unittest.main()

"""Unit tests: pure-logic pieces of the MC core.

These do not depend on the golden fixtures. They cover the angle/units helpers,
the Metropolis-Hastings acceptance rule, the Rubin-count scaling, and the
domain guards of the differential-rate density (the "integrand at known
limits").
"""
import math
import unittest

import numpy as np

import _harness as H

ev = H.load_events()


class WrapDegrees(unittest.TestCase):
    def test_known_values(self):
        # wrap_degrees(x) = ((x + 180) % 360) - 180, range [-180, 180)
        self.assertEqual(ev.wrap_degrees(0.0), 0.0)
        self.assertEqual(ev.wrap_degrees(180.0), -180.0)
        self.assertEqual(ev.wrap_degrees(-180.0), -180.0)
        self.assertAlmostEqual(ev.wrap_degrees(370.0), 10.0, places=12)
        self.assertAlmostEqual(ev.wrap_degrees(-190.0), 170.0, places=12)

    def test_in_range_and_idempotent(self):
        xs = np.linspace(-1000, 1000, 401)
        w = np.array([ev.wrap_degrees(x) for x in xs])
        self.assertTrue(np.all(w >= -180.0) and np.all(w < 180.0))
        # wrapping an already-wrapped value is a no-op
        w2 = np.array([ev.wrap_degrees(x) for x in w])
        np.testing.assert_allclose(w, w2, rtol=0, atol=1e-12)


class DistanceModulus(unittest.TestCase):
    def test_known(self):
        # mu0 = 5 (log10(d_pc) - 1); kpc_from_mu0(15) = 10**(15/5 - 2) = 10 kpc
        self.assertAlmostEqual(float(ev.kpc_from_mu0(15.0)), 10.0, places=10)
        self.assertAlmostEqual(float(ev.mu0_from_kpc(10.0)), 15.0, places=10)

    def test_roundtrip(self):
        mu0 = np.linspace(10.0, 18.0, 50)
        back = np.array([ev.mu0_from_kpc(ev.kpc_from_mu0(m)) for m in mu0])
        np.testing.assert_allclose(back, mu0, rtol=1e-12, atol=1e-10)


class MetropolisAcceptanceRule(unittest.TestCase):
    """The library rule (events.py make_events):
        accept = (new_lograte > lograte) or (exp(new_lograte - lograte) > u)
    """
    @staticmethod
    def accept(new, old, u):
        return (new > old) or (np.exp(new - old) > u)

    def test_uphill_always_accepted(self):
        for u in (1e-9, 0.5, 1.0 - 1e-9):
            self.assertTrue(self.accept(-10.0, -12.0, u))

    def test_equal_accepted(self):
        # exp(0) = 1 > u for any u drawn from [0, 1)
        self.assertTrue(self.accept(-5.0, -5.0, 0.999999))

    def test_downhill_threshold(self):
        new, old = -10.1, -10.0          # delta = -0.1 -> exp = 0.904837...
        ratio = math.exp(new - old)
        self.assertTrue(self.accept(new, old, ratio - 1e-6))   # u just below ratio
        self.assertFalse(self.accept(new, old, ratio + 1e-6))  # u just above ratio

    def test_minus_inf_proposal_rejected(self):
        # a proposal in the zero-density region (lograte = -inf) is never accepted
        self.assertFalse(self.accept(-np.inf, -20.0, 0.5))


class RatesToRubinCounts(unittest.TestCase):
    N_TRISTAR = 11_433_322_690
    HOURS_10YR = 24 * 365 * 10  # survey hours assumed by the scaling

    def test_scaling_formula(self):
        rates = {0: 1e-10, 1: 3e-10}              # mean = 2e-10 / hour / source
        got = ev.rates_to_rubin_counts(rates)
        mean = sum(rates.values()) / len(rates)
        expected = round(mean * self.N_TRISTAR * self.HOURS_10YR)
        self.assertEqual(got, expected)

    def test_linear_in_rate(self):
        base = {0: 1e-10, 1: 2e-10, 2: 3e-10}
        scaled = {k: 10.0 * v for k, v in base.items()}
        self.assertEqual(ev.rates_to_rubin_counts(scaled),
                         round(10 * sum(base.values()) / len(base)
                               * self.N_TRISTAR * self.HOURS_10YR))

    def test_custom_n_tristar(self):
        rates = {0: 2e-10}
        self.assertEqual(ev.rates_to_rubin_counts(rates, n_tristar=1),
                         round(2e-10 * self.HOURS_10YR))


class DensityDomainGuards(unittest.TestCase):
    """sample_density_single_source returns 0 outside the physical support;
    the log variant returns -inf there. params =
    [source_index, l, b, mu0, dl, umin, crossing_time].
    """
    def setUp(self):
        self.mw = ev.MilkyWayModel()
        self.mu0 = 14.5
        self.ds = float(ev.kpc_from_mu0(self.mu0))

    def _p(self, dl=None, umin=1.0, ct=200.0, l=2.0, b=-3.0):
        if dl is None:
            dl = 0.5 * self.ds
        return [0, l, b, self.mu0, dl, umin, ct]

    def test_dl_negative(self):
        self.assertEqual(ev.sample_density_single_source(self._p(dl=-1.0), self.mw), 0)

    def test_dl_beyond_source(self):
        self.assertEqual(ev.sample_density_single_source(self._p(dl=self.ds + 5.0), self.mw), 0)

    def test_umin_nonpositive(self):
        self.assertEqual(ev.sample_density_single_source(self._p(umin=0.0), self.mw), 0)

    def test_crossing_time_nonpositive(self):
        self.assertEqual(ev.sample_density_single_source(self._p(ct=0.0), self.mw), 0)

    def test_latitude_out_of_custom_bounds(self):
        # force the b-guard with a tight latitude window
        self.assertEqual(
            ev.sample_density_single_source(self._p(b=-3.5), self.mw, bbounds=(-1.0, 1.0)), 0)

    def test_log_is_minus_inf_on_zero(self):
        self.assertEqual(ev.sample_density_single_source_log(self._p(dl=-1.0), self.mw), -np.inf)


if __name__ == "__main__":
    unittest.main()

"""Integration: exercise the real LensCalcPy analytic-rate path and the real
Metropolis-Hastings event sampler end to end (no mocks).

Requires a working LensCalcPy (+ numba + scipy). Runs on this machine in the
project conda env.
"""
import unittest
import warnings

import numpy as np

import _harness as H

ev = H.load_events()


class LensCalcRatePath(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mw = ev.MilkyWayModel()

    def test_integrand_maker_returns_callable(self):
        ds = float(ev.kpc_from_mu0(14.5))
        f = ev.differential_rate_integrand_mw_maker(2.0, -3.0, ds, 5, 1.0, self.mw)
        self.assertTrue(callable(f))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = f(1.0, 0.5 * ds, 200.0)        # (umin, dl, t)
        self.assertTrue(np.isfinite(val))
        self.assertGreaterEqual(val, 0.0)

    def test_rate_positive_and_monotone_in_ds(self):
        # Farther sources subtend a longer line of sight through the lens
        # population, so the total rate should grow with ds.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rates = [ev.source_lensing_rate(2.0, -3.0, float(ev.kpc_from_mu0(mu0)),
                                            self.mw, u_t=5, mass=1.0)
                     for mu0 in (13.5, 14.5, 15.5)]
        for r in rates:
            self.assertTrue(np.isfinite(r) and r > 0.0)
        self.assertTrue(rates[0] < rates[1] < rates[2], rates)


class MetropolisHastingsEndToEnd(unittest.TestCase):
    def test_real_make_events_invariants(self):
        sources = H.build_sources()
        np.random.seed(H.SEED)
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = ev.make_events(sources, path, **H.MAKE_EVENTS_PARAMS)
        finally:
            os.unlink(path)

        p = H.MAKE_EVENTS_PARAMS
        self.assertEqual(df.shape[0], p["n_survey_events"])
        self.assertEqual(list(df.columns),
                         ["source_index", "gall", "galb", "mu0", "dl", "umin",
                          "crossing_time", "lograte"])
        # every retained sample sits in the physical proposal support
        self.assertTrue(np.all(np.isfinite(df["lograte"].to_numpy())))
        self.assertTrue(np.all(df["dl"].to_numpy() >= 0.0))
        self.assertTrue(np.all((df["umin"].to_numpy() >= 0.0)
                               & (df["umin"].to_numpy() <= p["u_t"])))
        ct = df["crossing_time"].to_numpy()
        self.assertTrue(np.all((ct >= p["t_min"] - 1e-6) & (ct <= p["t_max"] + 1e-6)))
        self.assertTrue(np.all(df["source_index"].to_numpy() < sources.shape[0]))


if __name__ == "__main__":
    unittest.main()

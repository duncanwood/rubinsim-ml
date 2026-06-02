"""Parity tests: the (cleaned) code must reproduce the captured goldens.

These are the contract that proves the cleanup changed no behavior. The
make_events check is an exact seeded reproduction; the rate/density checks use
assert_allclose with tolerances justified inline.
"""
import json
import os
import unittest
import warnings

import numpy as np
import pandas as pd

import _harness as H

ev = H.load_events()


def _run_make_events():
    sources = H.build_sources()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # outfile path is irrelevant to the returned frame; use a temp sink
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            return ev.make_events(sources, path, rng=H.seeded_rng(), **H.MAKE_EVENTS_PARAMS)
        finally:
            os.unlink(path)


class MakeEventsParity(unittest.TestCase):
    def setUp(self):
        self.golden = pd.read_csv(H.GOLDEN / "make_events_seed0.csv")

    @unittest.skipIf(os.environ.get("CI"),
                     "exact MH golden is platform-specific (a float-flip in an "
                     "accept/reject can change the event set); runs on the "
                     "reference machine, not in cross-platform CI")
    def test_exact_reproduction(self):
        df = _run_make_events().reset_index(drop=True)
        self.assertEqual(list(df.columns), list(self.golden.columns))
        self.assertEqual(df.shape, self.golden.shape)
        # integer label column: exact
        np.testing.assert_array_equal(df["source_index"].to_numpy().astype(int),
                                      self.golden["source_index"].to_numpy().astype(int))
        # float columns: exact in-env. rtol/atol 1e-12 only absorb the CSV text
        # round-trip at the ULP level (verified bit-identical across numpy
        # 1.24/1.26 and numba 0.57/0.59).
        for col in ["gall", "galb", "mu0", "dl", "umin", "crossing_time", "lograte"]:
            np.testing.assert_allclose(df[col].to_numpy(), self.golden[col].to_numpy(),
                                       rtol=1e-12, atol=1e-12, err_msg=f"column {col}")

    def test_determinism(self):
        a = _run_make_events().reset_index(drop=True)
        b = _run_make_events().reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


class RateParity(unittest.TestCase):
    def setUp(self):
        self.golden = json.loads((H.GOLDEN / "rates_small.json").read_text())
        self.mw = ev.MilkyWayModel()

    def test_source_lensing_rate(self):
        # rate_total integrates via scipy.quad; observed bit-identical across
        # scipy 1.11/1.13, so rtol=1e-9 is a conservative cushion.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for g in self.golden["rates"]:
                ds = float(ev.kpc_from_mu0(g["mu0"]))
                r = ev.source_lensing_rate(g["l"], g["b"], ds, self.mw,
                                           u_t=g["u_t"], mass=g["mass"])
                np.testing.assert_allclose(float(r), g["rate_per_hour"], rtol=1e-9,
                                           err_msg=f"rate at {g['l'],g['b'],g['mu0']}")

    def test_density_probes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for g in self.golden["densities"]:
                d = ev.sample_density_single_source(g["params"], self.mw, mass=1.0,
                                                    u_t=5, t_e=True)
                if g["density"] == 0.0:
                    self.assertEqual(float(d), 0.0)
                else:
                    np.testing.assert_allclose(float(d), g["density"], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()

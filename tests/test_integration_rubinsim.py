"""Integration: the Rubin/LSST MAF detection-metric path (rubinsim.py).

Two layers:
  * make_full_event_df is pure pandas and always runs.
  * The real rubin_sim MAF MicrolensingMetric run needs (a) an importable
    rubin_sim, (b) the rubin_sim reference data (throughputs), and (c) an opsim
    baseline .db. When any is absent the test skips with an explicit reason
    rather than passing silently. On this machine the reference data and opsim
    baseline are NOT installed (see AUDIT.md "Data dependencies"), so the real
    run skips; the test is written to execute once they are present.
"""
import os
import unittest
import warnings

import numpy as np
import pandas as pd

import _harness as H

ev = H.load_events()


def _maf_importable():
    try:
        import rubin_sim.maf  # noqa: F401
        return True
    except Exception:
        return False


def _rubin_data_ready():
    try:
        from rubin_sim.data import get_data_dir
        dd = get_data_dir()
    except Exception:
        return False
    base = os.path.join(dd, "throughputs", "baseline")
    return os.path.exists(os.path.join(base, "total_u.dat")) or \
        os.path.exists(os.path.join(base, "total_u.dat.gz"))


def _find_opsim():
    try:
        from rubin_sim.data import get_baseline
        b = get_baseline()
        if b and os.path.exists(b):
            return b
    except Exception:
        pass
    return None


HAVE_MAF = _maf_importable()


class MakeFullEventDf(unittest.TestCase):
    """The source-attribute join used before the slicer is built (pure pandas)."""
    def test_merge_attaches_source_columns(self):
        events = pd.read_csv(H.GOLDEN / "make_events_seed0.csv")
        sources = H.build_sources()
        full = ev.make_full_event_df(events, sources)
        for col in ["ra", "dec", "umag", "gmag", "rmag", "imag", "zmag", "ymag"]:
            self.assertIn(col, full.columns)
        # all event source_index values are valid -> inner join keeps every row
        self.assertEqual(full.shape[0], events.shape[0])
        # ra/dec come from the labelled source row
        for _, row in full.iterrows():
            si = int(row["source_index"])
            self.assertAlmostEqual(row["ra"], sources.loc[si, "ra"], places=9)
            self.assertAlmostEqual(row["dec"], sources.loc[si, "dec"], places=9)


@unittest.skipUnless(HAVE_MAF, "rubin_sim.maf not importable in this env (see AUDIT.md)")
class RealMafMetric(unittest.TestCase):
    def test_microlensing_metric_small_slice(self):
        if not _rubin_data_ready():
            self.skipTest("rubin_sim reference data absent: run `rs_download_data` "
                          "(see AUDIT.md 'Data dependencies')")
        baseline = _find_opsim()
        if baseline is None:
            self.skipTest("no opsim baseline .db found (see AUDIT.md 'Data dependencies')")

        import tempfile
        import lensemble  # full package import works once rubin_sim is available

        events = pd.read_csv(H.GOLDEN / "make_events_seed0.csv")
        sources = H.build_sources()
        full = ev.make_full_event_df(events, sources)
        outdir = tempfile.mkdtemp(prefix="lensemble_maf_")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lensemble.rubinsim.run_microlensing_metric(full, baseline, outdir)
        # run_all writes per-metric result artifacts under outdir
        produced = [f for _, _, fs in os.walk(outdir) for f in fs]
        self.assertTrue(len(produced) > 0, "MAF run produced no output files")


if __name__ == "__main__":
    unittest.main()

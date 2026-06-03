"""rubinml quickstart, runnable end to end with just the simulation half.

Reads a tiny bundled synthetic source catalog, Monte-Carlo samples microlensing
events from the analytic rate, and reports a few summaries + analytic rates.
The Rubin/LSST detection-metric half (rubinml.rubinsim) needs rubin_sim + survey
data and is only described at the end.

    python examples/quickstart.py

Needs: rubinml + LensCalcPy (see README "Installation"). No rubin_sim required.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import rubinml

HERE = Path(__file__).resolve().parent
SEED = 0


def main():
    sources = pd.read_csv(HERE / "sample_sources.csv")
    print(f"catalog: {sources.shape[0]} synthetic sources toward the bulge "
          f"({list(sources.columns)})\n")

    # --- Monte-Carlo sample events (pass rng for reproducibility) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = rubinml.make_events(
            sources, str(HERE / "events.pkl"),
            n_survey_events=500, ntoss=500, pbhmass=1.0,
            rng=np.random.default_rng(SEED),
        )

    ct = events["crossing_time"].to_numpy()
    print(f"sampled {events.shape[0]} events")
    print(f"  crossing time (h): {np.percentile(ct,5):.0f} .. "
          f"{np.median(ct):.0f} (median) .. {np.percentile(ct,95):.0f}")
    print(f"  impact parameter umin: 0 .. {events['umin'].max():.2f}")
    print(f"  lens distance dl (kpc): {events['dl'].min():.1f} .. {events['dl'].max():.1f}\n")

    # --- analytic event rate for a few sources (1/hour) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mw = rubinml.events.MilkyWayModel()
        rates = {}
        for i in (0, 100, 199):
            l, b, mu0 = sources.loc[i, ["gall", "galb", "mu0"]]
            ds = float(rubinml.kpc_from_mu0(mu0))
            rates[i] = rubinml.source_lensing_rate(l, b, ds, mw, mass=1.0)
    print("analytic per-source rates (1/hour):")
    for i, r in rates.items():
        print(f"  source {i:3d} (l={sources.loc[i,'gall']:+.1f}, ds={rubinml.kpc_from_mu0(sources.loc[i,'mu0']):.1f} kpc): {r:.3e}")
    print(f"\nscaled 10-yr Rubin count (toy, {len(rates)} sources): "
          f"{rubinml.rates_to_rubin_counts(rates):,}")

    # --- optional: a diagnostic plot if matplotlib/seaborn are installed ---
    if rubinml.plots is not None:
        rubinml.plots.crossing_time_hist(events, str(HERE), "quickstart")
        print(f"\nwrote {HERE / 'quickstart_crossing_time_hist.pdf'}")
    else:
        print("\n(install matplotlib/seaborn for plots)")

    print("\nNext: the detection half --")
    print("  full = rubinml.make_full_event_df(events, sources)")
    print("  rubinml.rubinsim.run_microlensing_metric(full, 'baseline.db', 'out/')")
    print("  ... needs rubin_sim + rs_download_data + an opsim baseline .db.")


if __name__ == "__main__":
    main()

"""Regenerate the golden fixtures under tests/golden/ (and the source CSV).

Run from the project conda env that has LensCalcPy + numba working:

    /Users/duncan/mambaforge/envs/rubinsim/bin/python tests/_capture_goldens.py

This captures CURRENT (pre/post-cleanup-identical) behavior. The cleanup edits
are dead-code/comment/format only, so the parity tests must keep matching these
goldens before and after. Seeding lives here, not in the library.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H  # noqa: E402


def main():
    H.GOLDEN.mkdir(parents=True, exist_ok=True)
    H.FIXTURES.mkdir(parents=True, exist_ok=True)
    ev = H.load_events()

    sources = H.build_sources()
    sources.to_csv(H.FIXTURES / "sources_small.csv", index=False)

    # --- make_events seeded golden -------------------------------------------
    out_pkl = str(H.GOLDEN / "_scratch_make_events.pkl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ev.make_events(sources, out_pkl, rng=H.seeded_rng(), **H.MAKE_EVENTS_PARAMS)
    Path(out_pkl).unlink(missing_ok=True)
    df.to_csv(H.GOLDEN / "make_events_seed0.csv", index=False)
    (H.GOLDEN / "make_events_params.json").write_text(json.dumps(
        {"seed": H.SEED, **H.MAKE_EVENTS_PARAMS, "n_rows": int(df.shape[0]),
         "columns": list(df.columns)}, indent=2))

    # --- analytic rate goldens (real LensCalcPy) -----------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mw = ev.MilkyWayModel()
        rates = []
        for (l, b, mu0) in H.RATE_PROBES:
            ds = float(ev.kpc_from_mu0(mu0))
            r = ev.source_lensing_rate(l, b, ds, mw, u_t=5, mass=1.0)
            rates.append({"l": l, "b": b, "mu0": mu0, "ds_kpc": ds,
                          "u_t": 5, "mass": 1.0, "rate_per_hour": float(r)})
        # Differential-rate density probes (sample_density_single_source), called
        # the way make_events calls it (t_e=True). params is
        # [source_index, l, b, mu0, dl, umin, crossing_time]. We lock one in-support
        # (nonzero) point taken from a golden event, and one out-of-bounds point
        # (dl > ds) that must hit the guard and return exactly 0.
        nonzero = [2, -0.5714285714285716, -3.5, 14.4,
                   6.676293302953483, 1.6415165414460549, 662.0508232833081]
        guard = [0, 2.0, -3.0, 14.5, 999.0, 1.0, 200.0]  # dl >> ds -> 0
        dens = []
        for tag, params in (("in_support", nonzero), ("guard_dl_gt_ds", guard)):
            d = ev.sample_density_single_source(params, mw, mass=1.0, u_t=5, t_e=True)
            dlog = ev.sample_density_single_source_log(params, mw, mass=1.0, u_t=5, t_e=True)
            dens.append({"tag": tag, "params": params,
                         "density": float(d), "logdensity": float(dlog)})
    (H.GOLDEN / "rates_small.json").write_text(json.dumps(
        {"rates": rates, "densities": dens}, indent=2))

    print("wrote goldens:")
    for p in sorted(H.GOLDEN.glob("*")):
        if not p.name.startswith("_scratch"):
            print("  ", p.relative_to(H.REPO))
    print("rows:", df.shape, "| rate[0]:", rates[0]["rate_per_hour"])


if __name__ == "__main__":
    main()

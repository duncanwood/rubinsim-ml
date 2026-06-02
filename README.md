# rubinml

Monte-Carlo simulation of gravitational **microlensing events** and their
detectability by the **Rubin Observatory / LSST**.

`rubinml` samples an analytic differential event-rate for an NFW dark-matter
(e.g. primordial black hole) lens population folded through a Milky Way model,
then pushes the sampled events through Rubin's `MicrolensingMetric` (MAF) to
estimate detection efficiency across observing-cadence strategies.

It is the laptop-minutes replacement for a cluster-weeks PopSyCLE
population-synthesis run: sampling is linear in the number of proposed events
(Metropolis-Hastings over the analytic rate, numba-JIT) rather than quadratic in
the source catalog. From the dissertation; now a reference / portfolio artifact.

The code has been audited and tested -- see [`MAP.md`](MAP.md) (architecture)
and [`AUDIT.md`](AUDIT.md) (findings + fixes).

## The pipeline

```
source catalog (gall, galb, mu0, ra, dec, *mag)
      |   make_events            -> Metropolis-Hastings sample of the analytic NFW rate
      v
event_df (source_index, dl, umin, crossing_time, lograte, ...)
      |   make_full_event_df     -> attach apparent magnitudes + sky position
      v
run_microlensing_metric[_mult]   -> Rubin MAF MicrolensingMetric on an opsim baseline
      |
      v
detection fractions  ->  plots (efficiency vs crossing time, cadence comparison, ...)
```

The two halves are decoupled: the **simulation + rate** half needs only
LensCalcPy; the **detection-metric** half needs `rubin_sim` and survey data.
`import rubinml` works with just the simulation half installed (`rubinml.rubinsim`
is `None` when `rubin_sim` is absent).

## Installation

Python 3.11. The reference environment is conda/mamba.

### 1. Core environment

```bash
mamba env create -f environment.yml
conda activate rubinml
```

`numpy` is pinned `<2` for `rubin_sim` / `LensCalcPy` compatibility.

**macOS note (libgfortran rpath).** Some conda-forge builds of the gfortran /
OpenBLAS stack ship a malformed duplicate `@loader_path` rpath that recent macOS
dyld rejects, breaking `import numpy` with
`Library not loaded: @rpath/libgfortran.5.dylib ... duplicate LC_RPATH`. If you
hit this, repair it in place (version-preserving, reversible, backs up each lib):

```bash
python scripts/fix_macos_conda_rpaths.py        # operates on $CONDA_PREFIX/lib
```

(The conda-native alternative -- upgrading libgfortran -- pulls numpy 2 and
breaks the pinned stack, so the in-place rpath fix is preferred here.)

### 2. LensCalcPy (analytic event rate)

This code depends on a *fork* of LensCalcPy, not the public package: the rate
model uses functions added on top of
[NolanSmyth/LensCalcPy](https://github.com/NolanSmyth/LensCalcPy) -- the `ds`
argument to `einstein_rad`, the Jacobian in `differential_rate_integrand`,
`differential_rate`, single-source event sampling -- and a `MilkyWayModel()` that
takes no required arguments. The public NolanSmyth `HEAD` has a different
`MilkyWayModel` API and the PyPI `0.0.3` build has an `einstein_rad()` bug.

The fork is published at
[duncanwood/LensCalcPy](https://github.com/duncanwood/LensCalcPy) (branch
`functional-refactor`); `requirements.txt` pins the exact commit. Install it
directly, or use a local editable checkout for development:

```bash
pip install "git+https://github.com/duncanwood/LensCalcPy.git@functional-refactor"
# or:  pip install -e /path/to/LensCalcPy
```

### 3. rubin_sim (detection metric -- optional)

Only needed for the MAF metric half.

```bash
pip install rubin_sim          # or editable from a checkout
rs_download_data               # reference throughputs/skybrightness (multi-GB)
```

You also need an **opsim baseline `.db`** (a cadence simulation) to run the
metric against -- download one from the Rubin survey-strategy releases and pass
its path to `run_microlensing_metric`.

### 4. The package

```bash
pip install -e rubinml
```

## Quickstart

Simulation half (needs only LensCalcPy). Pass an explicit `rng` for reproducible
output:

```python
import numpy as np, pandas as pd
import rubinml

# A real run uses a TRIStar/TRILEGAL catalog; here a tiny synthetic stand-in.
sources = pd.DataFrame({
    "gall": np.linspace(-4, 4, 8), "galb": np.linspace(-5, -1.5, 8),
    "mu0":  np.linspace(13.8, 15.2, 8),
    "ra":   np.linspace(264, 269, 8), "dec": np.linspace(-30, -26, 8),
    **{b: np.linspace(22, 20, 8) for b in
       ("umag", "gmag", "rmag", "imag", "zmag", "ymag")},
})

events = rubinml.make_events(
    sources, "events.pkl", n_survey_events=1000, pbhmass=1.0, ntoss=2000,
    rng=np.random.default_rng(0),
)

# analytic rate for one source (1/hour):
mw = rubinml.events.MilkyWayModel()
rate = rubinml.source_lensing_rate(2.0, -3.0, rubinml.kpc_from_mu0(14.5), mw)
```

Detection half (needs `rubin_sim` + reference data + an opsim `.db`):

```python
full = rubinml.make_full_event_df(events, sources)
rubinml.rubinsim.run_microlensing_metric(full, "baseline_v3.4_10yrs.db", "out/")
```

## Tests

```bash
python -m unittest discover -s tests        # 28 tests; pytest also works
```

Unit + exact-seeded parity (`tests/golden/`) + integration. Tests that need the
LensCalcPy fork (the analytic-rate and MC paths) self-skip when it is absent, and
the rubin_sim MAF integration self-skips without the reference data / opsim
baseline. On a fully-configured machine all run; in CI the public-reproducible
subset (pure logic + packaging) runs green and the rest skip.

## Reproducing the original (dissertation) results

The dissertation numbers were produced before the correctness fixes in
`AUDIT.md` (notably the MH source-index bug, #1). That state is tagged:

```bash
git checkout dissertation-original
```

Its test golden captures the original (pre-fix) sampler output, so the suite
reproduces the old numbers. `main`/`master` has the corrected behavior.

## Repository layout

```
rubinml/rubinml/       package: events.py (MC core), rubinsim.py (MAF), plots.py
backup scripts/        earlier standalone versions (historical; not imported)
notebooks/             analysis-driver notebooks + saved results
tests/                 unit + parity + golden + integration
scripts/               fix_macos_conda_rpaths.py (env repair helper)
MAP.md / AUDIT.md      architecture map / audit findings
environment.yml        conda env for the simulation core
requirements.txt       exact pins of the reference machine
```

## Adapting to other surveys or datasets

The simulation core (`events.py`) is largely survey-agnostic -- it depends on
the Galactic model (LensCalcPy) and a source catalog, with the survey entering
only through a few parameters: `u_t` (threshold impact parameter), and the
crossing-time bounds `t_min`/`t_max` (daily cadence to survey length). To use a
different source catalog, provide a DataFrame with columns `gall`, `galb`,
`mu0`, `ra`, `dec`, and per-band `*mag`.

The detection half (`rubinsim.py`) is **Rubin/LSST-specific**: it is built on
`rubin_sim` MAF, the `MicrolensingMetric`, opsim baselines, and the `ugrizy`
bands / single-visit depths (`plots.single_exp_m5`). Supporting another
telescope means, roughly:

1. parameterize the survey config (bands, single-visit depths, cadence, `u_t`)
   instead of the hardcoded LSST values;
2. document / adapt the source-catalog schema for that survey's photometry;
3. provide a detection backend for the new survey -- either that survey's own
   cadence-metric tooling, or a survey-agnostic light-curve injection + outlier
   test (the sibling `nsc-ml` repo is one such detector).

These are design changes, not yet implemented; the simulation half can be reused
today by supplying an appropriate catalog and parameters. A prior-art survey and
a concrete proposal for a shared, telescope-agnostic survey-config layer (also
intended for the sibling `nsc-ml` detector) is in
[`docs/survey-config.md`](docs/survey-config.md).

## References

Duncan Wood, PhD dissertation (Rubin/LSST microlensing chapter) -- see the rate
derivation and the detection-efficiency results.

## License

TODO -- add a license before distributing.

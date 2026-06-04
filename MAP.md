# lensemble: code map

Microlensing event simulation from my [dissertation](https://escholarship.org/uc/item/9g81m0j9).
It Monte-Carlo samples an analytic differential microlensing event rate for an NFW dark-matter
(e.g. primordial black hole) lens population folded through a Milky Way model,
then pushes the sampled events through the Rubin/LSST `MicrolensingMetric` (MAF)
to estimate detectability across observing-cadence strategies.

It is the laptop-minutes replacement for the cluster-weeks PopSyCLE
population-synthesis run: the sampler is linear in the number of proposed events
rather than quadratic in the source catalog, and the rate evaluation is the
analytic NFW integrand (LensCalcPy) instead of a resolved population.

This document maps the code. Findings and the environment/data caveats are in
`AUDIT.md`.

## Layout

```
lensemble/
  pyproject.toml       build config (setuptools; requires-python >=3.11)
  lensemble/
    __init__.py        from .events import *  /  .plots  /  .rubinsim   (star imports)
    events.py          MC core: analytic rate + Metropolis-Hastings sampler
    rubinsim.py        rubin_sim MAF orchestration (the detection metric)
    plots.py           event + detection diagnostic plots
  tests/               unit + parity + golden + integration
  requirements.txt     pinned deps
```

## Data flow

```
TRIStar source catalog (gall, galb, mu0, ra, dec, *mag)   [external parquet]
        |
        v
make_events(sources, ...)            events.py  -- MH sampler over event space
        |   event_df: source_index, gall, galb, mu0, dl, umin, crossing_time, lograte
        v
make_full_event_df(event_df, sources)   join apparent mags + ra/dec by source_index
        |   full_events_df
        v
run_microlensing_metric[_mult](full_events_df, baseline.db, outdir)   rubinsim.py
        |   UserPointsSlicer(ra, dec) + slice_points(crossing_time/24, umin, peak_time, apparent_m_*)
        |   maf.MicrolensingMetric() over a MetricBundleGroup on an opsim baseline
        v
detection arrays (.npz / .pickle)  ->  make_ndet_df / plots.py (efficiency, sky maps, cadence compare)
```

## events.py: MC core

Angle / unit helpers (numba `@njit`):
- `wrap_degrees(x)` -> wrap to [-180, 180).
- `kpc_from_mu0(mu0)` -> 10**(mu0/5 - 2) kpc (distance modulus -> distance).
- `mu0_from_kpc(kpc)` -> inverse.

Analytic rate (thin wrappers over `LensCalcPy.pbh`):
- `differential_rate_integrand_mw_maker(l, b, ds, u_t, mass, mw_model, ...)`
  -> closure `f(umin, dl, t)` capturing the sightline; this is the integrand.
- `source_lensing_rate(l, b, ds, mw_model, u_t=5, mass=1, tcad=24, tobs=24*365*10, ...)`
  -> `LensCalcPy.pbh.rate_total(...)`; total rate (1/hour) for one source.
- `sample_density_single_source(params, mw_model, lbounds, bbounds, mass, u_t, **lcp)`
  -> differential-rate density at one point of event space, 0 outside the
  physical support. `params = [source_index, l, b, mu0, dl, umin, crossing_time]`.
- `sample_density_single_source_log(...)` -> log of the above, -inf on 0.

Sampler:
- `make_events(sources, outfile, n_survey_events, u_t=5, t_min=24, t_max=24*365*10,
  pbhmass=1, ntoss=20000, write_progress=False)` -> `event_df`.
  Metropolis-Hastings over event space. Each proposal redraws a source index and
  draws `dl ~ U(0, ds)`, `umin ~ U(0, u_t)`, and `crossing_time` log-uniform in
  `[t_min, t_max]` (daily cadence to survey length). The proposal log-rate adds
  `log(ds)` (line-of-sight volume) and `log(crossing_time)` (Jacobian of the
  log-uniform t draw); acceptance is the standard log-space rule. First `ntoss`
  samples are burn-in. Uses the **legacy unseeded** `np.random` throughout
  (see AUDIT: reproducibility).
- `make_full_event_df(event_df, source_df)` -> merge apparent mags + ra/dec onto
  events by `source_index`.

Rates I/O + scaling:
- `calculate_lensing_rates(sources, mw, pbhmass, outdir, n_sources, u_t, write_csv)`
  -> per-source `rate_total` over a random subset; writes csv/pickle.
- `load_rates_from_file`, `get_partial_rates`, `rubin_counts_from_rates_file`.
- `rates_to_rubin_counts(rates, n_tristar=11_433_322_690)`
  -> mean rate * `n_tristar` sources * `24*365*10` survey hours -> expected
  10-yr Rubin event count.
- `log_rates_to_rubin_counts(...)`, **dead and broken** (`np.rates`), uncalled
  (AUDIT).

## rubinsim.py: MAF detection metric

- `run_microlensing_metric(events, baseline_file, outdir, t_start=1, t_end=3652)`
  single-strategy run: builds a `UserPointsSlicer(ra, dec)`, sets slice_points
  (`crossing_time/24` hours->days, `impact_parameter`, `peak_time ~ U(t_start,
  t_end)`, `apparent_m[_no_blend]_<band>`), runs `maf.MicrolensingMetric()` in a
  `MetricBundleGroup`.
- `run_microlensing_metric_mult(events_list, sources, baseline_file, outdir,
  constraint=None, metric_options={}, ...)` multi-mass loop. Pickles
  `(events_info, bundle.metric_values)` per mass.
- `run_microlensing_metric_mult_Fisher_Npts_Nights(...)` runs three
  `metric_calc` modes (`Fisher`, `Npts`, `Nnights`) per event set.
- `make_metric_plots(bundles, outDir)` HealpixSkyMap renders (note an
  indentation bug, AUDIT).
- `make_ndet_df(result_files)` -> DataFrame of detection fraction per (opsim, mass).

## plots.py: diagnostics

Event distributions (`dl_hist`, `umin_hist`, `crossing_time_hist`,
`crossing_time_umin_scatter`, `mag_in_te_bins`), detection efficiency in
crossing-time bins (`efficiency_in_te_bins`, `bin_efficiency`,
`detection_scatter_mag`), and a cadence-vs-PBH-mass comparison
(`compare_opsims`, which imports `rubinsim.make_ndet_df`). `single_exp_m5` /
`color_filter` are per-band constants. Many functions carry auto-generated
`_summary_`/`_description_` docstring stubs (AUDIT: cosmetic).

## The MicrolensingMetric

`rubinsim.py` imports the installed `rubin_sim.maf.MicrolensingMetric`. On this
machine that name resolves to the `_new` variant
(`.../rubin_sim/maf/maf_contrib/microlensing_metric_new.py`): it adds a
`Nnights` mode, combines `m52snr` with a Poisson SNR in quadrature, and loads
instrument zeropoints at construction, which is why it needs the `rubin_sim`
throughput data to even instantiate. The science does not depend on a custom
metric.

## How it runs on this machine

Conda env `rubinsim` (Python 3.11.9): numpy 1.26.4, pandas 2.2.2, numba 0.59.1,
scipy 1.13.1, rubin_sim 2.0.1.dev, LensCalcPy 0.0.3 (editable from
`../LensCalcPy`), matplotlib 3.8.4, seaborn 0.13.2. `import lensemble` succeeds and
`make_events` + the analytic-rate path run. Tests:

```
/Users/duncan/mambaforge/envs/rubinsim/bin/python -m unittest discover -s tests
```

The Rubin MAF metric path needs the `rubin_sim` reference data and an opsim
baseline `.db`, neither of which is installed here (see `AUDIT.md`, Environment
& Data). Some conda-forge builds ship a broken numpy (a dyld regression).
`scripts/fix_macos_conda_rpaths.py` repairs it in place (see the README).

# rubinml -- audit

Proposals and findings, not applied changes. The only edits made on the
`cleanup/rubinml-audit` branch were dead-code/comment removal, sparse comments,
and one whitespace fix (see `git log`); the parity tests prove behavior is
unchanged. Everything below is a recommendation to be applied (or not) later.

## Verdict

The MC core is sound and the method is what the dissertation describes
(Metropolis-Hastings sampling of the analytic NFW differential event rate, then
the Rubin MAF metric). The code is research-grade: terse, notebook-driven, with
the usual stale comments and a few latent inconsistencies. **One probable
correctness bug** (the source-index/coordinate mismatch in the MH loop, #1) is
worth a careful look because it can bias the apparent magnitudes attached to
each event. The rest is hygiene. Nothing here was changed because all of it can
alter outputs or structure.

Severity tags: [BUG] likely wrong result, [REPRO] reproducibility,
[SMELL] latent footgun, [STRUCT] structure/imports, [ENV] environment/data,
[COSMETIC].

---

## Status -- fixes applied 2026-06-02 (branch `fix/rubinml-audit-findings`)

Applied (suite parity-green; goldens recaptured where behavior changed):
- #1 MH source-index bug -- fixed (`source_arr[new_source_index]`); regression
  test asserts each event's `source_index` matches its own `(gall,galb,mu0)`.
- #2 dead/broken `log_rates_to_rubin_counts` -- removed.
- #3 seed-point distance formula -- now `kpc_from_mu0`.
- #4 `u_t` default mismatch -- aligned to 5.
- #5 `make_metric_plots` indentation -- fixed (by inspection; not runtime-tested,
  needs MAF data).
- #6 unseeded RNG -- `make_events`/`calculate_lensing_rates` take `rng=None`
  (default `np.random.default_rng()`); `np.random.*` -> `rng.*`.
- #7 `tqdm.notebook` -> `tqdm.auto` (runs in a plain interpreter).
- #8 magic numbers -- `N_TRISTAR`, `SURVEY_HOURS`, `DAILY_CADENCE_HOURS` named.
- #11 unused imports + unused `resultDbs` -- removed.
- #9 path handling -- `make_events` uses `Path(outfile).parent`; `compare_opsims`
  takes a `baseline_key` parameter.
- #10 star imports removed -- `__init__` exports explicit names; `plots` and
  `rubinsim` import optionally, so the MC core imports without matplotlib/seaborn
  or rubin_sim (verified both ways).
- #16 distutils `setup.py` replaced by `pyproject.toml`.

Deferred (still proposals):
- remaining cosmetics (plots.py `_summary_` docstring stubs, mixed indentation).
- #14/#15 are machine state (env repaired; data still absent), not code.

The findings below are retained as the rationale of record.

---

## Correctness

### 1. [BUG] MH proposal reads the previous source's coordinates, not the proposed one
`events.py`, `make_events` inner loop:

```python
new_source_index = np.random.randint(sources.shape[0])   # propose a NEW index
new_l, new_b, new_mu0 = source_arr[source_index]          # but read the OLD index
...
new_event = [new_source_index, new_l, new_b, new_mu0, new_dl, new_umin, new_crossing_time]
```

`source_arr` is indexed by `source_index` (the current/previous state) while the
event is *labelled* with the freshly drawn `new_source_index`. So every stored
sample's `source_index` and its `(gall, galb, mu0)` are off by one MH step. The
chain still walks over sources (next iteration reads `source_arr[new_source_index]`),
but the label lags the coordinates by one accepted move.

Demonstrated in the golden (`tests/golden/make_events_seed0.csv`): row 0 has
`source_index=2` but carries the coordinates of source index 3
(`gall=-0.5714..., mu0=14.4`).

Why it matters: `make_full_event_df` joins apparent magnitudes and `ra/dec` onto
events **by `source_index`**. With the mismatch, an event's lensing geometry
(`l, b, ds`, hence its rate and crossing-time draw) is paired with a *different*
star's brightness and sky position. For a randomly ordered catalog
(`rand_tristar_*`) that pairs each event with effectively random photometry,
which feeds straight into the detection metric. This could bias detection
efficiency; the size of the effect depends on catalog ordering and the
distance-magnitude correlation.

Proposed fix (NOT applied -- changes outputs): `source_arr[new_source_index]`.
If current results are to be trusted as-is, confirm first whether downstream
analysis ever relied on `source_index` for the join, or only used the event
geometry. Re-run the goldens after fixing.

### 2. [BUG] `log_rates_to_rubin_counts` is dead and broken
`events.py`: `return round(sum(np.rates.values()) * n_tristar / ...)` -- `np.rates`
is an `AttributeError` (numpy has no `rates`). The function is never called
anywhere in the package or notebooks. Proposal: delete it, or fix to
`sum(rates.values())` if a log-space variant was actually wanted. Left in place
because removing a public (star-exported) symbol is an API change.

### 3. [SMELL] Initial-point distance uses a different formula than `kpc_from_mu0`
`events.py`, `make_events` seed point: `ds = np.power(10, mu0/5. - 3)`, i.e.
`10**(mu0/5 - 3)`, which is **10x smaller** than `kpc_from_mu0(mu0) = 10**(mu0/5 - 2)`
used everywhere else. This is the old (pre-correction) distance-modulus formula
(it also appears in the dead comment removed during cleanup). It only affects the
single seed sample, which is inside the `ntoss` burn-in and discarded, so results
are unaffected -- but it is a latent inconsistency. Proposal: use `kpc_from_mu0`.
Note the seed-point log-rate also omits the `+log(crossing_time)` Jacobian that
the loop applies (again burn-in only).

### 4. [SMELL] Inconsistent `u_t` defaults
`sample_density_single_source(..., u_t=5)` vs `sample_density_single_source_log(..., u_t=2)`.
The main path passes `u_t` explicitly so the mismatch is never exercised, but the
defaults disagreeing is a footgun. Proposal: make both default to the same value
(5 is what the sampler and rate use).

### 5. [SMELL] `make_metric_plots` indentation drops a colour scale
`rubinsim.py`: in the `for c in max_colors:` loop only `plotDict` is set inside
the loop; the `PlotHandler` construction and the plotting `for k in bundles`
block are dedented to function scope, so only the last `c` (300) is ever used and
the `c=30` map is never produced. Plotting only (no science impact). Proposal:
indent the plotting block into the loop, or drop the loop.

---

## Reproducibility

### 6. [REPRO] Unseeded legacy RNG in the library
`make_events` (and `calculate_lensing_rates`) use module-level `np.random.*`
(legacy Mersenne Twister) with no seed and no injected generator, so runs are not
reproducible without a process-global `np.random.seed`. The tests seed in the
harness; the library should not be seeded, but it should accept an injected RNG.
Proposal: add `rng: np.random.Generator = None` (default
`np.random.default_rng()`) and replace `np.random.randint/random` with
`rng.integers/rng.random`. This is the single highest-value modernization for a
reference artifact. (Note: it changes the number stream, so it invalidates the
current goldens -- recapture after.)

### 7. [REPRO/STRUCT] `from tqdm.notebook import tqdm` breaks non-notebook use
Both `events.py` and `rubinsim.py` import `tqdm.notebook`, which raises
`ImportError: IProgress not found` at call time outside a Jupyter session with
ipywidgets. `make_events` therefore cannot run from a plain interpreter/script as
written. The test harness installs a passthrough shim. Proposal:
`from tqdm.auto import tqdm` (transparently falls back to the text bar).

### Reproducibility notes
- The legacy MT stream is deterministic for a fixed `np.random.seed` and a fixed
  call sequence, independent of numpy version. The seeded `make_events` golden
  was verified **bit-identical** across env `science` (numpy 1.24.3 / numba 0.57)
  and env `rubinsim` (numpy 1.26.4 / numba 0.59), and `source_lensing_rate` was
  bit-identical across scipy 1.11 and 1.13. Goldens were captured in `rubinsim`.
- numba `@njit` helpers are pure and deterministic.

---

## Constants, magic numbers, paths

### 8. [SMELL] Name the magic numbers
- `n_tristar = 11_433_322_690` (`rates_to_rubin_counts`): the TRIStar source
  count used to scale a mean per-source rate to the full population. Belongs in a
  named module constant or config, not a default argument.
- `24*365*10` (= 87600 survey hours) appears in `rates_to_rubin_counts` and as
  the `tobs` default in `source_lensing_rate`. Define `SURVEY_HOURS` once.
- `u_t = 5` (threshold impact parameter, Einstein radii), `t_min = 24` (h, daily
  cadence), `t_max = 24*365*10` (h, survey length), `ntoss = 20000` (burn-in):
  fine as defaults but worth documenting as a small parameter block.
- The "0.01 radius" is `angular_radius=.01` (deg) in the Q3C `get_nearby_sources`
  query in `backup scripts/microlensing_event_sampling.py`; it is not in the
  active code path.

### 9. [SMELL] Output-path handling
`make_events` derives `basedir` by string-splitting `outfile` on `/`
(`'/'.join(outfile.split("/")[:-1])`); `calculate_lensing_rates` builds
timestamped filenames with `time.time()`. Proposal: use `pathlib` consistently
(`Path(outfile).parent`) and pass output directories explicitly. No hardcoded
absolute paths remain in the active code (the old `$HOME/rubin-user/...` default
was a dead comment, removed during cleanup). `compare_opsims` hardcodes the
baseline key `'baseline_v3.6_10yrs'` for normalization -- make it a parameter.

---

## Structure / imports

### 10. [STRUCT] Star imports and the rubin_sim coupling
`__init__.py` does `from .events import *`, `from .plots import *`,
`from .rubinsim import *`. Because `rubinsim` (via `plots` too) imports
`rubin_sim`, **`import rubinml` cannot succeed without rubin_sim installed**, even
to use only the MC core. Proposal: drop the star imports for explicit names, and
either lazy-import `rubin_sim` inside the metric functions or split the metric
module so `events` is usable standalone. (The tests load `events.py` directly to
sidestep this.)

### 11. [STRUCT] Unused imports
- `events.py`: `glob` (unused).
- `rubinsim.py`: `datetime`, `pathlib.Path`, `csv`, `numba.njit`,
  `tqdm`, `rubin_sim.utils as rsUtils`, `rubin_sim.data.get_baseline`,
  `rubin_sim.maf.db as db` -- all imported, none used.
Removing unused imports is low-risk but is left as a proposal to keep the audit
diff strictly dead-comment/format only.

### 12. [SMELL] `make_full_event_df` working-tree change
The current `make_full_event_df` selects only `ra, dec, *mag` and merges
`left_on='source_index'` (an earlier version also carried `gall, galb, mu0` and
merged on the index). This is the join that interacts with bug #1: it overwrites
nothing but attaches the labelled source's photometry. Worth re-checking together
with #1.

### 13. [COSMETIC] one-iteration loop
`make_events` initializes the seed point inside `for i in range(1):`. Harmless;
could be a straight assignment.

---

## Environment & data (base state -- record and continue)

### 14. [ENV] Broken numpy in the `rubinsim` conda env (repaired during audit)
On first run, `import numpy` in env `rubinsim` failed:
`Library not loaded: @rpath/libgfortran.5.dylib ... duplicate LC_RPATH '@loader_path'`.
This is a conda packaging / new-macOS dyld regression: several gfortran-stack
dylibs were built with a duplicate `@loader_path` rpath (one with a trailing
slash) that current dyld rejects, which cascades through libopenblas ->
libgfortran -> libquadmath and breaks numpy and everything downstream.

To get the package running on this machine, the audit deduped the rpaths of the
affected dylibs (each backed up alongside as `*.bak-rpath`):
`lib/libopenblas.0.dylib`, `lib/libgfortran.5.dylib`, `lib/libquadmath.0.dylib`,
plus `libhdf5_fortran`, `libhdf5hl_fortran`, `liboorb`, and three `libprotobuf*`
that had the same defect. This is a reversible environment repair, not a repo
change.

Revert (if desired): for each, `cp <lib>.bak-rpath <lib>` in
`/Users/duncan/mambaforge/envs/rubinsim/lib/`. A cleaner permanent fix is
`conda install -c conda-forge --force-reinstall libopenblas libgfortran5
libquadmath` (or rebuild the env from `requirements.txt`).

Also note: a stale **PyPI** build of `LensCalcPy 0.0.3` (in env `science`) has an
`einstein_rad(dl, mass)` call missing the `ds` argument (`pbh.py`), which breaks
the rate path under numba. The editable repo checkout at `../LensCalcPy` has the
correct `einstein_rad(dl, mass, ds)`; prefer it. The test harness prepends it.

### 15. [ENV] Missing reference data and opsim baseline
- `RUBIN_SIM_DATA_DIR` is unset and `~/rubin_sim_data` is empty; the installed
  `MicrolensingMetric` (the `_new` variant) calls `load_inst_zeropoints()` at
  construction and fails with `total_u.dat` not found. Fix: `rs_download_data`
  (the rubin_sim reference throughputs/skybrightness set, multi-GB).
- No opsim baseline `.db` is present (`get_baseline()` raises). The MAF
  integration test (`tests/test_integration_rubinsim.py`) therefore **skips with
  reason**; it is written to run the real metric once the data + a baseline `.db`
  are available. The only local opsim-related DB is a MAF *results* DB at
  `physics/microlensing/opsim/resultsDb_sqlite.db`, not a cadence baseline.
- The source catalog `rand_tristar_10_000_000.parquet` (loaded by the notebooks)
  is not on this machine; tests use a small synthetic stand-in
  (`tests/fixtures/sources_small.csv`). Reading the real catalog needs `pyarrow`.

### 17. [ENV/REPRO] The analytic-rate path depends on a private LensCalcPy fork
`events.py`'s rate model (`source_lensing_rate`, `sample_density_single_source`,
the `make_events` MC) calls LensCalcPy functions that exist only on the project's
local fork -- the `ds` arg to `einstein_rad`, the Jacobian in
`differential_rate_integrand`, `differential_rate`, single-source sampling -- and
a `MilkyWayModel()` with no required args. The local LensCalcPy checkout is at a
commit (`fac1a16`) **not present on `NolanSmyth/LensCalcPy`**, plus uncommitted
`pbh.py` changes. The public package has a different `MilkyWayModel` API (and an
`einstein_rad` bug in 0.0.3), so the rate path does not run against it. **This is
the deepest reproducibility blocker** -- no CI or external user can run the
simulation without this fork.

Fix (needs the fork owner): commit the LensCalcPy working-tree changes, push the
fork to a public remote (e.g. `github.com/duncanwood/LensCalcPy`), then pin that
commit in `requirements.txt` + `.github/workflows/tests.yml` and drop the
`lenscalc_rate_available()` test skips. Until then CI runs only the public-
reproducible subset (pure logic + packaging); the rate/MC tests + quickstart
self-skip (see the CI run, `OK (skipped=14)`).

### 16. [ENV] Packaging
`rubinml/setup.py` uses `distutils` (removed in Python 3.12) with `version='0.0'`
and `py_modules=['rubinml', ...]` (the code is a package, not top-level modules).
Proposal: a minimal `pyproject.toml` with `install_requires` from
`requirements.txt`.

---

## Backups

Keep `backup scripts/` as historical provenance -- do not delete. They document
where the installed metric came from and the pre-MCMC sampling approach. They are
not imported by the package (verified) and the science does not depend on them
(see MAP.md STEP 1b). If repo tidiness matters, move them under `docs/legacy/`
rather than removing them.

---

## Cosmetic

- Auto-generated `_summary_` / `_description_` docstring stubs remain in
  `plots.py` and `make_events`; replace with one-line descriptions or drop.
- `plots.py` mixes 2-space and 4-space indentation across functions (not changed
  -- reindenting would obscure any future diff).
- `resultDbs = {}` in `rubinsim.py` is assigned and never used.

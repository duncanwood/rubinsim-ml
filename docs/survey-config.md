# Survey configuration -- design note

Status: **proposal** (2026-06-02). Prior art researched; no code written yet,
pending the "where does it live" decision at the end. Targets both this repo
(rubin-sim-ml: event sim + Rubin/LSST metric) and the sibling **nsc-ml**
(multi-band time-series detection), which today both hardcode LSST `ugrizy`
bands and single-visit depths.

## Goal

A telescope-agnostic config carrying a survey's (a) bands/filters, (b) single-
visit 5-sigma depths (m5), (c) cadence, (d) footprint -- so the same simulation
and detection code can target Rubin, Roman, ZTF, OGLE, ... instead of LSST only.

## Prior art -- what exists (and the gap)

There is **no standalone, permissively-licensed Python object that covers all
four axes**. The landscape splits three ways:

- **Per-visit observation tables** are the de-facto survey carriers, but they are
  flat tabular formats, not config objects: the OpSim DB schema, `rubin_scheduler`
  `ObservationArray`, the sncosmo obs table, the SNANA `SIMLIB` format.
- **Filter/bandpass registries** solve only the band axis: `speclite` (BSD-3),
  sncosmo `Bandpass` (BSD-3), `synphot` (BSD-3).
- **Microlensing modeling packages** (`pyLIMA`, `MulensModel`) carry a
  per-telescope filter + a lightcurve but **no m5, cadence, or footprint** -- they
  consume data, they do not define surveys.

**Key constraint: `rubin_sim` is GPL-3.0** (verified from its LICENSE file).
rubin-sim-ml already imports it for the MAF metric, which is fine -- but the
*shared* config layer must NOT depend on it, or copyleft leaks into nsc-ml and
the portfolio package. Keep `rubin_sim` as an optional, isolated backend.

## Recommendation -- compose + a thin config

Don't reinvent a framework; don't pull `rubin_sim` into the shared layer.

1. Roll a small (~100-line) `SurveyConfig` dataclass whose vocabulary **mirrors
   the OpSim columns** (`fieldRA`, `fieldDec`, `observationStartMJD`, `filter`,
   `fiveSigmaDepth`), so Rubin opsim baselines load with zero translation and
   nsc-ml gets a clean target.
2. **Compose `speclite`** (BSD-3) for actual filter-response curves (it bundles
   lsst/sdss/decam/panstarrs/hsc/Euclid/gaia/bessell; add custom curves for
   ZTF/OGLE).
3. Support **both** a per-band scalar `m5` (survey-design mode -- what nsc-ml
   needs) and an optional per-visit table (opsim mode -- what rubin-sim-ml has),
   so one schema serves both repos.
4. Adapters wrap existing libs: `from_opsim()` (optional `rubin_sim`/sqlite
   backend), `to_sncosmo_obs()` (BSD event realization; does the standard
   `m5 -> skynoise` conversion -- OpSimSummary is the reference implementation),
   optional `to_simlib()` (SNANA interchange).

Net: **compose** (speclite, sncosmo) + **build** the thin config object +
**wrap** rubin_sim as an optional GPL-isolated backend. This avoids both
reinvention and license entanglement.

## Proposed schema (sketch -- not yet implemented)

```python
@dataclass
class Band:
    name: str                         # 'g','r',... or 'W149','Z087'
    speclite_name: str | None = None  # -> reuse speclite curve, e.g. 'lsst2023-g'
    m5_single_visit: float | None = None   # 5-sigma point-source limiting mag (AB)
    zp: float | None = None; zpsys: str = "ab"; gain: float = 1.0   # sncosmo vocab

@dataclass
class Footprint:                      # the genuinely new bit (nothing reusable)
    fields_radec: "np.ndarray | None" = None   # (Nfield, 2) deg
    region: str | None = None                  # 'galactic-bulge' | healpix path
    area_sq_deg: float | None = None

@dataclass
class SurveyConfig:
    name: str                          # 'rubin-baseline-v4','ztf-public','ogle-iv'
    bands: dict[str, Band]
    footprint: Footprint
    cadence_days: dict[str, float] | None = None     # design mode (per band)
    visits: "astropy.table.Table | None" = None      # data mode: OpSim-shaped rows
    mjd_range: tuple[float, float] | None = None
    pixel_scale_arcsec: float | None = None
    telescope: str | None = None
    # adapters: to_sncosmo_obs(), from_opsim(sqlite), to_simlib(path)
```

| Axis | Reuse | Build |
|---|---|---|
| bands / curves | speclite (BSD) | `Band` + name->speclite map |
| depth m5 | rubin_sim `calc_m5` as *optional* backend (GPL); opsim `fiveSigmaDepth` | per-band `m5` + `m5->skynoise` converter |
| cadence | opsim visit table; SIMLIB MJDs | `cadence_days` design dict; `from_opsim` |
| footprint | opsim RA/Dec set | `Footprint` (field list or region) -- new |

## Seed data -- every depth needs a cited anchor

Do not hardcode unsourced numbers (per the design-doc evidence rule). Anchors:
- Rubin ugrizy single-visit m5 -- SMTN-002 / `rubin_sim` `SysEngVals`.
- ZTF g/r/i -- ZTF survey/system papers.
- OGLE-IV I/V -- OGLE-IV survey docs.
- Roman Z087/W149/F184 (exp 290/52/145 s) -- Penny et al. 2019.

## Open decision (yours)

Where should the shared config live?
1. A small **shared package** both repos depend on (cleanest DRY; one more thing
   to install/maintain).
2. A **vendored module** copied into each repo (no packaging; manual sync).
3. **Start here in rubin-sim-ml**, designed location-agnostic, **extract** to a
   shared package once it stabilizes (lowest friction now).

## References

- rubin_scheduler `ObservationArray` -- https://github.com/lsst/rubin_scheduler
  (field names verified only via a fetch summary -- confirm against the installed
  version before coding adapters).
- rubin_sim (GPL-3.0) -- https://rubin-sim.lsst.io/ ; m5/SNR theory SMTN-002
  https://smtn-002.lsst.io/ ; data https://rubin-sim.lsst.io/data-download.html
- OpSim column schema -- https://www.lsst.org/scientists/simulations/opsim
  (the public column doc is a v3.x snapshot; confirm against the actual baseline
  DB this repo uses).
- sncosmo (BSD-3) -- https://sncosmo.readthedocs.io/en/stable/simulation.html
- speclite (BSD-3) -- https://speclite.readthedocs.io/en/latest/filters.html
- SNANA SIMLIB -- https://snana-starterkit.readthedocs.io/ ; arXiv:0908.4280
- OpSimSummary (opsim -> SIMLIB/sncosmo bridge) -- https://github.com/LSSTDESC/OpSimSummary
- pyLIMA -- https://github.com/ebachelet/pyLIMA ; MulensModel --
  https://rpoleski.github.io/MulensModel/ (licenses not definitively confirmed --
  verify before depending).

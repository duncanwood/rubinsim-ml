"""Shared test harness for the rubinml audit suite.

Loads rubinml.events as a standalone module (bypassing the package __init__,
which pulls in rubin_sim via `from .rubinsim import *`), installs a behavior-
preserving tqdm shim, and provides the canonical small source set used by the
golden / parity tests.

Design notes (all of this lives in the harness, NOT the library):
  - Seeding: the legacy numpy RNG inside make_events is unseeded. The tests seed
    it (np.random.seed(0)) for deterministic goldens.
  - tqdm shim: events.py does `from tqdm.notebook import tqdm`, which raises at
    call time outside a Jupyter+ipywidgets session. The shim is a passthrough
    that only wraps iteration; it changes no computed value or RNG draw.
  - LensCalcPy: a stale PyPI build of 0.0.3 has an einstein_rad() signature bug
    (pbh.py calls einstein_rad(dl, mass), missing ds) that breaks the rate path
    under numba. The editable repo checkout at ../LensCalcPy is correct, so we
    prepend it to sys.path when present.
See AUDIT.md for the library-side recommendations these shims stand in for.
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                  # .../rubin-sim-ml
EVENTS_PY = REPO / "rubinml" / "rubinml" / "events.py"
FIXTURES = HERE / "fixtures"
GOLDEN = HERE / "golden"

# Prefer the editable repo LensCalcPy over a possibly-stale site-packages build.
_LCP = REPO.parent / "LensCalcPy"                   # .../microlensing/LensCalcPy
if (_LCP / "LensCalcPy" / "__init__.py").exists():
    sys.path.insert(0, str(_LCP))


def install_tqdm_shim():
    if isinstance(sys.modules.get("tqdm.notebook"), types.ModuleType) and \
            getattr(sys.modules["tqdm.notebook"], "_rubinml_shim", False):
        return
    shim = types.ModuleType("tqdm.notebook")
    shim._rubinml_shim = True
    shim.tqdm = lambda iterable=None, *a, **k: (iterable if iterable is not None else [])
    sys.modules["tqdm.notebook"] = shim


def load_events():
    """Import rubinml/rubinml/events.py as a standalone module."""
    install_tqdm_shim()
    spec = importlib.util.spec_from_file_location("rubinml_events", str(EVENTS_PY))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rubinml_events"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_sources():
    """Canonical small synthetic source set toward the Galactic bulge.

    Columns: the MC core uses gall/galb/mu0; make_full_event_df uses ra/dec and
    the six *mag columns. The real catalog (rand_tristar_10_000_000.parquet) is
    an external dependency not present on this machine; this stand-in is fixed
    and deterministic so the goldens are reproducible. A human-readable copy is
    committed at tests/fixtures/sources_small.csv.
    """
    import numpy as np
    import pandas as pd

    n = 8
    return pd.DataFrame({
        "gall": np.linspace(-4.0, 4.0, n),
        "galb": np.linspace(-5.0, -1.5, n),
        "mu0":  np.linspace(13.8, 15.2, n),
        "ra":   np.linspace(264.0, 269.0, n),
        "dec":  np.linspace(-30.0, -26.0, n),
        "umag": np.linspace(22.5, 24.5, n),
        "gmag": np.linspace(21.5, 23.5, n),
        "rmag": np.linspace(20.5, 22.5, n),
        "imag": np.linspace(20.0, 22.0, n),
        "zmag": np.linspace(19.5, 21.5, n),
        "ymag": np.linspace(19.2, 21.2, n),
    })


# Canonical make_events parameters for the seeded golden.
MAKE_EVENTS_PARAMS = dict(
    n_survey_events=10,
    u_t=5.0,
    t_min=24.0,
    t_max=24.0 * 365 * 10,
    pbhmass=1.0,
    ntoss=10,
)
SEED = 0

# (l, b, mu0) probes for the rate goldens. ds = kpc_from_mu0(mu0).
RATE_PROBES = [
    (2.0, -3.0, 14.5),
    (1.0, -2.0, 14.0),
    (3.0, -4.0, 15.0),
]

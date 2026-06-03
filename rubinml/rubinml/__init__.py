"""rubinml: Monte-Carlo microlensing event simulation + Rubin/LSST detection.

The MC core (`events`) imports with a minimal dependency set. Plotting (`plots`,
needs matplotlib/seaborn) and the metric orchestration (`rubinsim`, needs
rubin_sim) are optional, imported only if available, so `import rubinml`
works for the simulation half on its own.
"""
from . import events
from .events import (
    wrap_degrees,
    kpc_from_mu0,
    mu0_from_kpc,
    differential_rate_integrand_mw_maker,
    source_lensing_rate,
    calculate_lensing_rates,
    load_rates_from_file,
    rates_to_rubin_counts,
    rubin_counts_from_rates_file,
    get_partial_rates,
    sample_density_single_source,
    sample_density_single_source_log,
    make_events,
    make_full_event_df,
    N_TRISTAR,
    SURVEY_HOURS,
    DAILY_CADENCE_HOURS,
)

__all__ = [
    "events",
    "wrap_degrees",
    "kpc_from_mu0",
    "mu0_from_kpc",
    "differential_rate_integrand_mw_maker",
    "source_lensing_rate",
    "calculate_lensing_rates",
    "load_rates_from_file",
    "rates_to_rubin_counts",
    "rubin_counts_from_rates_file",
    "get_partial_rates",
    "sample_density_single_source",
    "sample_density_single_source_log",
    "make_events",
    "make_full_event_df",
    "N_TRISTAR",
    "SURVEY_HOURS",
    "DAILY_CADENCE_HOURS",
]

try:
    from . import plots
    __all__.append("plots")
except ImportError:  # matplotlib/seaborn not installed
    plots = None

try:
    from . import rubinsim
    __all__.append("rubinsim")
except ImportError:  # rubin_sim not installed, simulation half still usable
    rubinsim = None

"""Package-surface test: the de-starred __init__ keeps the public API and makes
plots/rubinsim optional (AUDIT #10). Runs in the full env; the without-rubin_sim
path is validated separately (see AUDIT 'Status')."""
import unittest


class PackageApi(unittest.TestCase):
    def test_public_api(self):
        import lensemble
        for name in ("make_events", "make_full_event_df", "source_lensing_rate",
                     "rates_to_rubin_counts", "calculate_lensing_rates",
                     "N_TRISTAR", "SURVEY_HOURS", "DAILY_CADENCE_HOURS"):
            self.assertTrue(hasattr(lensemble, name), f"missing lensemble.{name}")
        self.assertTrue(callable(lensemble.make_events))
        self.assertIsNotNone(lensemble.events)            # core always imports

    def test_optional_submodules_present_as_attrs(self):
        import lensemble
        # plots (matplotlib/seaborn) and rubinsim (rubin_sim) are optional: the
        # attribute always exists, bound to the module or to None.
        self.assertTrue(hasattr(lensemble, "plots"))
        self.assertTrue(hasattr(lensemble, "rubinsim"))


if __name__ == "__main__":
    unittest.main()

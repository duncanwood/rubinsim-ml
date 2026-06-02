"""Package-surface test: the de-starred __init__ keeps the public API and makes
plots/rubinsim optional (AUDIT #10). Runs in the full env; the without-rubin_sim
path is validated separately (see AUDIT 'Status')."""
import unittest


class PackageApi(unittest.TestCase):
    def test_public_api(self):
        import rubinml
        for name in ("make_events", "make_full_event_df", "source_lensing_rate",
                     "rates_to_rubin_counts", "calculate_lensing_rates",
                     "N_TRISTAR", "SURVEY_HOURS", "DAILY_CADENCE_HOURS"):
            self.assertTrue(hasattr(rubinml, name), f"missing rubinml.{name}")
        self.assertTrue(callable(rubinml.make_events))
        self.assertIsNotNone(rubinml.events)            # core always imports

    def test_optional_submodules_present_as_attrs(self):
        import rubinml
        # plots (matplotlib/seaborn) and rubinsim (rubin_sim) are optional: the
        # attribute always exists, bound to the module or to None.
        self.assertTrue(hasattr(rubinml, "plots"))
        self.assertTrue(hasattr(rubinml, "rubinsim"))


if __name__ == "__main__":
    unittest.main()

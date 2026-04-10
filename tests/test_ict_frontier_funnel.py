import unittest

from research.ict.analyze_ict_frontier_funnel import _build_funnel_summary


class TestICTFrontierFunnel(unittest.TestCase):
    def test_build_funnel_summary_aggregates_stage_counts(self):
        metadata = {
            "bullish_sweep_candidates": 10,
            "bearish_sweep_candidates": 6,
            "bullish_sweeps": 4,
            "bearish_sweeps": 2,
            "bullish_shift_candidates": 3,
            "bearish_shift_candidates": 1,
            "bullish_shifts": 2,
            "bearish_shifts": 1,
            "bullish_retest_candidates": 2,
            "bearish_retest_candidates": 1,
            "long_entries": 2,
            "short_entries": 0,
            "delivery_missing_shifts": 4,
            "score_filtered_shifts": 1,
            "macro_timing_filtered_sweeps": 3,
            "kill_zone_filtered_sweeps": 2,
            "sweep_blocked_by_existing_pending": 5,
            "sweep_expired_before_shift": 2,
            "armed_setup_expired_before_retest": 1,
        }

        funnel = _build_funnel_summary(metadata)

        self.assertEqual(funnel["stages"]["raw_sweep_candidates"], 16)
        self.assertEqual(funnel["stages"]["accepted_sweeps"], 6)
        self.assertEqual(funnel["stages"]["shift_candidates"], 4)
        self.assertEqual(funnel["stages"]["armed_setups"], 3)
        self.assertEqual(funnel["stages"]["retest_candidates"], 3)
        self.assertEqual(funnel["stages"]["entries"], 2)
        self.assertEqual(funnel["direction_breakdown"]["bullish_armed_setups"], 2)
        self.assertEqual(funnel["conversion_rates_pct"]["raw_sweep_to_entry_rate"], 12.5)
        self.assertEqual(funnel["filtered_breakdown"]["sweep_blocked_by_existing_pending"], 5)
        self.assertEqual(funnel["top_filters"][0]["label"], "sweep_blocked_by_existing_pending")


if __name__ == "__main__":
    unittest.main()

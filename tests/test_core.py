from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import event_f1
import prepare_data
import saocp
import select_validation_config
import verify_locked_results


class CoreTests(unittest.TestCase):
    def test_pm_formula_and_feature_order(self) -> None:
        source = pd.DataFrame(
            {
                "B": [5.0],
                "Bx_rms": [1.0],
                "By_rms": [2.0],
                "Bz_rms": [2.0],
                "Np": [4.0],
                "V": [400.0],
                "Vth": [30.0],
            }
        )
        result = prepare_data.add_derived_features(source)
        expected = 1e-18 * 5.0**2 / (2.0 * constants.mu_0)
        self.assertAlmostEqual(float(result.loc[0, "Pm"]), expected)
        self.assertEqual(tuple(result.columns[-4:]), prepare_data.DERIVED_FEATURES)

    def test_saocp_does_not_use_current_block_labels_for_current_threshold(self) -> None:
        probability = np.asarray([0.1, 0.8, 0.2, 0.9])
        calibration = np.asarray([0.1, 0.2, 0.3, 0.4])
        first = saocp.make_calibrator(calibration, coverage=0.8, lifetime=8)
        second = saocp.make_calibrator(calibration, coverage=0.8, lifetime=8)
        _, threshold_a, _ = saocp.online_predict(
            probability,
            np.asarray([0, 0, 0, 0]),
            first,
            coverage=0.8,
            block_size=2,
            policy="positive_inclusion",
        )
        _, threshold_b, _ = saocp.online_predict(
            probability,
            np.asarray([1, 1, 0, 0]),
            second,
            coverage=0.8,
            block_size=2,
            policy="positive_inclusion",
        )
        np.testing.assert_allclose(threshold_a[:2], threshold_b[:2])

    def test_event_constructor_never_bridges_large_observation_gap(self) -> None:
        time_index = pd.DatetimeIndex(
            [
                "2010-01-01 00:00",
                "2010-01-01 00:05",
                "2010-01-02 00:00",
                "2010-01-02 00:05",
            ]
        )
        protocol = event_f1.EventProtocol(
            correction_points=0,
            minimum_points=1,
            merge_hours=12.0,
            max_observation_gap_minutes=30.0,
        )
        events = event_f1.construct_events(np.ones(4), time_index, protocol)
        self.assertEqual(len(events), 2)

    def test_legacy_event_f1_reproduces_many_to_many_catalog_counting(self) -> None:
        origin = pd.Timestamp("2010-01-01")
        predicted = [event_f1.Event(origin, origin + pd.Timedelta(hours=10))]
        truth = [
            event_f1.Event(origin, origin + pd.Timedelta(hours=4)),
            event_f1.Event(origin + pd.Timedelta(hours=6), origin + pd.Timedelta(hours=9)),
        ]
        metrics = event_f1.event_f1(predicted, truth)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 0, 0))
        self.assertEqual(metrics["f1"], 1.0)

    def test_one_to_one_event_f1_rejects_all_period_shortcut(self) -> None:
        origin = pd.Timestamp("2010-01-01")
        time_index = pd.date_range(origin, periods=30 * 24, freq="1h")
        all_one_labels = np.ones(len(time_index), dtype=np.uint8)
        protocol = event_f1.EventProtocol(
            correction_points=0,
            minimum_points=0,
            merge_hours=12.0,
            max_observation_gap_minutes=90.0,
        )
        truth = [
            event_f1.Event(origin + pd.Timedelta(days=2), origin + pd.Timedelta(days=3)),
            event_f1.Event(origin + pd.Timedelta(days=8), origin + pd.Timedelta(days=9)),
            event_f1.Event(origin + pd.Timedelta(days=14), origin + pd.Timedelta(days=15)),
        ]
        events = event_f1.construct_events(all_one_labels, time_index, protocol)
        self.assertEqual(len(events), 1)
        legacy = event_f1.event_f1(events, truth)
        robust = event_f1.event_f1_one_to_one(events, truth)
        self.assertEqual(legacy["f1"], 1.0)
        self.assertEqual((robust["tp"], robust["fp"], robust["fn"]), (1, 0, 2))
        self.assertAlmostEqual(float(robust["f1"]), 0.5)

    def test_compact_locked_result_arithmetic(self) -> None:
        count = verify_locked_results.verify(ROOT / "results" / "backbone_event_f1.csv")
        count += verify_locked_results.verify(
            ROOT / "results" / "pm_window_ablation_event_f1.csv"
        )
        self.assertEqual(count, 16)

    def test_primary_results_count_every_unmatched_prediction(self) -> None:
        for filename in (
            "backbone_event_f1.csv",
            "pm_window_ablation_event_f1.csv",
        ):
            frame = pd.read_csv(ROOT / "results" / filename)
            self.assertTrue((frame["fp"] == frame["predicted_events"] - frame["tp"]).all())
            self.assertTrue((frame["fn"] == frame["truth_events"] - frame["tp"]).all())
            self.assertTrue((frame["status"] == "locked_one_to_one").all())

    def test_primary_protocol_has_no_duration_fp_exemption(self) -> None:
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text())
        self.assertNotIn("false_positive_min_hours", protocol["event_constructor"])
        self.assertEqual(
            protocol["primary_metric"]["fp"],
            "all unmatched constructed predictions",
        )
        self.assertFalse(protocol["legacy_comparator_only"]["used_by_primary_metric"])

    def test_reporting_label_distinguishes_online_from_implementation(self) -> None:
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text())
        terminology = protocol["reporting_terminology"]
        self.assertEqual(terminology["comparison_labels"], ["Static", "Online"])
        self.assertEqual(terminology["implementation_id"], "saocp")
        self.assertIn("SAOCP-inspired", terminology["implementation_description"])
        self.assertFalse(terminology["upstream_reference_implementation_included"])

    def test_static_tie_break_order_is_locked(self) -> None:
        base = {"f1": 0.7, "precision": 0.6, "recall": 0.8}
        near = {**base, "threshold": 0.6}
        far = {**base, "threshold": 0.8}
        self.assertGreater(
            select_validation_config.static_key(near),
            select_validation_config.static_key(far),
        )


if __name__ == "__main__":
    unittest.main()

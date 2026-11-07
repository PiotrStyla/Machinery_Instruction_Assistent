import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_machine_training import experiment_state, metric_series


class TrainingChartTest(unittest.TestCase):
    def test_baseline_validation_and_final_test_are_distinct_series(self):
        rows = [
            {"event": "step", "global_step": 1, "loss": 4.0},
            {"event": "evaluation", "global_step": 0, "split": "valid", "mean_loss": 3.0, "exact_match_rate": 0.0},
            {"event": "evaluation", "global_step": 0, "split": "test", "mean_loss": 3.2, "exact_match_rate": 0.0},
            {"event": "evaluation", "global_step": 24, "split": "valid", "mean_loss": 2.5, "exact_match_rate": 0.25},
            {"event": "evaluation", "global_step": 24, "split": "test", "mean_loss": 2.7, "exact_match_rate": 0.5},
        ]
        self.assertEqual(metric_series(rows, "train_loss"), [(1.0, 4.0)])
        self.assertEqual(metric_series(rows, "valid_loss", "valid"), [(0.0, 3.0), (24.0, 2.5)])
        self.assertEqual(metric_series(rows, "exact_match", "test"), [(0.0, 0.0), (24.0, 50.0)])

    def test_missing_score_is_not_a_zero_measurement(self):
        self.assertEqual(metric_series([{"event": "evaluation", "global_step": 0, "split": "test"}], "exact_match", "test"), [])

    def test_invalid_measurement_cannot_be_plotted_as_a_valid_value(self):
        with self.assertRaises(ValueError):
            metric_series([{"event": "step", "global_step": 1, "loss": float("nan")}], "train_loss")

    def test_epoch_run_end_does_not_mean_complete_training_experiment(self):
        rows = [{"event": "experiment_start", "epochs": 3, "train_examples": 24},
                {"event": "step", "global_step": 24, "loss": 2},
                {"event": "run_end", "global_step": 24}]
        state = experiment_state(rows)
        self.assertEqual((state["status"], state["step"], state["total"]), ("running", 24, 72))
        self.assertEqual(experiment_state(rows + [{"event": "experiment_failed"}])["status"], "failed")
        self.assertEqual(experiment_state(rows + [{"event": "experiment_end", "global_step": 72}])["status"], "complete")


if __name__ == "__main__":
    unittest.main()

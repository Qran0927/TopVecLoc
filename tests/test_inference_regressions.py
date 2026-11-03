from pathlib import Path
import re
import unittest

import numpy as np
import torch

from utils.viz_localization import likelihood_overlay


ROOT = Path(__file__).resolve().parents[1]


class InferenceRegressionTests(unittest.TestCase):
    def test_main_uses_current_no_cross_attention_checkpoint_and_ckpt_rotations(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn("checkpoint-step-102000-14.827799.ckpt", source)
        self.assertNotRegex(source, r"Demo\([^)]*num_rotations\s*=\s*128")

    def test_main_builds_graph_from_same_bbox_as_map(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertNotIn("graph_bbox = bbox + float(tile_size_meters)", source)
        self.assertRegex(source, r"build_hetero_graph\(tiler\.osm,\s*bbox,\s*proj\)")

    def test_demo_loads_checkpoints_with_weights_only_disabled(self):
        source = (ROOT / "demo.py").read_text(encoding="utf-8")

        self.assertIn("weights_only=False", source)

    def test_likelihood_overlay_handles_zero_probability_map(self):
        overlay = likelihood_overlay(np.zeros((4, 4), dtype=np.float32))

        self.assertTrue(np.isfinite(overlay).all())

    def test_fast_template_sampler_does_not_force_cuda(self):
        source = (ROOT / "models" / "voting.py").read_text(encoding="utf-8")
        sampler_source = source.split("class UAVTemplateSamplerFast", 1)[1].split(
            "class UAVTemplateSampler", 1
        )[0]

        self.assertNotIn(".cuda()", sampler_source)

    def test_tile_size_is_used_as_map_width_not_radius(self):
        demo_source = (ROOT / "demo.py").read_text(encoding="utf-8")
        batch_graph_source = (ROOT / "batch_build_graphs.py").read_text(encoding="utf-8")
        graph_builder_source = (ROOT / "models" / "graph_builder.py").read_text(encoding="utf-8")

        self.assertRegex(demo_source, r"half_size_meters\s*=\s*float\(tile_size_meters\)\s*/\s*2")
        self.assertRegex(batch_graph_source, r"half_size_meters\s*=\s*tile_size_meters\s*/\s*2")
        self.assertRegex(graph_builder_source, r"half_size_meters\s*=\s*tile_size_meters\s*/\s*2")

    def test_graphgps_uses_groupnorm_for_single_sample_eval(self):
        source = (ROOT / "models" / "graphgps_encoder.py").read_text(encoding="utf-8")

        self.assertIn("nn.GroupNorm(1, output_dim)", source)
        self.assertNotIn("self.map_norm = nn.BatchNorm2d(output_dim", source)

    def test_offline_evaluator_reports_recall_metrics(self):
        source = (ROOT / "evaluate_ckpt.py").read_text(encoding="utf-8")

        self.assertIn("POSITION_THRESHOLDS = (1, 3, 5, 10, 20, 50)", source)
        self.assertIn("YAW_THRESHOLDS = (1, 3, 5, 10, 20)", source)
        self.assertIn("Position R@{threshold}m", source)
        self.assertIn("Yaw R@{threshold}deg", source)
        self.assertIn("center baseline error", source)
        self.assertIn("model better than center", source)
        self.assertIn("location_error", source)

    def test_offline_evaluator_diagnostics_include_center_baseline(self):
        from evaluate_ckpt import compute_diagnostics

        xy_errors = [0.0, 10.0, 5.0]
        yaw_errors = [0.5, 6.0, 15.0]
        pred_uvs = [[128.0, 128.0], [140.0, 128.0], [170.0, 128.0]]
        gt_uvs = [[128.0, 128.0], [130.0, 128.0], [140.0, 128.0]]

        diagnostics = compute_diagnostics(
            xy_errors,
            yaw_errors,
            pred_uvs,
            gt_uvs,
            image_size=256,
            pixel_per_meter=1,
        )

        self.assertAlmostEqual(diagnostics["center_errors"][1], 2.0)
        self.assertAlmostEqual(diagnostics["position_recall"][5], 2 / 3 * 100)
        self.assertAlmostEqual(diagnostics["yaw_recall"][10], 2 / 3 * 100)
        self.assertAlmostEqual(diagnostics["model_better_than_center"], 1 / 3 * 100)
        self.assertTrue(np.allclose(diagnostics["pred_uv"]["mean"], [146.0, 128.0]))

    def test_offline_evaluator_batches_single_node_graph_tensors(self):
        from evaluate_ckpt import to_batched_sample

        sample = {
            "image": torch.zeros(3, 4, 4),
            "uv": torch.zeros(2),
            "graph": {
                "poi": {
                    "coords": torch.zeros(1, 2),
                    "types": torch.zeros(1, dtype=torch.long),
                    "num_nodes": 1,
                },
                "bbox_min": torch.zeros(2),
            },
        }

        batch = to_batched_sample(sample, torch.device("cpu"))

        self.assertEqual(tuple(batch["image"].shape), (1, 3, 4, 4))
        self.assertEqual(tuple(batch["graph"]["poi"]["coords"].shape), (1, 1, 2))
        self.assertEqual(tuple(batch["graph"]["poi"]["types"].shape), (1, 1))
        self.assertEqual(tuple(batch["graph"]["bbox_min"].shape), (1, 2))
        self.assertEqual(batch["graph"]["poi"]["num_nodes"], 1)


if __name__ == "__main__":
    unittest.main()

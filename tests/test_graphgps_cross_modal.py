import unittest
import importlib.util
import tempfile
from pathlib import Path

import torch

from dataset.torch import collate_graphs


ROOT = Path(__file__).resolve().parents[1]


def _load_class(module_name, relative_path, class_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


GraphGPSMapEncoder = _load_class(
    "graphgps_encoder_under_test",
    "models/graphgps_encoder.py",
    "GraphGPSMapEncoder",
)
GlobalTransformer = _load_class(
    "graphgps_transformer_under_test",
    "models/graphgps_encoder.py",
    "GlobalTransformer",
)
DifferentiableRenderer = _load_class(
    "graphgps_renderer_under_test",
    "models/graphgps_encoder.py",
    "DifferentiableRenderer",
)
HybridMapEncoder = _load_class(
    "hybrid_map_encoder_under_test",
    "models/hybrid_map_encoder.py",
    "HybridMapEncoder",
)


def _tiny_graph(poi_type=0, area_type=0):
    return {
        "intersection": {
            "num_nodes": 2,
            "coords": torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            "degree": torch.tensor([0.0, 0.0]),
            "road_type_dist": torch.zeros(2, 10),
            "avg_segment_len": torch.tensor([0.0, 0.0]),
            "poi_density": torch.tensor([1.0, 0.0]),
            "building_density": torch.tensor([0.0, 1.0]),
        },
        "poi": {
            "num_nodes": 2,
            "coords": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            "types": torch.tensor([poi_type, 2], dtype=torch.long),
        },
        "area": {
            "num_nodes": 2,
            "coords": torch.tensor([[0.0, 1.0], [2.0, 1.0]]),
            "types": torch.tensor([area_type, 2], dtype=torch.long),
            "area_size": torch.tensor([20.0, 30.0]),
        },
        "road_edges": {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_type": torch.zeros(0, dtype=torch.long),
            "edge_length": torch.zeros(0),
        },
        "poi_to_intersection": {
            "edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            "edge_dist": torch.tensor([1.0, 1.0]),
        },
        "area_to_intersection": {
            "edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            "edge_dist": torch.tensor([1.0, 1.0]),
        },
        "bbox_min": torch.tensor([-8.0, -8.0]),
        "bbox_max": torch.tensor([8.0, 8.0]),
    }


class GraphGPSCrossModalTests(unittest.TestCase):
    def test_graphgps_returns_zero_features_for_empty_graph(self):
        graph = _tiny_graph()
        for node_type in ("intersection", "poi", "area"):
            graph[node_type]["num_nodes"] = 0
            for key, value in graph[node_type].items():
                if torch.is_tensor(value):
                    graph[node_type][key] = value[:0]
        for edge_type in (
            "road_edges",
            "poi_to_intersection",
            "area_to_intersection",
        ):
            for key, value in graph[edge_type].items():
                if key == "edge_index":
                    graph[edge_type][key] = value[:, :0]
                else:
                    graph[edge_type][key] = value[:0]

        encoder = GraphGPSMapEncoder(
            hidden_dim=16,
            output_dim=4,
            num_layers=1,
            output_size=8,
        ).eval()
        with torch.no_grad():
            output = encoder({"graph": graph})["map_features"][0]

        self.assertEqual(tuple(output.shape), (1, 4, 8, 8))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_graphgps_accepts_legacy_single_sample_batch(self):
        torch.manual_seed(3)
        encoder = GraphGPSMapEncoder(
            hidden_dim=16,
            output_dim=4,
            num_layers=1,
            output_size=8,
        ).eval()
        raw_graph = _tiny_graph()

        def add_batch_dim(value):
            if torch.is_tensor(value):
                return value.unsqueeze(0)
            if isinstance(value, dict):
                return {key: add_batch_dim(item) for key, item in value.items()}
            return value

        with torch.no_grad():
            raw_output = encoder({"graph": raw_graph})["map_features"][0]
            legacy_output = encoder(
                {"graph": add_batch_dim(raw_graph)}
            )["map_features"][0]

        self.assertTrue(
            torch.allclose(raw_output, legacy_output, atol=1e-5, rtol=1e-5)
        )

    def test_batched_renderer_matches_single_graph_calls(self):
        torch.manual_seed(5)
        renderer = DifferentiableRenderer(output_size=8, output_dim=4, knn_k=2)
        embeddings = torch.randn(2, 3, 4)
        coords = torch.tensor(
            [
                [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
            ]
        )
        mask = torch.tensor([[True, True, False], [True, True, True]])
        bbox_min = torch.tensor([[-2.0, -2.0], [-3.0, -3.0]])
        bbox_max = torch.tensor([[2.0, 2.0], [3.0, 3.0]])

        batched = renderer(embeddings, coords, mask, bbox_min, bbox_max)
        singles = torch.cat(
            [
                renderer(
                    embeddings[index : index + 1],
                    coords[index : index + 1],
                    mask[index : index + 1],
                    bbox_min[index : index + 1],
                    bbox_max[index : index + 1],
                )
                for index in range(2)
            ]
        )

        self.assertEqual(tuple(batched.shape), (2, 4, 8, 8))
        self.assertTrue(torch.allclose(batched, singles, atol=1e-5, rtol=1e-5))

    def test_graphgps_batch_matches_individual_eval_and_backpropagates(self):
        torch.manual_seed(19)
        encoder = GraphGPSMapEncoder(
            hidden_dim=16,
            output_dim=4,
            num_layers=1,
            output_size=8,
        )
        graphs = [_tiny_graph(poi_type=0, area_type=0), _tiny_graph(poi_type=1, area_type=1)]
        graph_batch = collate_graphs(graphs)

        encoder.eval()
        with torch.no_grad():
            batched = encoder({"graph": graph_batch})["map_features"][0]
            singles = torch.cat(
                [
                    encoder({"graph": collate_graphs([graph])})["map_features"][0]
                    for graph in graphs
                ]
            )

        self.assertEqual(tuple(batched.shape), (2, 4, 8, 8))
        self.assertTrue(torch.allclose(batched, singles, atol=1e-4, rtol=1e-4))

        encoder.train()
        output = encoder({"graph": graph_batch})["map_features"][0]
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in encoder.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_batched_transformer_ignores_padding(self):
        torch.manual_seed(7)
        layer = GlobalTransformer(hidden_dim=8, num_heads=2, dropout=0.0).eval()
        features = torch.randn(2, 3, 8)
        valid_mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        changed_padding = features.clone()
        changed_padding[0, 2] = 1e6

        with torch.no_grad():
            output = layer(features, valid_mask)
            changed_output = layer(changed_padding, valid_mask)

        self.assertEqual(tuple(output.shape), (2, 3, 8))
        self.assertTrue(
            torch.allclose(output[0, :2], changed_output[0, :2], atol=1e-5)
        )
        self.assertTrue(torch.equal(output[0, 2], torch.zeros(8)))

    def test_collate_graphs_pads_nodes_and_offsets_edges(self):
        graph_a = _tiny_graph()
        graph_b = _tiny_graph()
        graph_b["intersection"]["num_nodes"] = 3
        graph_b["intersection"]["coords"] = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
        )
        for field in (
            "degree",
            "avg_segment_len",
            "poi_density",
            "building_density",
        ):
            graph_b["intersection"][field] = torch.zeros(3)
        graph_b["intersection"]["road_type_dist"] = torch.zeros(3, 10)
        graph_a["road_edges"] = {
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
            "edge_type": torch.tensor([2], dtype=torch.long),
            "edge_length": torch.tensor([3.0]),
        }
        graph_b["road_edges"] = {
            "edge_index": torch.tensor([[1], [2]], dtype=torch.long),
            "edge_type": torch.tensor([3], dtype=torch.long),
            "edge_length": torch.tensor([4.0]),
        }

        batch = collate_graphs([graph_a, graph_b])

        self.assertEqual(tuple(batch["intersection"]["coords"].shape), (2, 3, 2))
        self.assertEqual(
            batch["intersection"]["mask"].tolist(),
            [[True, True, False], [True, True, True]],
        )
        self.assertEqual(batch["intersection"]["num_nodes"].tolist(), [2, 3])
        self.assertEqual(tuple(batch["bbox_min"].shape), (2, 2))
        self.assertEqual(
            batch["road_edges"]["edge_index"].tolist(),
            [[0, 4], [1, 5]],
        )
        self.assertEqual(batch["road_edges"]["graph_index"].tolist(), [0, 1])

    def test_exhaustive_voting_matches_map_spatial_size_with_current_config(self):
        from omegaconf import OmegaConf
        from models.maplocnet import MapLocNet

        cfg = OmegaConf.load(ROOT / "conf" / "maplocnet.yaml")
        OmegaConf.resolve(cfg)
        cfg.model.image_encoder.backbone.pretrained = False

        model = MapLocNet(cfg.model).eval()
        f_map = torch.randn(1, cfg.model.matching_dim, cfg.model.image_size, cfg.model.image_size)
        template_size = cfg.model.image_size // 2 + 1
        f_bev = torch.randn(1, cfg.model.matching_dim, template_size, template_size)

        with torch.no_grad():
            scores = model.exhaustive_voting(f_bev, f_map)

        self.assertEqual(
            tuple(scores.shape),
            (1, cfg.model.num_rotations, cfg.model.image_size, cfg.model.image_size),
        )

    def test_maplocnet_does_not_construct_cross_attention(self):
        from omegaconf import OmegaConf
        from models.maplocnet import MapLocNet

        cfg = OmegaConf.load(ROOT / "conf" / "maplocnet.yaml")
        OmegaConf.resolve(cfg)
        cfg.model.image_encoder.backbone.pretrained = False

        model = MapLocNet(cfg.model).eval()

        self.assertFalse(hasattr(model, "cross_attn"))
        self.assertFalse(any(key.startswith("cross_attn.") for key in model.state_dict()))

    def test_graphgps_map_features_depend_on_poi_and_area_semantics(self):
        torch.manual_seed(11)
        encoder = GraphGPSMapEncoder(
            hidden_dim=16,
            output_dim=4,
            num_layers=1,
            output_size=16,
        ).eval()

        with torch.no_grad():
            out_a = encoder({"graph": _tiny_graph(poi_type=0, area_type=0)})["map_features"][0]
            out_b = encoder({"graph": _tiny_graph(poi_type=1, area_type=1)})["map_features"][0]

        self.assertFalse(torch.allclose(out_a, out_b))

    def test_hybrid_map_encoder_recovers_cnn_feature_when_topology_residual_is_zero(self):
        from omegaconf import OmegaConf

        torch.manual_seed(13)
        conf = OmegaConf.create(
            {
                "type": "hybrid",
                "embedding_dim": 4,
                "output_dim": 4,
                "num_classes": {"areas": 1, "ways": 1, "nodes": 1},
                "backbone": None,
                "unary_prior": False,
                "hidden_dim": 16,
                "num_layers": 1,
            }
        )
        encoder = HybridMapEncoder(conf, matching_dim=4, output_size=16).eval()
        torch.nn.init.zeros_(encoder.graph_proj.weight)
        torch.nn.init.zeros_(encoder.graph_proj.bias)
        batch = {
            "map": torch.zeros(1, 3, 16, 16, dtype=torch.long),
            "graph": _tiny_graph(),
        }

        with torch.no_grad():
            cnn_feature = encoder.cnn_encoder(batch)["map_features"][0]
            pred = encoder(batch)
            fused = pred["map_features"][0]

        self.assertEqual(tuple(fused.shape), (1, 4, 16, 16))
        self.assertTrue(torch.allclose(fused, cnn_feature, atol=1e-6, rtol=1e-6))

    def test_hybrid_map_encoder_starts_as_gated_cnn_baseline(self):
        from omegaconf import OmegaConf

        conf = OmegaConf.create(
            {
                "type": "hybrid",
                "embedding_dim": 4,
                "output_dim": 4,
                "num_classes": {"areas": 1, "ways": 1, "nodes": 1},
                "backbone": None,
                "unary_prior": False,
                "hidden_dim": 16,
                "num_layers": 1,
                "fusion": {"gate_init": -5.0},
            }
        )
        encoder = HybridMapEncoder(conf, matching_dim=4, output_size=16).eval()
        batch = {
            "map": torch.zeros(1, 3, 16, 16, dtype=torch.long),
            "graph": _tiny_graph(),
        }

        with torch.no_grad():
            cnn_feature = encoder.cnn_encoder(batch)["map_features"][0]
            pred = encoder(batch)

        self.assertEqual(tuple(encoder.fusion_gate.shape), (1, 4, 1, 1))
        self.assertTrue(torch.equal(encoder.graph_proj.weight, torch.zeros_like(encoder.graph_proj.weight)))
        self.assertTrue(torch.equal(encoder.graph_proj.bias, torch.zeros_like(encoder.graph_proj.bias)))
        self.assertTrue(torch.equal(pred["map_features"][0], cnn_feature))
        self.assertIn("map_features_graph_residual", pred)
        self.assertTrue(
            torch.allclose(
                pred["fusion_gate"],
                torch.full((1, 4, 1, 1), torch.sigmoid(torch.tensor(-5.0))),
            )
        )

    def test_hybrid_map_encoder_rejects_mismatched_cnn_and_graph_batches(self):
        from omegaconf import OmegaConf

        conf = OmegaConf.create(
            {
                "type": "hybrid",
                "embedding_dim": 4,
                "output_dim": 4,
                "num_classes": {"areas": 1, "ways": 1, "nodes": 1},
                "backbone": None,
                "unary_prior": False,
                "hidden_dim": 16,
                "num_layers": 1,
            }
        )
        encoder = HybridMapEncoder(conf, matching_dim=4, output_size=16).eval()
        batch = {
            "map": torch.zeros(2, 3, 16, 16, dtype=torch.long),
            "graph": collate_graphs([_tiny_graph()]),
        }

        with self.assertRaisesRegex(ValueError, "batch sizes differ"):
            encoder(batch)

    def test_hybrid_map_encoder_can_load_cnn_checkpoint_without_freezing_cnn(self):
        from omegaconf import OmegaConf

        torch.manual_seed(17)
        conf = OmegaConf.create(
            {
                "type": "hybrid",
                "embedding_dim": 4,
                "output_dim": 4,
                "num_classes": {"areas": 1, "ways": 1, "nodes": 1},
                "backbone": None,
                "unary_prior": False,
                "hidden_dim": 16,
                "num_layers": 1,
                "freeze_cnn": False,
            }
        )
        source_encoder = HybridMapEncoder(conf, matching_dim=4, output_size=16).cnn_encoder
        state_dict = {
            f"model.map_encoder.{key}": value
            for key, value in source_encoder.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "cnn.ckpt"
            torch.save({"state_dict": state_dict}, ckpt_path)
            conf.cnn_checkpoint = str(ckpt_path)
            encoder = HybridMapEncoder(conf, matching_dim=4, output_size=16)

        self.assertTrue(any(p.requires_grad for p in encoder.cnn_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in encoder.cnn_encoder.parameters()))

    def test_maplocnet_can_select_cnn_map_encoder_with_type_key(self):
        from omegaconf import OmegaConf, open_dict
        from models.maplocnet import MapLocNet
        from models.map_encoder import MapEncoder

        cfg = OmegaConf.load(ROOT / "conf" / "maplocnet.yaml")
        with open_dict(cfg.model):
            cfg.model.map_encoder.type = "cnn"
        OmegaConf.resolve(cfg)
        cfg.model.image_encoder.backbone.pretrained = False

        model = MapLocNet(cfg.model).eval()

        self.assertIsInstance(model.map_encoder, MapEncoder)


if __name__ == "__main__":
    unittest.main()

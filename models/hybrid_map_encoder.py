"""
CNN + GNN （）

：
  CNN （），GNN （）。
  GNN  CNN ， CNN 。

：
  fused = proj_cnn(CNN_feat) + sigmoid(gate) * proj_gnn(GNN_feat)

， CNN 。，
 CNN ，。

 GraphGPS ：
  pure graphgps:  GNN（）
  hybrid:         = CNN  + GNN （）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.graphgps_encoder import GraphGPSMapEncoder
from models.map_encoder import MapEncoder


def cnn_map_encoder_conf(conf):
    """ CNN """
    allowed = (
        "name", "trainable", "freeze_batch_normalization",
        "embedding_dim", "output_dim", "num_classes",
        "backbone", "unary_prior",
    )
    return OmegaConf.create({key: conf[key] for key in allowed if key in conf})


class HybridMapEncoder(nn.Module):
    """CNN  + GNN ，。"""

    def __init__(self, conf, matching_dim: int, output_size: int):
        super().__init__()
        # ── ① CNN  ──
        self.cnn_encoder = MapEncoder(cnn_map_encoder_conf(conf))

        # ── ② GNN  ──
        self.graph_encoder = GraphGPSMapEncoder(
            hidden_dim=conf.get("hidden_dim", 128),
            output_dim=matching_dim,
            num_layers=conf.get("num_layers", 4),
            output_size=output_size,
        )

        # ── ③  ──
        cnn_output_dim = conf.get("output_dim", matching_dim)
        self.cnn_proj = (
            nn.Identity()
            if cnn_output_dim == matching_dim
            else nn.Conv2d(cnn_output_dim, matching_dim, 1)
        )
        self.graph_proj = nn.Conv2d(matching_dim, matching_dim, 1)
        gate_init = float(conf.get("fusion", {}).get("gate_init", -5.0))
        self.fusion_gate = nn.Parameter(
            torch.full((1, matching_dim, 1, 1), gate_init)
        )
        nn.init.zeros_(self.graph_proj.weight)
        nn.init.zeros_(self.graph_proj.bias)

    def forward(self, data):
        cnn_pred = self.cnn_encoder(data)         # VGG19 → [B, 8, 256, 256]
        graph_pred = self.graph_encoder(data)     # GraphGPS → [B, 8, 256, 256]

        fused_features = []
        cnn_features_fused_space = []
        graph_residuals = []
        gate = torch.sigmoid(self.fusion_gate)
        for cnn_feat, graph_feat in zip(cnn_pred["map_features"], graph_pred["map_features"]):
            if cnn_feat.shape[0] != graph_feat.shape[0]:
                raise ValueError(
                    "CNN and GraphGPS batch sizes differ: "
                    f"{tuple(cnn_feat.shape)} vs {tuple(graph_feat.shape)}"
                )
            if graph_feat.shape[-2:] != cnn_feat.shape[-2:]:
                graph_feat = F.interpolate(
                    graph_feat, size=cnn_feat.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            cnn_fused_space = self.cnn_proj(cnn_feat)
            graph_residual = self.graph_proj(graph_feat)
            fused = cnn_fused_space + gate * graph_residual
            fused_features.append(fused)
            cnn_features_fused_space.append(cnn_fused_space)
            graph_residuals.append(graph_residual)

        return {
            "map_features": fused_features,
            "map_features_cnn": cnn_pred["map_features"],
            "map_features_cnn_fused_space": cnn_features_fused_space,
            "map_features_graph": graph_pred["map_features"],
            "map_features_graph_residual": graph_residuals,
            "fusion_gate": gate,
            **({"log_prior": cnn_pred["log_prior"]} if "log_prior" in cnn_pred else {}),
        }

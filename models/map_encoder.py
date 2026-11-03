"""
CNN （， VGG19）

: OSM  [B, 3, H, W] (areas/ways/nodes )
:  [B, 8, H, W]（ FFT ）

:
  OSM  [B, 3, 256, 256]
    → Embedding(3×16) → [B, 48, 256, 256]
    → VGG19 + UNet → [B, 8, 256, 256]

 GraphGPS :
  CNN:  VGG19 （，）
  GNN: （，）
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .feature_extractor import FeatureExtractor


class MapEncoder(BaseModel):
    """ → CNN → """

    default_conf = {
        "type": "cnn",
        "hidden_dim": None,
        "num_layers": None,
        "fusion": {"gate_init": -5.0},
        "embedding_dim": "???",   # （=16）
        "output_dim": None,       # （=8， matching_dim ）
        "num_classes": "???",     # : {areas:7, ways:10, nodes:33}
        "backbone": "???",        # VGG19 
        "unary_prior": False,     # （）
    }

    def _init(self, conf):
        # ── ① :  →  ──
        # Embedding  = ( + 1)，+1 ""（ 0）
        self.embeddings = nn.ModuleDict({
            k: nn.Embedding(n + 1, conf.embedding_dim)
            for k, n in conf.num_classes.items()
        })
        # areas: Embedding(8, 16)   ways: Embedding(11, 16)   nodes: Embedding(34, 16)
        # : 3 × 16 = 48 

        input_dim = len(conf.num_classes) * conf.embedding_dim   # = 48

        # ── ②  ──
        output_dim = conf.output_dim
        if output_dim is None:
            output_dim = conf.backbone.output_dim
        if conf.unary_prior:
            output_dim += 1   # 

        # ── ③ （）──
        if conf.backbone is None:
            # :  1×1 conv 
            self.encoder = nn.Conv2d(input_dim, output_dim, 1)
        elif conf.backbone == "simple":
            # : 3 
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, output_dim, 3, padding=1),
            )
        else:
            # : VGG19 + UNet（feature_extractor.py  FeatureExtractor）
            self.encoder = FeatureExtractor({
                **conf.backbone,
                "input_dim": input_dim,
                "output_dim": output_dim,
            })

    def _forward(self, data):
        """
        : data["map"]  [B, 3, H, W] （0=, 1~N=）
        : map_features  [B, output_dim, H, W]

        :
          ①  Embedding → 3×16=48 
          ②  VGG19+UNet 
          ③ :  log_prior
        """
        # ①  Embedding 
        embeddings = [
            self.embeddings[k](data["map"][:, i])
            for i, k in enumerate(("areas", "ways", "nodes"))
        ]  #  [B, H, W, 16]
        embeddings = torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2)  # → [B, 48, H, W]

        # ② CNN 
        if isinstance(self.encoder, BaseModel):
            features = self.encoder({"image": embeddings})["feature_maps"]
        else:
            features = [self.encoder(embeddings)]

        # ③ : 
        pred = {}
        if self.conf.unary_prior:
            pred["log_prior"] = [f[:, -1] for f in features]  #  = 
            features = [f[:, :-1] for f in features]

        pred["map_features"] = features   # [B, 8, H, W]
        return pred

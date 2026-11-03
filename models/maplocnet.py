"""
MapLocNet 

:
   [3,256,256] → ResNet50 +FPN
   (/)   → 
  → FFT  → (x, y, yaw) 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

from models.base import BaseModel
from models.voting import (
    argmax_xyr, conv2d_fft_batchwise, expectation_xyr,
    log_softmax_spatial, mask_yaw_prior, nll_loss_xyr,
    nll_loss_xyr_smoothed, TemplateSampler,
    UAVTemplateSampler, UAVTemplateSamplerFast,
)
from . import get_model
from .graphgps_encoder import GraphGPSMapEncoder
from .hybrid_map_encoder import HybridMapEncoder, cnn_map_encoder_conf
from .map_encoder import MapEncoder
from .metrics import (
    AngleError, AngleRecall, Location2DError, Location2DRecall,
)


def one_sided_regression_penalty(student_nll, teacher_nll, margin=0.0):
    """Penalize only samples whose NLL regresses beyond ``margin``."""
    if margin < 0:
        raise ValueError("safeguard margin must be non-negative")
    return F.relu(student_nll - teacher_nll.detach() - margin)


class MapLocNet(BaseModel):
    """ FFT """

    default_conf = {
        "image_size": "???",
        "val_citys": "???",
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",              # BEV 
        "latent_dim": "???",          # 。
        "matching_dim": "???",            #  UAV 
        "scale_range": [0, 9],        # 
        "num_scale_bins": "???",       #  bin 
        "z_min": None,                 # BEV 
        "z_max": "???",                     # BEV /
        "x_max": "???",                 # BEV  x 
        "pixel_per_meter": "???",      # BEV 
        "num_rotations": "???",         # /
        "add_temperature": False,          #  softmax  temperature
        "normalize_features": False,         # 
        "padding_matching": "replicate",       # 、 padding 
        "apply_map_prior": True,                  # 
        "do_label_smoothing": False,          # 
        "sigma_xy": 1,
        "sigma_r": 2,
        # deprecated
        "depth_parameterization": "scale",        # 
        "norm_depth_scores": False,          # /
        "normalize_scores_by_dim": False,         # 
        "normalize_scores_by_num_valid": True,       # 
        "prior_renorm": True,                    # ，# 。
        "retrieval_dim": None,         # 
        "safeguard": {
            "weight": 0.0,
            "margin": 0.0,
        },
    }

    def _init(self, conf):
        # ──  ──
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        # ── ① : ResNet50 + FPN ──
        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)

        # ── ② :  ──
        map_encoder_type = conf.map_encoder.get("type", "cnn")
        if map_encoder_type == "graphgps":
            # （）
            self.map_encoder = GraphGPSMapEncoder(
                hidden_dim=conf.map_encoder.get("hidden_dim", 128),
                output_dim=conf.matching_dim,
                num_layers=conf.map_encoder.get("num_layers", 4),
                output_size=conf.image_size,
            )
        elif map_encoder_type == "hybrid":
            # CNN + GNN 
            self.map_encoder = HybridMapEncoder(
                conf.map_encoder,
                matching_dim=conf.matching_dim,
                output_size=conf.image_size,
            )
        else:
            #  CNN（VGG19 + UNet，）
            self.map_encoder = MapEncoder(cnn_map_encoder_conf(conf.map_encoder))

        self.template_sampler = UAVTemplateSampler(conf.num_rotations)

        # ── ⑥ （）──
        if conf.add_temperature:
            temperature = nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    # ================================================================
    # 
    # ================================================================

    def exhaustive_voting(self, f_bev, f_map):
        """
        FFT 。

        ①  L2 （）
        ②  64  → 
        ③ 
        ④  [B, 64, H, W] （64=）
        """
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        templates = self.template_sampler(f_bev)   # [1, 64, 8, 129, 129]

        with torch.autocast("cuda", enabled=False):  # FFT  float32
            scores = conv2d_fft_batchwise(
                f_map.float(),       # signal [1, 8, 256, 256]
                templates.float(),   # kernel [1, 64, 8, 129, 129]
                padding_mode=self.conf.padding_matching,
            )  # → [1, 64, 128, 128]

        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)
        return scores

    def _score_feature_pair(self, f_image, f_map, data, log_prior=None):
        """Match one image/map feature pair with the standard post-processing."""
        scores = self.exhaustive_voting(f_image, f_map).moveaxis(1, -1)
        if log_prior is not None and self.conf.apply_map_prior:
            scores = scores + log_prior.unsqueeze(-1)
        if "map_mask" in data:
            scores = scores.masked_fill(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        return scores, log_softmax_spatial(scores)

    def _forward(self, data):
        pred = {}

        # ── ①  ──
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]          # [B, 8, 256, 256]

        # ── ②  ──
        f_image = self.image_encoder(data)["feature_maps"][0]  # [B, 128, 128, 128]
        #f_image = self.feature_projection(f_image)             # → [B, 8, 128, 128]

        log_prior = pred_map.get("log_prior", [None])[0]
        scores, log_probs = self._score_feature_pair(
            f_image, f_map, data, log_prior=log_prior
        )

        teacher = {}
        safeguard_weight = float(self.conf.safeguard.weight)
        if (
            self.training
            and safeguard_weight > 0
            and "map_features_cnn_fused_space" in pred_map
        ):
            scores_cnn, log_probs_cnn = self._score_feature_pair(
                f_image,
                pred_map["map_features_cnn_fused_space"][0],
                data,
                log_prior=log_prior,
            )
            teacher = {
                "scores_cnn": scores_cnn,
                "log_probs_cnn": log_probs_cnn,
            }

        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)       # argmax 
            uvr_avg, _ = expectation_xyr(log_probs.exp())  # （）

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            **teacher,
        }

    # ================================================================
    #  & 
    # ================================================================

    def _pose_nll(self, log_probs, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        if self.conf.do_label_smoothing:
            return nll_loss_xyr_smoothed(
                log_probs, xy_gt, yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r, mask=data.get("map_mask"),
            )
        return nll_loss_xyr(log_probs, xy_gt, yaw_gt)

    def loss(self, pred, data):
        """
         (NLL)

        loss = -log_probs[, ]
        = -log()

         data:
          data["uv"]:               (u, v)
          data["roll_pitch_yaw"]:   yaw
        """
        nll = self._pose_nll(pred["log_probs"], data)
        total = nll
        loss = {"nll": nll}
        if "log_probs_cnn" in pred:
            teacher_nll = self._pose_nll(pred["log_probs_cnn"], data)
            safeguard = one_sided_regression_penalty(
                nll,
                teacher_nll,
                margin=float(self.conf.safeguard.margin),
            )
            total = total + float(self.conf.safeguard.weight) * safeguard
            loss["cnn_teacher_nll"] = teacher_nll.detach()
            loss["safeguard"] = safeguard
        loss["total"] = total
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        """
        :  ×3， ×1， ×6
        """
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError("uv_expectation", self.conf.pixel_per_meter),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_1m": Location2DRecall(1.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_3m": Location2DRecall(3.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_1°": AngleRecall(1.0, "yaw_max"),
            "yaw_recall_3°": AngleRecall(3.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }

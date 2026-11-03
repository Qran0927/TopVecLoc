"""
MapLocNet  Demo 

：
  ① Demo           — ，
  ② read_input_image_test —  + 
  ③ vis_image_feature     —  PCA 
  ④ show_result           —  + 
  ⑤ load_test_data        — 
"""

import time
import csv
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import torch
import torchvision.transforms as tvf
from PIL import Image
from geopy.distance import geodesic
from sklearn.decomposition import PCA

from dataset import UavMapDatasetModule
from dataset.image import resize_image, pad_image, rectify_image
from dataset.torch import unbatch_to_device
from evaluation.run import resolve_checkpoint_path, pretrained_models
from evaluation.viz import plot_example_single
from logger import logger
from models.maplocnet import MapLocNet
from models.voting import fuse_gps, argmax_xyr
from osm.raster import Canvas
from osm.tiling import TileManager
from osm.viz import Colormap, plot_nodes
from utils.exif import EXIF
from utils.geo import BoundaryBox, Projection
from utils.io import read_image
from utils.viz_2d import features_to_RGB, plot_images
from utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from utils.wrappers import Camera


# ============================================================
# 、
# ============================================================

def load_checkpoint(path):
    """
     PyTorch checkpoint 。

     weights_only=False（），
    。
    """
    try:
        return torch.load(
            path,
            map_location=(lambda storage, loc: storage),  #  CPU
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=(lambda storage, loc: storage))


def load_model_state(model, state):
    """
    。

     legacy  BatchNorm （ checkpoint ），
    ，。
    """
    incompatible = model.load_state_dict(state, strict=False)
    #  BatchNorm running stats  checkpoint 
    allowed_unexpected = {
        "map_encoder.map_norm.running_mean",
        "map_encoder.map_norm.running_var",
        "map_encoder.map_norm.num_batches_tracked",
    }
    unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Checkpoint is not compatible with the current model. "
            f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
        )
    if incompatible.unexpected_keys:
        print("Ignored legacy BatchNorm state:", incompatible.unexpected_keys)


# ============================================================
# 、
# ============================================================

def vis_image_feature(F):
    """
     → RGB 。

    ：
      [C, H, W] →  180×180 →  [N, C]
      → PCA  3  → L2  →  [0, 1]
      → reshape  [180, 180, 3] RGB 

    ：""
    """

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    F = F[:, 0:180, 0:180]           #  180×180 
    c, h, w = F.shape
    F = np.rollaxis(F, 0, 3)          # [C, H, W] → [H, W, C]
    F_flat = F.reshape(-1, c)         #  [H×W, C]

    flatten = normalize([F_flat])[0]
    flatten = np.nan_to_num(flatten, nan=0)

    pca = PCA(n_components=3)
    flatten = pca.fit_transform(flatten)     # [N, C] → [N, 3]
    flatten = (normalize(flatten) + 1) / 2   #  [0, 1]

    F_rgb, _ = np.split(flatten, [h * w], axis=0)
    F_rgb = F_rgb.reshape((h, w, 3))
    return F_rgb


def distance(lat1, lon1, lat2, lon2):
    """（）"""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def show_result(map_vis_image, pre_uv, pre_yaw):
    """
     + 。

    :
      map_vis_image:  (H, W, 3)
      pre_uv:         (u, v) 
      pre_yaw:        (0°~360°)
    :
      
    """
    # 
    gray_mask = np.zeros_like(map_vis_image)
    gray_mask.fill(128)
    image = cv2.addWeighted(map_vis_image, 1, gray_mask, 0, 0)

    # ：，
    u, v = pre_uv
    x1, y1 = int(u), int(v)
    angle = pre_yaw - 90                 # ：0° =  → 
    length = 20
    x2 = int(x1 + length * np.cos(np.radians(angle)))
    y2 = int(y1 + length * np.sin(np.radians(angle)))
    cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 0, 0), 2, 5, 0, 0.3)
    return image


# ============================================================
# 、
# ============================================================

def xyz_to_latlon(x, y, z):
    """ (XYZ) →  (lat, lon)（WGS84 ）"""
    wgs84 = pyproj.CRS('EPSG:4326')
    xyz_crs = pyproj.CRS('+proj=geocent +datum=WGS84 +units=m +no_defs')
    transformer = pyproj.Transformer.from_crs(xyz_crs, wgs84)
    lon, lat, _ = transformer.transform(x, y, z)
    return lat, lon


# ============================================================
# 、
# ============================================================

def crop_image(image, width, height):
    """"""
    x = int((image.shape[1] - width) / 2)
    y = int((image.shape[0] - height) / 2)
    return image[y:y + height, x:x + width]


def crop_square(image):
    """（）"""
    height, width = image.shape[:2]
    min_length = min(height, width)
    top = (height - min_length) // 2
    bottom = top + min_length
    left = (width - min_length) // 2
    right = left + min_length
    return image[top:bottom, left:right]


def read_input_image_test(image, prior_latlon, tile_size_meters):
    """
     +  GPS 。

    :
      image:              (numpy array  )
      prior_latlon:       GPS (latitude, longitude)
      tile_size_meters:  （）

    :
      image:                 resize  (256×256)
      camera:                (，demo  None)
      roll_pitch:            (，demo  None)
      proj:                  
      bbox:                  
      latlon:                
    """
    # 1.  resize
    image = cv2.resize(image, (256, 256))
    roll_pitch = None

    # 2. （ GPS， EXIF）
    latlon = None
    if prior_latlon is not None:
        latlon = prior_latlon
        logger.info("Using prior latlon %s.", prior_latlon)

    if latlon is None:
        #  EXIF  GPS 
        with open(image, "rb") as fid:
            exif = EXIF(fid, lambda: image.shape[:2])
        geo = exif.extract_geo()
        if geo:
            alt = geo.get("altitude", 0)
            latlon = (geo["latitude"], geo["longitude"], alt)
            logger.info("Using prior location from EXIF.")
        else:
            logger.info("Could not find any prior location in the image EXIF metadata.")

    latlon = np.array(latlon)

    # 3. 
    # Projection:  GPS （）
    proj = Projection(*latlon)
    center = proj.project(latlon)

    # 4.  bbox
    # ， tile_size/2 
    half_size_meters = float(tile_size_meters) / 2
    bbox = BoundaryBox(center, center) + half_size_meters

    camera = None
    image = cv2.resize(image, (256, 256))
    return image, camera, roll_pitch, proj, bbox, latlon


# ============================================================
# 、Demo  —— 
# ============================================================

class Demo:
    """
    MapLocNet 。

    ：
      demo = Demo(experiment_or_path="path/to/checkpoint.ckpt")
      uv, yaw, prob, neural_map, image, data, pred = demo.localize(image, camera, canvas)
    """

    def __init__(
        self,
        experiment_or_path: Optional[str] = "removed_baseline_MGL",
        device=None,
        **kwargs
    ):
        """
        :
          experiment_or_path: checkpoint   
          device:              (None =  CPU/GPU)
        """
        # ①  checkpoint 
        if experiment_or_path in pretrained_models:
            experiment_or_path, _ = pretrained_models[experiment_or_path]
        path = resolve_checkpoint_path(experiment_or_path)

        # ②  checkpoint
        ckpt = load_checkpoint(path)
        config = ckpt["hyper_parameters"]          # 
        config.model.update(kwargs)
        config.model.image_encoder.backbone.pretrained = False

        # ③  MapLocNet 
        model = MapLocNet(config.model).eval()

        # ④ （ "model." ）
        state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}
        load_model_state(model, state)

        # ⑤ 
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # ⑥ 
        self.model = model
        self.config = config
        self.device = device

    def prepare_data(
        self,
        image: np.ndarray,
        camera: Camera,
        canvas: Canvas,
        roll_pitch: Optional[Tuple[float]] = None,
    ):
        """
        。

        :
          image:       UAV  [H, W, 3] uint8 (0~255)
          camera:      （demo  None）
          canvas:      OSM  Canvas 
          roll_pitch:  /（）

        :
          : {'map': , 'image': }
        """
        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        return {
            'map': torch.from_numpy(canvas.raster).long(),
            'image': image,
        }

    def localize(self, image: np.ndarray, camera: Camera, canvas: Canvas, graph=None, **kwargs):
        """
        。

        :
          image:   UAV  [H, W, 3] uint8
          camera:  
          canvas:  OSM 
          graph:    dict（GraphGPS ，CNN ）

        :
          uv:          (u, v)
          yaw:         (0°~360°)
          prob:        [H, W, 64rotations]
          neural_map:  [8, H, W]
          image:      
          data_:      （ batch ）
          pred:       
        """
        # ── ①  ──
        data = self.prepare_data(image, camera, canvas, **kwargs)
        if graph is not None:
            data['graph'] = graph

        #  batch （ dict  tensor）
        def batch_value(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)[None]
            if isinstance(value, dict):
                return {key: batch_value(item) for key, item in value.items()}
            return value

        data_ = {k: batch_value(v) for k, v in data.items()}

        # ── ②  ──
        start = time.time()
        with torch.no_grad():            # 
            pred = self.model(data_)     # → MapLocNet._forward()
        elapsed = time.time() - start

        # ── ③ ： ──
        lp_xyr = pred["log_probs"].squeeze(0)  # [H, W, 64] 
        xyr = argmax_xyr(lp_xyr).cpu()         # argmax → (u, v, yaw)

        # ── ④  ──
        prob = lp_xyr.exp().cpu()                                          # （）
        neural_map = pred["map"]["map_features"][0].squeeze(0).cpu()       # 

        # ── ⑤  ──
        prob_xy = prob.max(-1).values
        scores = pred.get("scores")
        if scores is not None:
            scores = scores.detach().cpu()
            print(
                "scores stats:",
                f"min={scores.min().item():.4f}",
                f"max={scores.max().item():.4f}",
                f"std={scores.std().item():.4f}",
            )
        print(
            "prob_xy stats:",
            f"min={prob_xy.min().item():.6e}",
            f"max={prob_xy.max().item():.6e}",
            f"mean={prob_xy.mean().item():.6e}",
        )
        print('total time:', elapsed)

        return xyr[:2], xyr[2], prob, neural_map, data["image"], data_, pred


# ============================================================
# 、
# ============================================================

def load_test_data(root: Path, city: str, index: int):
    """
    。

    :
      root:   (UAV2MAP/)
      city:  
      index: 

    :
      data_dict:         
      uav_image_rgb:      UAV  (H, W, 3)
      map_vis_image:      (H, W, 3)
      uav_path:          
      [map_lat, map_long]: 
    """
    uav_image_path = root / city / 'uav'
    map_path = root / city / 'map'
    map_vis = root / city / 'map_vis'
    info_path = root / city / 'info.csv'

    #  info.csv
    info = np.loadtxt(str(info_path), dtype=str, delimiter=",", skiprows=1)
    id, uav_name, map_name, \
        uav_long, uav_lat, \
        map_long, map_lat, \
        tile_size_meters, pixel_per_meter, \
        u, v, yaw, dis = info[index]

    #  UAV 
    uav_image_rgb = cv2.imread(str(uav_image_path / uav_name))
    uav_image_rgb = cv2.cvtColor(uav_image_rgb, cv2.COLOR_BGR2RGB)

    # 
    map_vis_image = cv2.imread(str(map_vis / uav_name))
    map_vis_image = cv2.cvtColor(map_vis_image, cv2.COLOR_BGR2RGB)

    # 
    map_data = np.load(str(map_path / map_name))

    # 
    tfs = tvf.Compose([tvf.ToTensor(), tvf.Resize(256)])
    uav_image = tfs(uav_image_rgb)

    return {
        'map': torch.from_numpy(np.ascontiguousarray(map_data)).long().unsqueeze(0),
        'image': torch.tensor(uav_image).unsqueeze(0),
        'roll_pitch_yaw': torch.tensor((0, 0, float(yaw))).float().unsqueeze(0),
        'pixels_per_meter': torch.tensor(float(pixel_per_meter)).float().unsqueeze(0),
        "uv": torch.tensor([float(u), float(v)]).float().unsqueeze(0),
    }, uav_image_rgb, map_vis_image, str(uav_image_path / uav_name), [float(map_lat), float(map_long)]


# ============================================================
# 、
# ============================================================

if __name__ == '__main__':
    """
    ： checkpoint，。

    ：
      python demo.py
    """
    experiment_or_path = "weight/last-step-checkpointing.ckpt"
    image_path = 'images/00000.jpg'
    prior_latlon = (37.75704325989902, -122.435941445631)
    tile_size_meters = 128

    # 1.  Demo 
    demo = Demo(experiment_or_path=experiment_or_path, device='cpu')

    # 2. 
    image, camera, gravity, proj, bbox, true_prior_latlon = read_input_image_test(
        image_path,
        prior_latlon=prior_latlon,
        tile_size_meters=tile_size_meters,
    )

    # 3.  OSM 
    tiler = TileManager.from_bbox(
        projection=proj, bbox=bbox + 10, ppm=1, tile_size=tile_size_meters
    )
    canvas = tiler.query(bbox)

    # 4. 
    uv, yaw, prob, neural_map, image_rectified, data_, pred = demo.localize(
        image, camera, canvas
    )

    # 5. 
    prior_latlon_pred = proj.unproject(canvas.to_xy(uv))
    print(f": {prior_latlon_pred}")

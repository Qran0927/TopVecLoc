"""
MapLocNet Gradio Web 

： Web ， UAV  +  GPS ，
      OSM ， () + 。

：
  ①  + 
  ②  GPS  OSM  →  + 
  ③ ： + GraphGPS  →  → 
  ④ ：、、
"""

import os
import urllib.request

import cv2
import gradio as gr  # type: ignore
import numpy as np
import requests

from demo import Demo, read_input_image_test, show_result, vis_image_feature
from logger import logger
from models.graph_builder import build_hetero_graph  # OSM → 
from osm.tiling import TileManager                    # OSM 
from osm.viz import Colormap, GeoPlotter              #  + 
from utils.viz_2d import features_to_RGB, plot_images  # 
from utils.viz_localization import likelihood_overlay   # 

  
# ── （OSM API ）─────────────────────────────
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

session = requests.Session()
session.proxies = {
    "http": "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897",
}
proxy_handler = urllib.request.ProxyHandler(session.proxies)
urllib.request.install_opener(urllib.request.build_opener(proxy_handler))


# ──  ─────────────────────────────────────────────────
#  checkpoint 
experiment_or_path = (
    "./experiments/maplocnet__CNN+GNN_1/checkpoint-best-step-150000.ckpt"
)
# Demo 、、
model = Demo(experiment_or_path=experiment_or_path, device="cpu")

    
# ──  ─────────────────────────────────────────────────

def likelihood_for_display(prob):
    """
    。

    : prob —  (H, W, 64rotations)
    :  (H, W) （95% ）
    """
    prob_xy = prob.numpy().max(-1)         #  → 
    cutoff = np.quantile(prob_xy, 0.95)    # 95% （）
    display = np.clip(prob_xy - cutoff, 0, None)
    max_ = display.max()
    if not np.isfinite(max_) or max_ <= 0:
        return np.zeros_like(prob_xy)
    return display / max_


def trained_tile_size_meters():
    """（）"""
    return int(round(model.config.model.image_size / model.config.model.pixel_per_meter))


# ──  ──────────────────────────────────────────────

def demo_localize(image, longitude, latitude, tile_size_meters):
    """
    MapLocNet 。

    :
      image:              UAV  (numpy array, H×W×3)
      longitude, latitude:  GPS （）
      tile_size_meters:    （），

    :
      fig:      5  (UAV / + /  / UAV / )
      plotter.fig: 
      prior_latlon_pred: 
    """

    # ──  1： +  ──
    expected_tile_size = trained_tile_size_meters()
    if int(tile_size_meters) != expected_tile_size:
        logger.warning(
            "Overriding demo tile_size_meters=%s with trained map size %s.",
            tile_size_meters,
            expected_tile_size,
        )

    tile_size_meters = expected_tile_size
    prior_latlon = (latitude, longitude)

    # read_input_image_test:  resize +  +  bbox
    image, camera, gravity, proj, bbox, _ = read_input_image_test(
        image,
        prior_latlon=prior_latlon,
        tile_size_meters=tile_size_meters,
    )

    # ──  2： OSM  ──
    # TileManager.from_bbox:  bbox  OSM API 
    tiler = TileManager.from_bbox(
        projection=proj,
        bbox=bbox,
        ppm=1,                   #  = 1（1  = 1 ）
        tile_size=tile_size_meters,
    )
    canvas = tiler.query(bbox)   #  bbox 

    # ──  3：（GraphGPS ）──
    #  OSM  + POI +  → 
    graph = build_hetero_graph(tiler.osm, bbox, proj)

    # ──  4： ──
    # localize  MapLocNet._forward():
    #    → ResNet50 →  → 
    #    → GraphGPS  →  → 
    #   → FFT  → LogSoftmax → (u, v, yaw) 
    uv, yaw, prob, neural_map, image_rectified, data_, pred = model.localize(
        image, camera, canvas, graph=graph
    )

    # 
    prior_latlon_pred = proj.unproject(canvas.to_xy(uv))

    # ──  5： ──
    # 5a.  + 
    map_viz = Colormap.apply(canvas.raster)                         # OSM  → 
    map_vis_image_result = show_result(
        (map_viz * 255).astype(np.uint8), uv, yaw
    )  # 

    # 5b. UAV （PCA  RGB）
    uav_feature_rgb = vis_image_feature(pred["features_image"][0].cpu().numpy())

    # 5c. 
    map_viz_resized = cv2.resize(map_viz, (prob.numpy().shape[0], prob.numpy().shape[1]))
    likelihood_display = likelihood_for_display(prob)
    overlay = likelihood_overlay(
        likelihood_display,
        map_viz_resized.mean(-1, keepdims=True),
        p_rgb=1.0,
        p_alpha=1.0,
        thresh=0.05,
    )

    # 5d.  → RGB 
    (neural_map_rgb,) = features_to_RGB(neural_map.numpy())

    # 5e.  5 
    fig = plot_images(
        [image, map_vis_image_result / 255, overlay, uav_feature_rgb, neural_map_rgb],
        titles=["UAV image", "map", "likelihood", "UAV feature", "map feature"],
    )

    # ──  6： ──
    bbox_latlon = proj.unproject(canvas.bbox)
    plotter = GeoPlotter(zoom=16.5)
    plotter.raster(map_viz_resized, bbox_latlon, opacity=0.5)        # ：OSM 
    plotter.raster(
        likelihood_overlay(likelihood_display, p_rgb=1.0, p_alpha=1.0, thresh=0.05),
        proj.unproject(bbox),
    )                                                                # ：
    plotter.points(prior_latlon[:2], "red", name="location prior", size=10)      # ：
    plotter.points(proj.unproject(canvas.to_xy(uv)), "black", name="argmax", size=10)  # ：
    plotter.bbox(bbox_latlon, "blue", name="map tile")               # ：

    return fig, plotter.fig, str(prior_latlon_pred)


# ── Gradio  ──────────────────────────────────────────
title = "MapLocNet"
description = "UAV Vision-based Geo-Localization Using Vectorized Maps"

interface = gr.Interface(
    fn=demo_localize,          # 
    inputs=[
        "image",                                                     # 
        gr.Number(label="Prior location longitude"),                 # 
        gr.Number(label="Prior location latitude"),                  # 
        gr.Radio([256], value=256, label="Map size (meters)", info="trained map size"),
    ],
    outputs=["plot", "plot", "text"],  #  + 
    title=title,
    description=description,
    examples=[
        # ： +  +  + 
        ["images/00000.jpg", -122.435941445631, 37.75704325989902, 256],
        ["UAV2MAP_v1.0/UAV2MAP/SanFrancisco/uav/00065.jpg", -122.439549287093, 37.76469297766308, 256],
        ["UAV2MAP_v1.0/UAV2MAP/SanFrancisco/uav/00099.jpg", -122.426020795146, 37.74273619637015, 256],
        ["UAV2MAP_v1.0/UAV2MAP/SanFrancisco/uav/00136.jpg", -122.419879226396, 37.76078942318915, 256],
        ["UAV2MAP_v1.0/UAV2MAP/SanFrancisco/uav/00167.jpg", -122.404532868464, 37.77316405293505, 256],
        ["UAV2MAP_v1.0/UAV2MAP/SanFrancisco/uav/00168.jpg", -122.40925245662, 37.75616341400768, 256],
        ["image/Test/reference_images/offset_0_None/00000(1).jpg", -73.9226449755007, 40.63257430786041, 256],
        ["image/Test/reference_images/offset_0_None/00001(1).jpg", -73.9225883959109, 40.63257430786041, 256],
        ["image//uav_visual_data/Sec-A/Tra.2/03_0001.jpg", 119.8863278, 32.3006743, 256],
        ["image//uav_visual_data/Sec-A/Tra.3/03_0016.jpg", 119.8729702, 32.3040129, 256],
    ],
)

if __name__ == "__main__":
    interface.launch(share=True)  # share=True ，

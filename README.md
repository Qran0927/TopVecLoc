# TopVecLoc

## Release Contents

This package contains TopVecLoc source code, the selected checkpoint, complete matching evaluation results, the TopVecLoc comparison checkpoint and compatible source, and graph features for seven UAV2MAP cities.

## Installation

Create a Python environment and install `requirements.txt` and `requirements-baselines.txt`.

## Official UAV2MAP Dataset

Obtain the original imagery and metadata from the official UAV2MAP distribution. They are not redistributed in this package. Set `UAV2MAP_ROOT` to the extracted official dataset directory.

## Installing the Supplied Graph Files

Copy each directory from `data/uav2map_graphs/<City>/` into the matching official city directory. Each `.pt` file uses the same sample identifier as the official map tile.

## Checkpoints

The selected model is `checkpoints/topvecloc_best.ckpt`. The only baseline checkpoint is `checkpoints/baselines/vecmaplocnet.ckpt`.

## Evaluation

Use `eval_uav.py` or the scripts under `evaluation/` with `configs/topvecloc.yaml`.

## Training

Set `UAV2MAP_ROOT`, review `configs/topvecloc.yaml`, and run `train.py`.

## Baselines

TopVecLoc source compatibility and measured outputs are provided for research comparison. No external baseline implementation or weight is included.

## Licenses and Attribution

Original TopVecLoc software is MIT licensed. Graph data retain separate terms described in the accompanying data notice.

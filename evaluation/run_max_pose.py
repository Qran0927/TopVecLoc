"""Batched max-probability UAV pose evaluation; never overwrite prior results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dataset.torch import collate
from eval_uav import SafeUavDataset, move_batch_to_device
from eval_uav_helpers import compute_paper_position_errors, signed_angle_error_deg
from evaluation.adapted_geometry import summarize_results
from evaluation.inference_interventions import Intervention, VARIANTS
from module import GenericModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--intervention', choices=VARIANTS, default='none')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shift-m', type=float, default=128)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    if args.intervention != 'none' and args.batch_size != 1:
        parser.error('Interventions require original single-image inference')
    if args.stride < 1 or (args.limit is not None and args.limit < 1) or args.shift_m < 0:
        parser.error('Invalid subset or coordinate-shift settings')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    cfg = OmegaConf.load('conf/maplocnet.yaml')
    ds = SafeUavDataset(Path(cfg.data.root), 'SanFrancisco',
                        T.Compose([T.ToTensor(), T.Resize(cfg.data.image_size)]))
    if ds.available_sample_indices() != list(range(10000)):
        raise RuntimeError('Expected complete SanFrancisco IDs 0..9999')
    positions = list(range(0,len(ds),args.stride))
    if args.limit is not None:
        positions = positions[:args.limit]
    ds = Subset(ds,positions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate if args.batch_size > 1 else None,
                        pin_memory=True)
    model = GenericModule.load_from_checkpoint(args.checkpoint, cfg=cfg, strict=True,
        find_best=False, map_location='cpu').eval().cuda()
    args.output.mkdir(parents=True)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    config = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
        city='SanFrancisco', sample_count=len(ds), batch_size=args.batch_size,
        estimator='joint maximum-probability pose: uv_max and yaw_max',
        thresholds=[1,3,5,10], comparison='strict < threshold',
        yaw_error='abs((prediction-ground_truth+180)%360-180)',
        torch_version=torch.__version__, model_config=OmegaConf.to_container(cfg,resolve=True))
    config.update(intervention=args.intervention, seed=args.seed, shift_m=args.shift_m,
        stride=args.stride, limit=args.limit, training='none; fixed checkpoint inference intervention',
        context_removal='zero context projection outputs including biases AND POI/building density inputs',
        road_intervention='only road adjacency; original node descriptors and edge-attribute slots retained',
        coordinate_shift='PE inputs only; identical offset for all node types per tile; rendering unchanged',
        caveat='Not retrained ablations; checkpoint was previously selected using test-subset results')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows = []
    fields = ['index','status','pred_u','pred_v','pred_yaw','gt_u','gt_v','gt_yaw',
              'position_max_error_m','lateral_max_error_m','longitudinal_max_error_m',
              'yaw_error_deg','original_edges','changed_edges','swaps']
    with open(args.output/'per_sample.csv','w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        encoder = model.model.map_encoder.graph_encoder
        with torch.no_grad(), Intervention(encoder,args.intervention,args.seed,args.shift_m) as intervention:
            for batch in loader:
                batch, change_info = intervention.prepare(batch)
                pred = model(move_batch_to_device(batch,torch.device('cuda')))
                uv = pred['uv_max'].cpu()
                yaw = pred['yaw_max'].cpu()
                if not torch.isfinite(uv).all() or not torch.isfinite(yaw).all():
                    raise RuntimeError('Nonfinite maximum-probability pose')
                for j, index in enumerate(batch['sample_index'].tolist()):
                    target = batch['uv'][j]
                    angle = float(batch['roll_pitch_yaw'][j,-1])
                    row = dict(index=index,status='ok',pred_u=float(uv[j,0]),
                        pred_v=float(uv[j,1]),pred_yaw=float(yaw[j]),
                        gt_u=float(target[0]),gt_v=float(target[1]),gt_yaw=angle,
                        yaw_error_deg=abs(signed_angle_error_deg(float(yaw[j]),angle)))
                    row.update(change_info)
                    row.update(compute_paper_position_errors(uv_max=uv[j],uv_gt=target,
                        gt_yaw_deg=angle,pixel_per_meter=float(batch['pixels_per_meter'][j])))
                    writer.writerow(row)
                    rows.append(row)
                if len(rows)%200 == 0 or len(rows)==len(ds):
                    handle.flush()
                    print(f'Processed {len(rows)}/{len(ds)}',flush=True)
        handle.flush()
    report = summarize_results(pd.DataFrame(rows))
    report.update(estimator='maximum-probability',checkpoint_sha256=digest)
    report.update(intervention=args.intervention,seed=args.seed,
        changed_edge_samples=sum(r['changed_edges']>0 for r in rows),
        total_successful_swaps=sum(r['swaps'] for r in rows))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()

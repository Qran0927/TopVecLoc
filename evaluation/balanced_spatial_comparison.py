"""Reproducible illustrative regions: geometry-only membership, explicit outcome categories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.plot_recall_comparison import ROOT
from utils.geo import Projection

INPUTS = {
    'metadata':'UAV2MAP_v1.0/UAV2MAP/SanFrancisco/info.csv',
    'baseline':'experiments/adapted_baselines/sanfrancisco_v1/removed_baseline-inspired-aerial-cnn/per_sample.csv',
    'ours':'experiments/max_pose_comparison/current_sf_serial_v1/per_sample.csv',
}


def load_table(root=ROOT):
    info = pd.read_csv(root/INPUTS['metadata']).set_index('id')
    if info.index.tolist()!=list(range(10000)):
        raise ValueError('Expected complete SanFrancisco metadata')
    projection = Projection(float(info.uav_lat.mean()),float(info.uav_long.mean()))
    xy = projection.project(info[['uav_lat','uav_long']].to_numpy())
    table = pd.DataFrame(index=info.index)
    table.index.name = 'sample_id'
    table['east_m'],table['north_m'] = xy.T
    table['longitude'],table['latitude'] = info.uav_long,info.uav_lat
    table['gt_yaw'] = info.yaw
    table['pixel_per_meter'] = info.pixel_per_meter
    for model in ('baseline','ours'):
        pred = pd.read_csv(root/INPUTS[model]).set_index('index')
        if pred.index.tolist()!=info.index.tolist() or not pred.status.eq('ok').all():
            raise ValueError('Prediction IDs/status mismatch')
        np.testing.assert_allclose(pred.gt_yaw,info.yaw,rtol=0,atol=2e-5)
        np.testing.assert_allclose(pred[['gt_u','gt_v']],info[['u','v']],rtol=0,atol=2e-5)
        east = (pred.pred_u-pred.gt_u)/info.pixel_per_meter
        north = -(pred.pred_v-pred.gt_v)/info.pixel_per_meter
        position = np.hypot(east,north)
        yaw = np.abs((pred.pred_yaw-pred.gt_yaw+180)%360-180)
        np.testing.assert_allclose(position,pred.position_max_error_m,atol=1e-8)
        np.testing.assert_allclose(yaw,pred.yaw_error_deg,atol=1e-8)
        table[f'{model}_east_m'] = table.east_m+east.to_numpy()
        table[f'{model}_north_m'] = table.north_m+north.to_numpy()
        table[f'{model}_position_error_m'] = position.to_numpy()
        table[f'{model}_yaw_error_deg'] = yaw.to_numpy()
        table[f'{model}_pred_yaw'] = pred.pred_yaw.to_numpy()
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError('Nonfinite sample data')
    return table,projection


def order_spatially(table, ids):
    remaining = set(map(int,ids))
    current = min(remaining,key=lambda i:(table.at[i,'east_m'],table.at[i,'north_m'],i))
    ordered = [current]; remaining.remove(current)
    while remaining:
        current_xy = table.loc[current,['east_m','north_m']].to_numpy(float)
        current = min(remaining,key=lambda i:(float(np.linalg.norm(
            table.loc[i,['east_m','north_m']].to_numpy(float)-current_xy)),i))
        ordered.append(current); remaining.remove(current)
    return ordered


def build_candidates(table,cell_size=400,group_size=20):
    cells = np.floor(table[['east_m','north_m']].to_numpy(float)/cell_size).astype(int)
    grouping = table.assign(cell_x=cells[:,0],cell_y=cells[:,1])
    result = []
    pos_scale = float(table.baseline_position_error_m.mean())
    yaw_scale = float(table.baseline_yaw_error_deg.mean())
    for (cx,cy),group in grouping.groupby(['cell_x','cell_y'],sort=True):
        if len(group)<group_size:
            continue
        center = (np.array([cx,cy])+.5)*cell_size
        distances = np.linalg.norm(group[['east_m','north_m']].to_numpy(float)-center,axis=1)
        ranks = np.lexsort((group.index.to_numpy(),distances))[:group_size]
        ids = order_spatially(table,group.index[ranks].tolist())
        selected = table.loc[ids]
        item = dict(cell_x=int(cx),cell_y=int(cy),cell_sample_count=len(group),ids=ids,
                    center_east_m=float(center[0]),center_north_m=float(center[1]))
        for model in ('baseline','ours'):
            for metric in ('position_error_m','yaw_error_deg'):
                item[f'{model}_{metric}'] = float(selected[f'{model}_{metric}'].mean())
        item['position_gain_m'] = item['baseline_position_error_m']-item['ours_position_error_m']
        item['yaw_gain_deg'] = item['baseline_yaw_error_deg']-item['ours_yaw_error_deg']
        item['joint_gain'] = item['position_gain_m']/pos_scale+item['yaw_gain_deg']/yaw_scale
        steps = np.linalg.norm(np.diff(selected[['east_m','north_m']].to_numpy(float),axis=0),axis=1)
        item['maximum_spatial_step_m'] = float(steps.max())
        result.append(item)
    return result


def choose_cases(groups,pos_scale,yaw_scale):
    chosen = []
    def separated(g):
        return all(np.hypot(g['center_east_m']-c['center_east_m'],
                            g['center_north_m']-c['center_north_m'])>=500 for c in chosen)
    def pick(pool,score,category):
        eligible = [g for g in pool if separated(g)]
        if not eligible:
            raise ValueError(f'No eligible separated {category} case; do not silently relax selection rules')
        selected = dict(min(eligible,key=lambda g:(score(g),tuple(g['ids']))),category=category)
        chosen.append(selected)
        return selected
    failures = [g for g in groups if g['position_gain_m']<0 and g['yaw_gain_deg']<0
                and (g['ours_position_error_m']>=10 or g['ours_yaw_error_deg']>=10)]
    if not failures:
        raise ValueError('No joint failure region satisfies the prespecified rule')
    target = float(np.median([g['joint_gain'] for g in failures]))
    failure = pick(failures,lambda g:abs(g['joint_gain']-target),'failure')
    ties = [g for g in groups if abs(g['position_gain_m'])<=1 and abs(g['yaw_gain_deg'])<=1]
    tie = pick(ties,lambda g:abs(g['position_gain_m'])/pos_scale+abs(g['yaw_gain_deg'])/yaw_scale,'near_tie')
    wins = [g for g in groups if g['position_gain_m']>=3 and g['yaw_gain_deg']>=3]
    if not wins:
        raise ValueError('No improvement regions satisfy the prespecified rule')
    improvements = []
    for quantile in (.5,.75):
        target = float(np.quantile([g['joint_gain'] for g in wins],quantile))
        improvements.append(pick(wins,lambda g:abs(g['joint_gain']-target),'improvement'))
    return [*improvements,tie,failure]


def prepare(output):
    table,projection = load_table()
    groups = build_candidates(table)
    cases = choose_cases(groups,float(table.baseline_position_error_m.mean()),
                         float(table.baseline_yaw_error_deg.mean()))
    output.mkdir(parents=True)
    pd.DataFrame(groups).assign(ids=lambda f:f.ids.map(json.dumps)).to_csv(output/'candidate_pool.csv',index=False)
    summary = []
    manifest = []
    for panel,case in zip('abcd',cases):
        item = {k:v for k,v in case.items() if k!='ids'}
        item.update(panel=panel,sample_count=len(case['ids']))
        summary.append(item)
        frame = table.loc[case['ids']].copy()
        frame['sequence_order'] = np.arange(1,len(frame)+1)
        frame['panel'],frame['category'] = panel,case['category']
        manifest.append(frame.reset_index())
    pd.DataFrame(summary).to_csv(output/'case_summary.csv',index=False)
    pd.concat(manifest,ignore_index=True).to_csv(output/'selected_samples.csv',index=False)
    notes = dict(protocol='Illustrative spatial sequences; NOT temporal trajectories or temporal tracking.',
        cell_size_m=400,group_size=20,minimum_grid_center_separation_m=500,
        point_selection='Nearest 20 to cell center using true geography only; no prediction-based point filtering.',
        group_selection='Outcome-stratified illustrations: joint gains >=3 m/deg at 50th/75th score quantiles; tie within 1 m/deg; joint failure near median failure score.',
        ordering='Geometry-only nearest-neighbor order, starting at western-most point; no timestamps.',
        prediction_processing='Original maximum-pose predictions; no smoothing, fitting, filtering or correction.',
        inputs={name:dict(path=value,sha256=hashlib.sha256((ROOT/value).read_bytes()).hexdigest()) for name,value in INPUTS.items()},
        projection_origin=list(projection.latlonalt),cases=cases,
        limitation='These outcome-selected examples illustrate cases, not unbiased aggregate performance. Checkpoint was previously test-subset-selected.')
    (output/'selection_protocol.json').write_text(json.dumps(notes,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    return table,projection,cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'figures/balanced_spatial_comparison')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepare(args.output)


if __name__ == '__main__':
    main()

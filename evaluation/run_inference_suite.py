"""Sequential fixed-checkpoint intervention queue and matching-ID report."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from evaluation.adapted_geometry import summarize_results
from evaluation.inference_interventions import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'experiments/maplocnet__topveloc_extrap_m1p0/checkpoint-extrapolated.ckpt'
CONTROL = ROOT/'experiments/max_pose_comparison/current_sf_serial_v1'
BASELINE = ROOT/'experiments/adapted_baselines/sanfrancisco_v1/removed_baseline-inspired-aerial-cnn'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stride',type=int,default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error('invalid subset')
    args.output.mkdir(parents=True)
    expected = list(range(0,10000,args.stride))
    if args.limit is not None:
        expected = expected[:args.limit]
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    baseline = pd.read_csv(BASELINE/'per_sample.csv').set_index('index').loc[expected].reset_index()
    reports = {'VecMapLocNet_released':summarize_results(baseline)}
    rows = [dict(variant='VecMapLocNet_released',sample_count=len(baseline),**reports['VecMapLocNet_released']['metrics'])]
    completed = []
    for variant in VARIANTS:
        status = dict(status='running',completed=completed,pending=[v for v in VARIANTS if v not in completed],
            protocol='fixed-weight sensitivity, not retrained ablations',seed=0,
            unsupported={'parameter_matched_raster':'No trained matched checkpoint. Not fabricated or substituted.'})
        (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        target = args.output/variant
        if args.limit is None and args.stride == 1 and variant in ('none','k3'):
            control_config = json.loads((CONTROL/'config.json').read_text())
            if control_config['checkpoint_sha256'] != digest or control_config['batch_size'] != 1:
                raise RuntimeError('Saved control provenance differs')
            target.mkdir()
            shutil.copyfile(CONTROL/'per_sample.csv',target/'per_sample.csv')
            (target/'config.json').write_text(json.dumps(dict(intervention=variant,
                reused_control=str(CONTROL),checkpoint_sha256=digest,
                reason='Unmodified checkpoint renderer K=3; identical configuration'),indent=2)+'\n')
            report = summarize_results(pd.read_csv(target/'per_sample.csv'))
            (target/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        else:
            command = [sys.executable,'-u','-m','evaluation.run_max_pose','--checkpoint',str(CHECKPOINT),
                       '--intervention',variant,'--output',str(target),'--stride',str(args.stride)]
            if args.limit is not None:
                command += ['--limit',str(args.limit)]
            with open(args.output/f'{variant}.log','w') as log:
                try:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError:
                    status.update(status='failed',failed_variant=variant)
                    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                    raise
        frame = pd.read_csv(target/'per_sample.csv')
        if frame['index'].tolist() != expected:
            raise RuntimeError(f'Mismatching sample IDs: {variant}')
        if variant in ('none','k3'):
            original = pd.read_csv(CONTROL/'per_sample.csv').set_index('index').loc[expected]
            if not np.array_equal(frame[['pred_u','pred_v','pred_yaw']].to_numpy(),
                                   original[['pred_u','pred_v','pred_yaw']].to_numpy()):
                raise RuntimeError('No-op/K=3 failed to reproduce original max-pose predictions')
        report = summarize_results(frame)
        reports[variant] = report
        rows.append(dict(variant=variant,sample_count=len(frame),**report['metrics']))
        completed.append(variant)
        pd.DataFrame(rows).to_csv(args.output/'comparison.csv',index=False)
        print(f'Completed {variant}: {len(frame)} samples',flush=True)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint file changed during evaluation')
    (args.output/'suite_summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    status.update(status='complete',completed=completed,pending=[])
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()

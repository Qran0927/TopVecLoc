"""TopVecLoc encoder-only timings; no matching or pose decoding."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from eval_uav import SafeUavDataset, move_batch_to_device
from evaluation.benchmark_model_efficiency import gpu_status
from evaluation.plot_pose_heatmap_cases import ROOT, CHECKPOINTS
from module import GenericModule

OUTPUT=ROOT/'experiments/encoder_runtime_shared_gpu_v1'


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(0)
    before=gpu_status()
    cfg=OmegaConf.load(ROOT/'conf/maplocnet.yaml')
    ds=SafeUavDataset(Path(cfg.data.root),'SanFrancisco',
                      T.Compose([T.ToTensor(),T.Resize(cfg.data.image_size)]))
    assert ds.available_sample_indices()==list(range(10000))
    ids=list(range(0,10000,100))
    batches=[move_batch_to_device(batch,torch.device('cuda')) for batch in
             DataLoader(Subset(ds,ids),batch_size=1,shuffle=False,num_workers=0)]
    model=GenericModule.load_from_checkpoint(CHECKPOINTS['ours'],cfg=cfg,strict=True,
        find_best=False,map_location='cpu').eval().cuda().model
    def total(batch):
        return model.map_encoder(batch),model.image_encoder(batch)
    stages={'image_encoder':model.image_encoder,
            'cnn_map_encoder':model.map_encoder.cnn_encoder,
            'gnn_map_encoder':model.map_encoder.graph_encoder,
            'all_encoders':total}
    rows=[]
    with torch.no_grad():
        for name,stage in stages.items():
            for i in range(20):
                output=stage(batches[i])
                del output
            torch.cuda.synchronize()
            print('Warmup complete:',name,flush=True)
        names=list(stages)
        for i,batch in enumerate(batches):
            order=names[i%4:]+names[:i%4]
            for name in order:
                torch.cuda.synchronize()
                start=time.perf_counter()
                output=stages[name](batch)
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                del output
                rows.append(dict(stage=name,sample_id=ids[i],latency_ms=elapsed))
            if (i+1)%20==0:
                print(f'Measured {i+1}/100 samples per stage',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUTPUT/'timings.csv',index=False)
    records=[]
    for name in names:
        values=frame.loc[frame.stage.eq(name),'latency_ms'].to_numpy()
        assert len(values)==100 and np.isfinite(values).all() and (values>0).all()
        records.append(dict(stage=name,mean_ms=float(values.mean()),mean_s=float(values.mean()/1000),
                            median_ms=float(np.median(values)),std_ms=float(values.std(ddof=1)),
                            p95_ms=float(np.quantile(values,.95))))
    pd.DataFrame(records).to_csv(OUTPUT/'summary.csv',index=False)
    protocol=dict(gpu=torch.cuda.get_device_name(),batch_size=1,precision='FP32; no autocast',
        sample_ids=ids,warmup_calls_per_stage=20,measured_calls_per_stage=100,
        timing='Synchronized wall time around each actual encoder forward, rotating stage order.',
        scope='GNN encoder includes context aggregation, geographic encoding, GraphGPS and IDW dense rendering; '
              'all_encoders is independently timed actual map_encoder + image_encoder, including residual fusion.',
        excluded='FFT matching, probability normalization, pose decoding, file I/O, image transform, '
                 'initial image/map GPU transfer and graph construction. Model-internal graph transfer remains timed.',
        total='Not a sum of independently measured columns; run-to-run GPU contention means it may differ from their sum.',
        gpu_condition='Shared GPU with other active compute processes; not an isolated GPU benchmark.',
        status_before=before,status_after=gpu_status(),torch=torch.__version__)
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__': main()

"""Paired batch-one fixed-checkpoint benchmark with explicit shared-GPU conditions."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from eval_uav import SafeUavDataset, move_batch_to_device
from evaluation.adapted_matchers import load_aerial_cnn
from evaluation.plot_pose_heatmap_cases import CHECKPOINTS, ROOT
from module import GenericModule

OUTPUT=ROOT/'experiments/model_efficiency_shared_gpu_v1'


def gpu_status():
    commands=[['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,memory.used,power.draw','--format=csv'],
              ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']]
    return [subprocess.check_output(command,text=True).strip() for command in commands]


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(0)
    cfg=OmegaConf.load(ROOT/'conf/maplocnet.yaml')
    before=gpu_status()
    dataset=SafeUavDataset(Path(cfg.data.root),'SanFrancisco',
                           T.Compose([T.ToTensor(),T.Resize(cfg.data.image_size)]))
    assert dataset.available_sample_indices()==list(range(10000))
    ids=list(range(0,10000,100))
    cached=[]; graph_sizes=[]
    for batch in DataLoader(Subset(dataset,ids),batch_size=1,shuffle=False,num_workers=0):
        graph=batch['graph']
        graph_sizes.append(dict(sample_id=int(batch['sample_index'][0]),
            road_nodes=int(graph['intersection']['num_nodes'][0]),
            road_directed_edges=int(graph['road_edges']['edge_index'].shape[-1]),
            poi_nodes=int(graph['poi']['num_nodes'][0]),area_nodes=int(graph['area']['num_nodes'][0])))
        # Match the established evaluator: RGB/map tensors transferred before timing;
        # the model's internal graph preparation and graph transfer remain timed.
        cached.append(move_batch_to_device(batch,torch.device('cuda')))
    models={'baseline':load_aerial_cnn(CHECKPOINTS['baseline'],'cuda'),
            'ours':GenericModule.load_from_checkpoint(CHECKPOINTS['ours'],cfg=cfg,strict=True,
                find_best=False,map_location='cpu').eval().cuda()}
    counts={method:sum(p.numel() for p in model.model.parameters()) for method,model in models.items()}
    print('Parameter counts:',counts,flush=True)
    rows=[]
    with torch.no_grad():
        for method,model in models.items():
            for i in range(20):
                prediction=model(cached[i])
                del prediction
            torch.cuda.synchronize()
            print('Warmup complete:',method,flush=True)
        snapshots=[]
        for i,batch in enumerate(cached):
            order=['baseline','ours'] if i%2==0 else ['ours','baseline']
            for method in order:
                torch.cuda.synchronize()
                start=time.perf_counter()
                prediction=models[method](batch)
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                assert torch.isfinite(prediction['uv_max']).all()
                rows.append(dict(method=method,sample_id=ids[i],order=order.index(method),latency_ms=elapsed))
                del prediction
            if (i+1)%20==0:
                snapshots.append(dict(completed_samples=i+1,gpu_status=gpu_status()))
                print(f'Measured {i+1}/100 paired samples',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUTPUT/'timings.csv',index=False)
    pd.DataFrame(graph_sizes).to_csv(OUTPUT/'graph_sizes.csv',index=False)
    summaries=[]
    for method in models:
        latency=frame.loc[frame.method.eq(method),'latency_ms'].to_numpy()
        assert len(latency)==100 and np.isfinite(latency).all() and (latency>0).all()
        summaries.append(dict(method=method,parameters=counts[method],parameters_m=counts[method]/1e6,
            mean_latency_ms=float(latency.mean()),std_latency_ms=float(latency.std(ddof=1)),
            median_latency_ms=float(np.median(latency)),p95_latency_ms=float(np.quantile(latency,.95)),
            throughput_hz=float(1000/latency.mean()),measured_calls=len(latency)))
    pd.DataFrame(summaries).to_csv(OUTPUT/'comparison.csv',index=False)
    protocol=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        batch_size=1,precision='FP32 weights/inputs; no autocast',
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        image_shape=[1,3,256,256],map_shape=[1,3,256,256],
        warmup_calls_per_model=20,measured_calls_per_model=100,sample_ids=ids,
        timing='Synchronized perf_counter around model forward; includes model-internal graph preparation, '
               'IDW rendering, CNN/GNN encoding, matching and both maximum/expectation pose outputs.',
        excluded='File reads, image transform, initial RGB/map host-to-device transfers, model loading and OSM graph construction.',
        throughput='1000 / arithmetic mean single-frame latency in ms; batch=1 sequential throughput, not batched peak FPS.',
        order='Alternating model order on each identical sample to reduce temporal ordering bias.',
        parameters='Unique registered model parameters counted with model.model.parameters(); buffers excluded.',
        gpu_condition='Shared GPU: another training process was already active; these are contention-affected timings, not isolated hardware performance.',
        status_before=before,status_samples=snapshots,status_after=gpu_status(),
        checkpoints={m:{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for m,p in CHECKPOINTS.items()})
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':
    main()

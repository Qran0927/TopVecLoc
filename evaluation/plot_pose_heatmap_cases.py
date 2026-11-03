"""Measured UAV/map examples: shared-scale pose probability maps, no synthetic heatmaps."""
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from eval_uav import SafeUavDataset, move_batch_to_device
from evaluation.adapted_matchers import load_aerial_cnn
from evaluation.balanced_spatial_comparison import ROOT, INPUTS, load_table
from module import GenericModule
from osm.viz import Colormap

OUTPUT = ROOT/'figures/pose_heatmap_success_cases'
CHECKPOINTS = {
    'baseline':ROOT/'experiments/last-step-checkpointing-v1.ckpt',
    'ours':ROOT/'experiments/maplocnet__topveloc_extrap_m1p0/checkpoint-extrapolated.ckpt'}


def choose_examples(table):
    pool = table[(table.ours_position_error_m<3)&(table.ours_yaw_error_deg<3)&
                 (table.baseline_position_error_m>10)&(table.baseline_yaw_error_deg>5)].copy()
    pool['gain'] = pool.baseline_position_error_m-pool.ours_position_error_m
    selected = []
    for quantile in [.25,.5,.75,.9]:
        target = pool.gain.quantile(quantile)
        order = pool.assign(distance=(pool.gain-target).abs()).sort_values(['distance'],kind='stable')
        for index,row in order.iterrows():
            if all(np.hypot(row.east_m-table.loc[i,'east_m'],
                            row.north_m-table.loc[i,'north_m'])>=600 for i in selected):
                selected.append(int(index))
                break
        else:
            raise RuntimeError('Not enough geographically separated improvement examples')
    return selected, len(pool)


def infer(ids, cfg):
    dataset = SafeUavDataset(Path(cfg.data.root),'SanFrancisco',
                            T.Compose([T.ToTensor(),T.Resize(cfg.data.image_size)]))
    assert dataset.available_sample_indices()==list(range(10000))
    results = {}
    for method, checkpoint in CHECKPOINTS.items():
        if method=='baseline':
            model = load_aerial_cnn(checkpoint,'cuda')
        else:
            model = GenericModule.load_from_checkpoint(checkpoint,cfg=cfg,strict=True,
                find_best=False,map_location='cpu').eval().cuda()
        loader = DataLoader(Subset(dataset,ids),batch_size=1,shuffle=False,num_workers=0)
        with torch.no_grad():
            for batch in loader:
                index = int(batch['sample_index'][0])
                prediction = model(move_batch_to_device(batch,torch.device('cuda')))
                # Maximum over heading preserves the joint MAP location as the heatmap peak.
                probability = prediction['log_probs'][0].exp().amax(dim=-1).cpu().numpy()
                uv = prediction['uv_max'][0].cpu().numpy()
                yaw = float(prediction['yaw_max'][0])
                gt = batch['uv'][0].numpy()
                gt_yaw = float(batch['roll_pitch_yaw'][0,-1])
                item = dict(probability=probability,uv=uv,yaw=yaw,gt=gt,gt_yaw=gt_yaw,
                    position_error_m=float(np.linalg.norm(uv-gt)/float(batch['pixels_per_meter'][0])),
                    yaw_error_deg=abs((yaw-gt_yaw+180)%360-180))
                assert np.isfinite(probability).all() and probability.max()>0
                results[index,method] = item
                np.savez_compressed(OUTPUT/f'{index}_{method}.npz',**item)
                print(index,method,'position',item['position_error_m'],'yaw',item['yaw_error_deg'],flush=True)
        del model
        torch.cuda.empty_cache()
    return dataset, results


def draw_pose(ax,uv,yaw,color,marker):
    ax.scatter(*uv,s=22,marker=marker,c=color,edgecolors='white',linewidths=.6,zorder=7)
    angle=np.deg2rad(yaw)
    ax.arrow(uv[0],uv[1],16*np.sin(angle),-16*np.cos(angle),
             width=.9,head_width=5,head_length=5,color=color,
             ec='white',lw=.55,zorder=6,length_includes_head=True)


def plot(ids,dataset,results):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':13,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    # Set physical size to a manuscript two-column width so text remains readable
    # when the exported PDF is placed at its intended size (about 183 mm).
    fig,axes=plt.subplots(4,4,figsize=(7.2,7.5))
    fig.subplots_adjust(left=.035,right=.915,bottom=.035,top=.91,wspace=.04,hspace=.08)
    maximum=max(item['probability'].max() for item in results.values())
    norm=LogNorm(vmin=maximum*1e-5,vmax=maximum)
    metadata=pd.read_csv(ROOT/INPUTS['metadata']).set_index('id')
    rows=[]
    for row,index in enumerate(ids):
        sample=dataset[index]
        uav=sample['image'].permute(1,2,0).numpy()
        raster=sample['map'].numpy()
        rgb=Colormap.apply(raster)
        nodes=np.column_stack(np.where(raster[2]>0))
        for column in range(4):
            ax=axes[row,column]
            ax.imshow(uav if column==0 else rgb,interpolation='nearest')
            if column>0 and len(nodes):
                ax.scatter(nodes[:,1],nodes[:,0],s=.35,c='#444444',alpha=.7)
            if column>=2:
                method='baseline' if column==2 else 'ours'
                item=results[index,method]
                heat=ax.imshow(item['probability'],cmap='turbo',norm=norm,alpha=.65,
                               interpolation='nearest')
                draw_pose(ax,item['gt'],item['gt_yaw'],'#151515','*')
                draw_pose(ax,item['uv'],item['yaw'],'#E1364B' if method=='baseline' else '#00A0AE','o')
                error_text=(rf"$Pos_{{error}} = {float(item['position_error_m']):.2f}\,\mathrm{{m}}$"+'\n'+
                            rf"$Yaw_{{error}} = {float(item['yaw_error_deg']):.2f}^\circ$")
                ax.text(.025,.975,error_text,
                        transform=ax.transAxes,va='top',fontsize=9.5,linespacing=1.25,
                        color='black',
                        bbox=dict(facecolor='white',edgecolor='#9CA3AA',linewidth=.5,
                                  alpha=1,boxstyle='square,pad=.24'))
                rows.append(dict(sample_id=index,method=method,uav_file=metadata.loc[index,'uav_name'],
                                 pred_u=float(item['uv'][0]),pred_v=float(item['uv'][1]),
                                 pred_yaw=item['yaw'],gt_u=float(item['gt'][0]),gt_v=float(item['gt'][1]),
                                 gt_yaw=item['gt_yaw'],position_error_m=item['position_error_m'],
                                 yaw_error_deg=item['yaw_error_deg']))
            if column==1:
                item=results[index,'ours']
                draw_pose(ax,item['gt'],item['gt_yaw'],'#151515','*')
            ax.set_xlim(-.5,255.5); ax.set_ylim(255.5,-.5)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#C9CDD0'); spine.set_linewidth(.7)
    for ax,title in zip(axes[0],['UAV image','Input map','VecMapLocNet','TopVecLoc']):
        ax.set_title(title,fontsize=11,fontweight='bold',pad=6)
    handles=[Line2D([],[],marker='*',color='#151515',ls='None',markersize=8,label='Ground-truth pose'),
             Line2D([],[],marker='o',color='#E1364B',ls='None',markersize=5,label='VecMapLocNet pose'),
             Line2D([],[],marker='o',color='#00A0AE',ls='None',markersize=5,label='TopVecLoc pose')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.49,.985),ncol=3,
               frameon=True,edgecolor='#D0D0D0',fontsize=9,columnspacing=1.0,handletextpad=.35)
    cax=fig.add_axes([.927,.20,.010,.55])
    colorbar=fig.colorbar(heat,cax=cax)
    fig.text(.995,.475,'Pose probability (shared log scale)',fontsize=7.5,
             rotation=90,ha='right',va='center')
    colorbar.ax.tick_params(labelsize=8,width=.5,length=2)
    stem=OUTPUT/'pose_heatmap_success_cases'
    for extension in ['png','tiff','pdf','svg']:
        kwargs={'dpi':600} if extension in ('png','tiff') else {}
        if extension=='tiff': kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+extension),**kwargs)
    fig.savefig(OUTPUT/'preview.png',dpi=220)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUTPUT/'actual_predictions.csv',index=False)
    return {'normalization':'One shared LogNorm for all 8 heatmaps; no per-panel rescaling',
            'vmin':float(norm.vmin),'vmax':float(norm.vmax),'alpha':.65,
            'heatmap':'max over yaw of globally normalized joint spatial-heading posterior',
            'interpolation':'nearest; no smoothing','dpi':600}


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(0)
    table,_=load_table()
    ids,pool_size=choose_examples(table)
    print('Selected IDs:',ids,'eligible improvements:',pool_size,flush=True)
    OUTPUT.mkdir(parents=True)
    table.loc[ids].to_csv(OUTPUT/'selection_source_metrics.csv')
    cfg=OmegaConf.load(ROOT/'conf/maplocnet.yaml')
    dataset,results=infer(ids,cfg)
    # Verify the claimed improvement with the actual newly inferred poses as well.
    for index in ids:
        assert results[index,'ours']['position_error_m']<3
        assert results[index,'ours']['yaw_error_deg']<3
        assert results[index,'baseline']['position_error_m']>10
        assert results[index,'baseline']['yaw_error_deg']>5
    plot_info=plot(ids,dataset,results)
    protocol=dict(sample_ids=ids,eligible_pool=pool_size,selection=
        'Successful improvement cases only: ours <3m/<3deg, baseline >10m/>5deg; '
        'position-gain quantiles 25/50/75/90%, at least 600m apart. Not representative random sampling.',
        checkpoints={k:{'path':str(v.relative_to(ROOT)),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()}
                     for k,v in CHECKPOINTS.items()},plot=plot_info,
        input_map='Actual 3-channel semantic raster: area/way colors and nonzero POI pixels as dark dots.',
        inference='Existing fixed checkpoints, original single-image maximum-pose inference; no training.',
        caveat='Current TopVecLoc checkpoint was previously selected using test-subset results.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(OUTPUT/'preview.png',flush=True)


if __name__=='__main__':
    main()

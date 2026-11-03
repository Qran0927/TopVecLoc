"""Reference-style encoder runtime table, using actual measured seconds."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.benchmark_encoder_runtime import OUTPUT as SOURCE, ROOT

OUTPUT=ROOT/'figures/topvecloc_encoder_runtime_parameters_final'


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    data=pd.read_csv(SOURCE/'summary.csv').set_index('stage')
    counts=json.loads((SOURCE/'parameter_counts.json').read_text())['counts']
    for key in data.index:
        data.loc[key,'parameters']=counts[key]
        data.loc[key,'parameters_m']=counts[key]/1e6
        data.loc[key,'fp32_weights_mb']=4*counts[key]/1e6
    OUTPUT.mkdir(parents=True)
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':15,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(12,4.4))
    fig.text(.5,.925,'TABLE V',ha='center',fontsize=16)
    fig.text(.5,.835,'ENCODER RUNTIME AND PARAMETER SIZE OF TOPVECLOC',ha='center',fontsize=17)
    ax=fig.add_axes([.035,.27,.93,.49]); ax.set_axis_off()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    borders=[0,.20,.40,.60,.80,1]
    labels=['','Image encoder\nResNet-50 + FPN','CNN map encoder\nVGG-19 + U-Net',
            'GNN map encoder\nGraphGPS + IDW','All encoders']
    keys=['image_encoder','cnn_map_encoder','gnn_map_encoder','all_encoders']
    for index,label in enumerate(labels):
        center=(borders[index]+borders[index+1])/2
        ax.text(center,.84,label,ha='center',va='center',fontsize=14,fontweight='bold',linespacing=1.3)
    for y,label,field in [(.56,'Runtime (s)','mean_s'),(.33,'Parameters (M)','parameters_m'),
                           (.11,'Weights (MB)','fp32_weights_mb')]:
        ax.text(.1,y,label,ha='center',va='center',fontsize=15,fontweight='bold')
        for index,key in enumerate(keys,1):
            center=(borders[index]+borders[index+1])/2
            ax.text(center,y,f'{data.loc[key,field]:.4f}',ha='center',va='center',fontsize=16)
    for y,width in [(1,1.2),(.68,.8),(.445,.6),(.22,.6),(0,1.2)]:
        ax.plot([0,1],[y,y],color='#303030',lw=width,clip_on=False)
    for x in borders[1:-1]:
        ax.plot([x,x],[0,1],color='#505050',lw=.8)
    fig.text(.035,.19,'Batch size = 1, FP32; 20 warmup calls and 100 measured frames per encoder.',fontsize=10.5)
    fig.text(.035,.125,'Shared GPU: RTX PRO 6000 Blackwell. All encoders includes fusion; matching and pose decoding excluded.',fontsize=10.5)
    fig.text(.035,.06,'M = 1,000,000 parameters; MB = FP32 parameter bytes / 1,000,000. Total includes 80 fusion parameters; buffers excluded.',fontsize=10.5)
    stem=OUTPUT/'encoder_runtime'
    for ext in ['png','tiff','pdf','svg']:
        opts={'dpi':600} if ext in ['png','tiff'] else {}
        if ext=='tiff': opts['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+ext),**opts)
    fig.savefig(OUTPUT/'preview.png',dpi=140)
    plt.close(fig)
    data.to_csv(OUTPUT/'table_data.csv')
    (OUTPUT/'caption_zh.txt').write_text(
        ' V　TopVecLoc。（M），FP324（MB=10⁶），'
        '、，。80。'
        '、CNNGNN，'
        '，FFT。GNN、'
        '、GraphGPSIDW。batch size1，FP32，20100。'
        'GPU，GPU；，。\n')
    print(OUTPUT/'preview.png',flush=True)


if __name__=='__main__': main()

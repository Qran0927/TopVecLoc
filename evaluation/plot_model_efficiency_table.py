"""Three-line table using measured parameter counts and paired GPU timings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

from evaluation.benchmark_model_efficiency import OUTPUT as SOURCE, ROOT

OUTPUT=ROOT/'figures/model_efficiency_shared_gpu'


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    data=pd.read_csv(SOURCE/'comparison.csv').set_index('method')
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    OUTPUT.mkdir(parents=True)
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':15,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    chinese=FontProperties(family='Noto Serif CJK SC')
    fig=plt.figure(figsize=(11,3.4))
    ax=fig.add_axes([.035,.26,.93,.60]); ax.set_axis_off()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.text(.5,.925,' VIII　',fontproperties=chinese,
             fontsize=16,ha='center',va='center')
    xs=[.04,.47,.70,.91]
    headers=['','（M）','（ms）','（Hz）']
    for x,header in zip(xs,headers):
        ax.text(x,.78,header,fontproperties=chinese,fontsize=14,
                ha='left' if x==xs[0] else 'center',va='center')
    for y,method,label in [(.47,'baseline','VecMapLocNet'),(.17,'ours','TopVecLoc')]:
        row=data.loc[method]
        text=[label,f'{row.parameters_m:.3f}',f'{row.mean_latency_ms:.2f}',f'{row.throughput_hz:.2f}']
        for column,(x,value) in enumerate(zip(xs,text)):
            ax.text(x,y,value,fontsize=16,ha='left' if column==0 else 'center',va='center')
    for y,width in [(.98,1.2),(.63,.8),(.015,1.2)]:
        ax.plot([0,1],[y,y],color='#202020',lw=width,clip_on=False)
    notes=['：NVIDIA RTX PRO 6000 Blackwell；batch size = 1；FP32；20，100。',
           '，GPU；、。']
    for y,text in zip([.17,.075],notes):
        fig.text(.035,y,text,fontproperties=chinese,fontsize=10.5,va='center')
    stem=OUTPUT/'model_complexity_efficiency'
    for ext in ['png','tiff','pdf','svg']:
        opts={'dpi':600} if ext in ('png','tiff') else {}
        if ext=='tiff': opts['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+ext),**opts)
    fig.savefig(OUTPUT/'preview.png',dpi=150)
    plt.close(fig)
    data.to_csv(OUTPUT/'table_data.csv')
    b=data.loc['baseline']; t=data.loc['ours']
    description=(f"TopVecLoc{int(t.parameters):,}，VecMapLocNet{int(b.parameters):,}；"
        f"{int(t.parameters-b.parameters):,}（{(t.parameters/b.parameters-1)*100:.2f}%）。"
        f"GPU，{b.mean_latency_ms:.2f}{t.mean_latency_ms:.2f} ms，"
        f"{b.throughput_hz:.2f}{t.throughput_hz:.2f} Hz。"
        "，GPUJetson。\n")
    (OUTPUT/'measurement_notes_zh.txt').write_text(description)
    print(description,flush=True)


if __name__=='__main__':
    main()

"""Boxplots of measured per-image maximum-pose errors, never synthetic quartiles."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from evaluation.plot_recall_comparison import ROOT

INTERVENTIONS = ('none','no_attention','no_pe','no_context','no_edges','random_edges')
METRICS = [('position_max_error_m','Localization error (m)'),
           ('yaw_error_deg','Yaw error (°)'),
           ('longitudinal_max_error_m','Longitudinal error (m)'),
           ('lateral_max_error_m','Lateral error (m)')]
RED, CYAN = '#EC8B7B','#66B6B5'


def load_samples(settings):
    samples = {}
    for setting in settings:
        path = (ROOT/'experiments/adapted_baselines/sanfrancisco_v1/removed_baseline-inspired-aerial-cnn/per_sample.csv'
                if setting=='VecMapLocNet' else
                ROOT/f'experiments/inference_interventions/full_v1/{setting}/per_sample.csv')
        frame = pd.read_csv(path)
        if frame['index'].tolist()!=list(range(10000)):
            raise ValueError(f'Incomplete or mismatched sample IDs: {setting}')
        errors = frame[[key for key,_ in METRICS]].to_numpy(float)
        if not np.isfinite(errors).all() or (errors<0).any():
            raise ValueError('Invalid errors; do not silently drop samples')
        samples[setting] = frame
    return samples


def describe_box(values):
    values = np.asarray(values,dtype=float)
    stats = boxplot_stats(values,whis=1.5)[0]
    return dict(n=len(values),mean=float(values.mean()),median=float(stats['med']),
        q1=float(stats['q1']),q3=float(stats['q3']),whisker_low=float(stats['whislo']),
        whisker_high=float(stats['whishi']),outlier_count=len(stats['fliers']),
        minimum=float(values.min()),maximum=float(values.max()))


def make_figure(samples,mode,show_outliers=False):
    intervention = mode=='interventions'
    metrics = METRICS if intervention else METRICS[:2]
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':12,
        'axes.linewidth':.85,'axes.edgecolor':'#33383C','axes.labelcolor':'#26323A',
        'xtick.color':'#33383C','ytick.color':'#33383C',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes = plt.subplots(1,len(metrics),figsize=(17,5.0) if intervention else (8.8,4.7))
    fig.subplots_adjust(left=.065 if intervention else .095,right=.985,
                        bottom=.23,top=.77 if intervention else .80,wspace=.34)
    labels = ['Full','w/o\nAttn','w/o\nPE','w/o\nContext','w/o\nEdges','Rewired'] if intervention else ['VecMapLocNet','TopVecLoc (ours)']
    colors = [CYAN]+[RED]*5 if intervention else [RED,CYAN]
    for i,(ax,(key,ylabel)) in enumerate(zip(axes,metrics)):
        stats = [boxplot_stats(frame[key].to_numpy(float),whis=1.5)[0] for frame in samples.values()]
        artists = ax.bxp(stats,positions=np.arange(1,len(stats)+1),widths=.55,
            showfliers=show_outliers,patch_artist=True,
            medianprops={'color':'#161B1F','linewidth':1.8},
            boxprops={'edgecolor':'#30383D','linewidth':1.05},
            whiskerprops={'color':'#30383D','linewidth':1.1},
            capprops={'color':'#30383D','linewidth':1.1},
            flierprops={'marker':'.','markersize':1.2,'markeredgewidth':0,
                        'markerfacecolor':'#52636D','alpha':.18})
        for box,color in zip(artists['boxes'],colors):
            box.set_facecolor(color)
        upper = max(float(frame[key].max()) for frame in samples.values()) if show_outliers else max(s['whishi'] for s in stats)
        ax.set_ylim(0,upper*1.10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=4))
        ax.set_xticks(range(1,len(labels)+1),labels,fontsize=10.5 if intervention else 12)
        ax.set_ylabel(ylabel,fontsize=14,labelpad=5)
        ax.grid(axis='y',color='#E5E8EA',linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(direction='in',top=True,right=True,length=3,width=.7,pad=5)
        ax.text(.5,-.24,f'({chr(97+i)})',ha='center',va='top',transform=ax.transAxes,fontsize=14)
    title = 'Fixed-weight inference interventions' if intervention else 'VecMapLocNet vs TopVecLoc'
    fig.suptitle(title+(' — full error range' if show_outliers else ''),y=.99,fontsize=17)
    if intervention:
        fig.legend(handles=[Patch(facecolor=CYAN,edgecolor='#30383D',label='Full TopVecLoc'),
                            Patch(facecolor=RED,edgecolor='#30383D',label='Inference interventions')],
            loc='upper center',bbox_to_anchor=(.53,.93),ncol=2,frameon=False,fontsize=12)
    fig.text(.53,.85 if intervention else .88,
        'All 10,000 images retained · '+('Outliers shown' if show_outliers else 'Outlier markers omitted'),
        ha='center',va='center',fontsize=10.5,color='#62717A')
    fig.text(.53,.025,'Box: 25th–75th percentiles; black line: median; whiskers: 1.5 × IQR.',
             ha='center',va='bottom',fontsize=10.5,color='#62717A')
    return fig,axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'figures/error_boxplots')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summaries = []
    for mode,settings in [('interventions',INTERVENTIONS),('baseline',('VecMapLocNet','none'))]:
        samples = load_samples(settings)
        metrics = METRICS if mode=='interventions' else METRICS[:2]
        columns = ['index']+[key for key,_ in METRICS]
        pd.concat([frame[columns].assign(setting=name) for name,frame in samples.items()],
            ignore_index=True).to_csv(args.output/f'{mode}_source_samples.csv',index=False)
        for setting,frame in samples.items():
            for key,_ in metrics:
                summaries.append(dict(mode=mode,setting=setting,metric=key,**describe_box(frame[key])))
        for full_range in [False,True]:
            fig,_ = make_figure(samples,mode,show_outliers=full_range)
            stem = args.output/(mode+'_boxplots'+('_full_range' if full_range else ''))
            for ext in ['png','tiff','pdf','svg']:
                options = {'dpi':600} if ext in ('png','tiff') else {}
                if ext=='tiff':
                    options['pil_kwargs'] = {'compression':'tiff_lzw'}
                fig.savefig(stem.with_suffix('.'+ext),**options)
            fig.savefig(args.output/(stem.name+'_preview.png'),dpi=150)
            plt.close(fig)
    pd.DataFrame(summaries).to_csv(args.output/'box_statistics.csv',index=False)
    notes = dict(backend='Python/matplotlib',dpi=600,n_per_setting=10000,
        estimator='Maximum-probability pose; original per-image errors, not recalled percentages.',
        statistics='Q1, median, Q3; standard 1.5 IQR whiskers; all samples used including extreme errors.',
        overview='Compact view hides only outlier markers and scales axes to whiskers; not a full-range plot.',
        transparency='Full-range counterparts show every outlier and use the complete error range.',
        uncertainty='Distributions across images; not uncertainty across retraining runs or confidence intervals.',
        inference_only=True,limitations='Test-subset-selected checkpoint. No retrained component ablation or significance claim.')
    (args.output/'figure_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()

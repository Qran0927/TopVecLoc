"""Signed percentage-point changes for fixed-weight inference interventions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from evaluation.plot_intervention_recalls import VARIANTS, read_data
from evaluation.plot_recall_comparison import ROOT, PANELS, THRESHOLDS, read_source

ROW_LABELS = ['w/o global attention','w/o geographic PE',
              'w/o POI/region context','w/o road edges','Random road rewiring']
GROUPS = {
    'components': (VARIANTS[1:], ROW_LABELS),
    'idw_origin': (('k1','k3','k5','k8','shift_origin'),
                   ['IDW K = 1','IDW K = 3 (reference)','IDW K = 5','IDW K = 8','Shifted PE origin']),
}


def load_group(path, group):
    variants,_ = GROUPS[group]
    return read_source(path,('none',*variants))


def calculate_deltas(data, variants=VARIANTS[1:]):
    cols = [f'{family}_recall_{t}{unit}' for family,unit,_ in PANELS for t in THRESHOLDS]
    indexed = data.set_index('variant')
    # Inputs are already percentage recalls. Differences are percentage points,
    # not relative percentage changes, and must NOT be multiplied by 100 again.
    return indexed.loc[list(variants),cols].subtract(indexed.loc['none',cols],axis='columns')


def make_figure(delta, row_labels=None):
    row_labels = ROW_LABELS if row_labels is None else row_labels
    if len(row_labels) != len(delta):
        raise ValueError('Every intervention must have its own row label')
    nrows = len(delta)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
        'text.color':'#27323A','axes.labelcolor':'#27323A',
        'xtick.color':'#27323A','ytick.color':'#27323A',
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    fig,axes = plt.subplots(1,4,figsize=(16,4.2))
    fig.subplots_adjust(left=.185,right=.905,top=.84,bottom=.23,wspace=.085)
    bound = max(.5,np.ceil(np.abs(delta.to_numpy(float)).max()*2)/2)
    norm = TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    cmap = LinearSegmentedColormap.from_list('signed_recall',
        ['#D1827A','#F8FAFB','#54B6B9'])
    for j,(ax,(family,unit,title)) in enumerate(zip(axes,PANELS)):
        values = delta[[f'{family}_recall_{t}{unit}' for t in THRESHOLDS]].to_numpy(float)
        image = ax.imshow(values,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
        ax.set_title(f'({chr(97+j)}) {title.split(" (")[0]}',fontsize=12,pad=13)
        ax.set_xticks(range(4),[f'R@{t}' for t in THRESHOLDS])
        ax.set_yticks(range(nrows),row_labels if j==0 else ['']*nrows)
        ax.tick_params(axis='both',length=0,pad=9)
        ax.set_xticks(np.arange(-.5,4,1),minor=True)
        ax.set_yticks(np.arange(-.5,nrows,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.2)
        ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(nrows):
            for col in range(4):
                value = round(float(values[row,col]),2)
                label = '0.00' if value==0 else f'{value:+.2f}'.replace('-','−')
                ax.text(col,row,label,ha='center',va='center',fontsize=10.5,
                        color='#253139')
    cax = fig.add_axes([.93,.23,.011,.61])
    bar = fig.colorbar(image,cax=cax,orientation='vertical',
                      ticks=[-bound,-bound/2,0,bound/2,bound])
    bar.outline.set_edgecolor('#C9CED1')
    bar.outline.set_linewidth(.6)
    bar.ax.tick_params(length=2,width=.6,labelsize=9)
    bar.set_label('Δ Recall (pp)',fontsize=10,labelpad=8)
    return fig,axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'experiments/inference_interventions/full_v1/comparison.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'figures/intervention_delta_heatmaps_1x4')
    parser.add_argument('--group',choices=GROUPS,default='components')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    variants,row_labels = GROUPS[args.group]
    data = load_group(args.source,args.group)
    delta = calculate_deltas(data,variants)
    fig,axes = make_figure(delta,row_labels)
    args.output.mkdir(parents=True)
    data.to_csv(args.output/'absolute_recalls.csv',index=False)
    delta.to_csv(args.output/'delta_percentage_points.csv')
    stem = args.output/('topvecloc_idw_origin_delta_heatmaps' if args.group=='idw_origin'
                       else 'topvecloc_intervention_delta_heatmaps')
    for ext in ['png','tiff','pdf','svg']:
        options = {'dpi':600} if ext in ('png','tiff') else {}
        if ext=='tiff':
            options['pil_kwargs'] = {'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+ext),**options)
    fig.savefig(args.output/'preview.png',dpi=150)
    norm = axes[0].images[0].norm
    notes = dict(backend='Python/matplotlib',dimensions_inches=[16,4.2],dpi=600,
        group=args.group,settings=list(variants),reference='Full TopVecLoc with IDW K=3; k3 is the same reused control',
        origin_shift='Uniform per-axis offsets in [-128,128] meters applied to PE inputs only; seed 0',
        layout='Minimal four-panel grid; single-line row labels; single shared vertical colorbar; exact cell values',
        definition='delta = intervention recall (%) minus full-model recall (%); units: percentage points',
        shared_color_limits=[float(norm.vmin),float(norm.vmax)],center=0,
        colors='red: lower recall; cyan: higher recall; shared symmetric scale for all panels',
        n_per_setting=10000,scope='Fixed-weight sensitivity, not retrained ablations.',
        uncertainty='Descriptive point differences, not statistical significance; no uncertainty estimated.',
        limitations='Previously test-subset-selected checkpoint; no causal contribution claim from this figure alone.')
    (args.output/'figure_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    plt.close(fig)
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()

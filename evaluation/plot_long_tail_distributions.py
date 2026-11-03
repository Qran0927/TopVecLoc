"""Full-sample boxplots and boundary-reflected KDEs for localization error tails."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from evaluation.balanced_spatial_comparison import ROOT, INPUTS
from evaluation.plot_error_boxplots import describe_box

OUTPUT=ROOT/'figures/long_tail_distributions_kde_labels_below'
NAMES={'baseline':'VecMapLocNet','ours':'TopVecLoc (ours)'}
COLORS={'baseline':'#E66B61','ours':'#159FAD'}
METRICS=[('position_max_error_m','Localization error (m)',280),
         ('yaw_error_deg','Yaw error (°)',180)]


def load_data():
    frames={}
    for method in NAMES:
        frame=pd.read_csv(ROOT/INPUTS[method]).sort_values('index')
        assert frame['index'].tolist()==list(range(10000))
        assert frame.status.eq('ok').all()
        for key,_,_ in METRICS:
            assert np.isfinite(frame[key]).all() and frame[key].ge(0).all()
        frames[method]=frame
    return frames


def pooled_bandwidth(arrays):
    pooled=np.concatenate(arrays)
    return float(pooled.std(ddof=1)*len(pooled)**(-.2))


def reflected_density(values,grid,bandwidth,upper=None):
    kde=gaussian_kde(values,bw_method=bandwidth/np.std(values,ddof=1))
    density=kde(grid)+kde(-grid)
    if upper is not None:
        density+=kde(2*upper-grid)
    return density


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    samples=load_data()
    OUTPUT.mkdir(parents=True)
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':13,
        'axes.linewidth':.9,'axes.edgecolor':'#30343B',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    box_fig,box_axes=plt.subplots(1,2,figsize=(13.2,5.6))
    kde_fig,kde_axes=plt.subplots(1,2,figsize=(13.2,5.6))
    axes=np.vstack([box_axes,kde_axes])
    for fig in [box_fig,kde_fig]:
        fig.subplots_adjust(left=.085,right=.985,bottom=.15,top=.80,wspace=.27)
    box_fig.subplots_adjust(bottom=.20)
    kde_fig.subplots_adjust(bottom=.25)
    summaries=[]; densities=[]; kde_notes={}
    for column,(key,label,upper) in enumerate(METRICS):
        box=axes[0,column]; density_ax=axes[1,column]
        arrays=[frame[key].to_numpy(float) for frame in samples.values()]
        bp=box.boxplot(arrays,positions=[1,2],widths=.48,patch_artist=True,
            showfliers=True,showmeans=True,
            medianprops=dict(color='#20252A',linewidth=2),
            meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor='#20252A',markersize=6),
            whiskerprops=dict(color='#30343B',linewidth=1.2),
            capprops=dict(color='#30343B',linewidth=1.2),
            flierprops=dict(marker='.',markersize=1.5,markeredgewidth=0,alpha=.16))
        for index,method in enumerate(NAMES):
            bp['boxes'][index].set(facecolor=COLORS[method],alpha=.72,edgecolor='#30343B')
            bp['fliers'][index].set(markerfacecolor=COLORS[method],
                                     markeredgecolor=COLORS[method])
            summaries.append(dict(method=NAMES[method],metric=key,
                                  **describe_box(arrays[index]),
                                  p90=float(np.quantile(arrays[index],.9)),
                                  p95=float(np.quantile(arrays[index],.95)),
                                  p99=float(np.quantile(arrays[index],.99)),
                                  above_50=int((arrays[index]>50).sum()),
                                  above_100=int((arrays[index]>100).sum())))
        box.set_yscale('symlog',linthresh=1,linscale=1)
        ticks=[0,1,3,5,10,30,50,100]+([180] if column else [200,280])
        box.yaxis.set_major_locator(FixedLocator(ticks))
        box.yaxis.set_major_formatter(ScalarFormatter())
        box.minorticks_off()
        box.set_ylim(0,upper*1.07)
        box.set_xticks([1,2],list(NAMES.values()),fontsize=13)
        box.set_ylabel(label,fontsize=15)
        box.text(.5,-.16,f'({chr(97+column)})',transform=box.transAxes,
                 ha='center',va='top',fontsize=16)
        bandwidth=pooled_bandwidth(arrays)
        grid=np.linspace(0,upper,2801)
        kde_notes[key]={'shared_absolute_bandwidth':bandwidth,
                        'boundary_reflection':[0,180] if column else [0]}
        for method,values in zip(NAMES,arrays):
            density=reflected_density(values,grid,bandwidth,180 if column else None)
            density_ax.plot(grid,density,color=COLORS[method],lw=2.1,label=NAMES[method])
            densities.append(pd.DataFrame({'metric':key,'method':NAMES[method],
                                            'error':grid,'density':density}))
            # Full probability mass must remain within the physical domain.
            integral=float(np.trapezoid(density,grid))
            assert abs(integral-1)<.002,(key,method,integral)
            kde_notes[key][method+'_integral']=integral
        density_ax.set_yscale('log')
        density_ax.set_xlim(0,upper)
        density_ax.set_ylim(1e-6,1)
        density_ax.set_xlabel(label,fontsize=15)
        density_ax.set_ylabel('Probability density (log scale)',fontsize=14)
        density_ax.text(.5,-.26,f'({chr(97+column)})',transform=density_ax.transAxes,
                        ha='center',va='top',fontsize=16)
        density_ax.text(.97,.97,'n = 10,000 per method',transform=density_ax.transAxes,
                        ha='right',va='top',fontsize=11,
                        bbox=dict(facecolor='white',alpha=.92,edgecolor='none',pad=3))
        if column==0:
            density_ax.set_xticks([0,50,100,150,200,250,280])
            lines=[]
            for threshold in [50,100]:
                rates=[100*np.mean(a>threshold) for a in arrays]
                lines.append(f'> {threshold} m: {rates[0]:.2f}% → {rates[1]:.2f}%')
            density_ax.text(.97,.86,'Failure rate (VecMapLocNet → TopVecLoc)\n'+'\n'.join(lines),
                transform=density_ax.transAxes,ha='right',va='top',fontsize=11,linespacing=1.4,
                bbox=dict(facecolor='white',alpha=.94,edgecolor='#D9DEE1',boxstyle='round,pad=.4'))
        else:
            density_ax.set_xticks([0,30,60,90,120,150,180])
        for ax in [box,density_ax]:
            ax.grid(axis='y',which='major',color='#E4E7EA',lw=.7)
            ax.set_axisbelow(True)
            ax.minorticks_off()
            ax.tick_params(axis='both',which='both',direction='in',top=True,right=True,length=3.5)
            ax.tick_params(axis='both',which='minor',bottom=False,top=False,left=False,right=False)
            assert len(ax.xaxis.get_minorticklocs())==0
            assert len(ax.yaxis.get_minorticklocs())==0
    handles=[Line2D([],[],color=COLORS[m],lw=2.5,label=NAMES[m]) for m in NAMES]
    handles.append(Line2D([],[],ls='None',marker='D',markerfacecolor='white',
                          markeredgecolor='#20252A',markersize=6,label='Mean (boxplots)'))
    box_fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.535,.825),
               ncol=3,frameon=True,edgecolor='#CFD4D8',fontsize=13)
    kde_fig.legend(handles=handles[:2],loc='lower center',bbox_to_anchor=(.535,.825),
                   ncol=2,frameon=True,edgecolor='#CFD4D8',fontsize=13)
    for fig,name in [(box_fig,'long_tail_boxplots'),(kde_fig,'long_tail_kde')]:
        stem=OUTPUT/name
        for ext in ['png','tiff','pdf','svg']:
            opts={'dpi':600} if ext in ('png','tiff') else {}
            if ext=='tiff': opts['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(stem.with_suffix('.'+ext),**opts)
        fig.savefig(OUTPUT/(name+'_preview.png'),dpi=120)
        plt.close(fig)
    pd.DataFrame(summaries).to_csv(OUTPUT/'distribution_statistics.csv',index=False)
    pd.concat(densities,ignore_index=True).to_csv(OUTPUT/'kde_source.csv',index=False)
    pd.concat([f[['index']+[m[0] for m in METRICS]].assign(method=NAMES[m])
               for m,f in samples.items()],ignore_index=True).to_csv(OUTPUT/'source_errors.csv',index=False)
    notes=dict(n_per_method=10000,dpi=600,dimensions_inches_per_figure=[13.2,5.6],
        ticks='Minor ticks removed on both axes; all remaining major ticks face inward.',
        sources={m:{'path':INPUTS[m],'sha256':hashlib.sha256((ROOT/INPUTS[m]).read_bytes()).hexdigest()}
                 for m in NAMES},
        boxplot='All outliers shown at their true values without jitter; box Q1–Q3, median black, '
                'whiskers 1.5 IQR, diamond arithmetic mean; y is linear below 1, log above 1.',
        density='Gaussian KDE, common absolute pooled Scott bandwidth within each metric; '
                'physical-boundary reflection. All data included. Linear error axis, logarithmic density axis. '
                'Plotting range 1e-6 to 1 hides only density curves below display floor, not samples.',
        kde=kde_notes,interpretation='Descriptive long error tails, not a claim of a power-law distribution or statistical significance.',
        limitations='Existing fixed checkpoints; current TopVecLoc was previously selected on test-subset results.')
    (OUTPUT/'figure_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    (OUTPUT/'caption_zh.md').write_text(
        ' X　VecMapLocNet  TopVecLoc  UAV2MAP 。'
        '10,000。（a）；（b）。'
        '2575，，，1.5，'
        '。1、1，。'
        '\n\n Y　VecMapLocNet  TopVecLoc  UAV2MAP 。'
        '10,000。（a）；（b）。'
        '，（Scott）；'
        '，。50 m100 m。'
        '；，。\n')
    print(OUTPUT,flush=True)


if __name__=='__main__':
    main()

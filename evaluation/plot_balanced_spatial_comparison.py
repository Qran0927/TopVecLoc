"""OSM-backed illustrative single-frame spatial sequences, never temporal tracking."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from evaluation.balanced_spatial_comparison import ROOT, load_table
from osm_trajectory_map import latlon_to_world_pixels, yaw_to_image_vectors, build_osm_mosaic
from osm_viewport import fixed_aspect_viewport, crop_mosaic_to_viewport

COLORS = {'gt':'#20252A','baseline':'#E58B2B','ours':'#1674B8'}
ZOOM = 17


def draw_case(ax, frame, projection, case, title, method=None):
    coordinates = {}
    for name,prefix in [('gt',''),('baseline','baseline_'),('ours','ours_')]:
        xy = frame[[prefix+'east_m',prefix+'north_m']].to_numpy(float)
        geo = projection.unproject(xy)
        coordinates[name] = latlon_to_world_pixels(geo[:,0],geo[:,1],ZOOM)
    viewport = fixed_aspect_viewport(np.vstack(list(coordinates.values())),padding_pixels=55)
    x0,x1,y0,y1 = viewport
    cover = np.array([[x0,y0],[x1-1,y1-1]])
    image,extent = build_osm_mosaic(cover,ROOT/'.cache/osm_tiles',zoom=ZOOM,padding_pixels=0)
    mosaic = crop_mosaic_to_viewport(image,extent,viewport)
    ax.imshow(mosaic,extent=(x0,x1,y1,y0),interpolation='bilinear',alpha=.82)
    for name,label,style,width,zorder in [
        ('baseline','VecMapLocNet','--',1.45,4),('ours','TopVecLoc','-',1.65,5),
        ('gt','Ground truth','-',1.05,6)]:
        if method is not None and name not in ('gt',method):
            continue
        xy = coordinates[name]
        line, = ax.plot(xy[:,0],xy[:,1],style,color=COLORS[name],lw=width,
                       marker='o',markersize=2.7,label=label,zorder=zorder)
        line.set_path_effects([pe.Stroke(linewidth=width+1.4,foreground='white',alpha=.85),pe.Normal()])
    center_lat = float(frame.latitude.mean())
    meters_per_pixel = 156543.03392*np.cos(np.deg2rad(center_lat))/(2**ZOOM)
    positions = np.arange(0,len(frame),2)
    for name,zorder in [('baseline',7),('ours',8),('gt',9)]:
        if method is not None and name not in ('gt',method):
            continue
        yaw = frame.gt_yaw.to_numpy() if name=='gt' else frame[name+'_pred_yaw'].to_numpy()
        vectors = yaw_to_image_vectors(yaw[positions])*(14/meters_per_pixel)
        xy = coordinates[name][positions]
        ax.quiver(xy[:,0],xy[:,1],vectors[:,0],vectors[:,1],angles='xy',
            scale_units='xy',scale=1,color=COLORS[name],width=.003,
            headwidth=4,headlength=4.5,zorder=zorder)
    for index,label,offset in [(0,'1',(5,6)),(-1,'20',(5,-10))]:
        text = ax.annotate(label,coordinates['gt'][index],xytext=offset,textcoords='offset points',
            fontsize=8,color=COLORS['gt'],fontweight='bold',zorder=12)
        text.set_path_effects([pe.Stroke(linewidth=2,foreground='white'),pe.Normal()])
    summary = (f"Position MAE: {case['baseline_position_error_m']:.2f} → {case['ours_position_error_m']:.2f} m\n"
               f"Yaw MAE: {case['baseline_yaw_error_deg']:.2f} → {case['ours_yaw_error_deg']:.2f}°")
    if method is not None:
        summary = (f"Position MAE: {case[method+'_position_error_m']:.2f} m\n"
                   f"Yaw MAE: {case[method+'_yaw_error_deg']:.2f}°")
    ax.text(.015,.975,summary,transform=ax.transAxes,ha='left',va='top',fontsize=12,
        linespacing=1.25,
        bbox=dict(boxstyle='round,pad=.32',facecolor='white',edgecolor='#B9C1C5',alpha=.96),zorder=15)
    ax.annotate('N',xy=(.955,.93),xytext=(.955,.82),xycoords='axes fraction',
        ha='center',va='center',fontsize=10,fontweight='bold',
        arrowprops=dict(arrowstyle='-|>',color='#20252A',lw=1),zorder=15)
    sx,sy = x0+.04*(x1-x0),y1-.055*(y1-y0)
    scale = 100/meters_per_pixel
    bar, = ax.plot([sx,sx+scale],[sy,sy],color='#20252A',lw=2,zorder=14)
    bar.set_path_effects([pe.Stroke(linewidth=3.8,foreground='white'),pe.Normal()])
    ax.text(sx+scale/2,sy-7,'100 m',ha='center',va='bottom',fontsize=7,
            bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1),zorder=15)
    ax.text(.99,.012,'© OpenStreetMap contributors',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=5.5,
        bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1),zorder=15)
    ax.set_xlim(x0,x1); ax.set_ylim(y1,y0)
    ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_title(title,fontfamily='DejaVu Sans',fontsize=16,fontweight='bold',pad=10)
    return dict(viewport=list(viewport),zoom=ZOOM,background='OpenStreetMap raster tiles; uniform alpha 0.82',
                yaw_arrow_sample_orders=(positions+1).tolist(),point_labels='1 and 20 are spatial order, not time')


def write_explanation(output):
    text = '''# 

，，、。400 m，20；，、，。，、。，。，，10,000。

：6 TopVecLocVecMapLocNet。（a、b）；（c）；（d）。、、VecMapLocNetTopVecLoc；，。20，120，。VecMapLocNetTopVecLoc。，；。，。：© OpenStreetMap contributors，https://www.openstreetmap.org/copyright。
'''
    (output/'explanation_and_caption_zh.md').write_text(text)


def create_layout(layout):
    if layout=='1x4':
        fig,axes = plt.subplots(1,4,figsize=(24,5.8))
        fig.subplots_adjust(left=.012,right=.995,bottom=.035,top=.84,wspace=.035)
    elif layout=='2x2':
        fig,axes = plt.subplots(2,2,figsize=(13,11.5))
        fig.subplots_adjust(left=.025,right=.985,bottom=.07,top=.91,wspace=.045,hspace=.17)
    else:
        raise ValueError(layout)
    return fig,axes


def add_shared_legend(fig,axes):
    # Resolve equal-aspect map boxes to center the compact legend over the maps.
    fig.canvas.draw()
    left = min(ax.get_position().x0 for ax in axes.flat)
    right = max(ax.get_position().x1 for ax in axes.flat)
    handles = [Line2D([],[],color=COLORS['gt'],lw=1.4,marker='o',markersize=3,label='Ground truth'),
               Line2D([],[],color=COLORS['baseline'],ls='--',lw=1.6,marker='o',markersize=3,label='VecMapLocNet'),
               Line2D([],[],color=COLORS['ours'],lw=1.8,marker='o',markersize=3,label='TopVecLoc')]
    legend = fig.legend(handles=handles,loc='lower center',bbox_to_anchor=((left+right)/2,.93),
        bbox_transform=fig.transFigure,borderaxespad=0,ncol=3,
        frameon=True,fancybox=True,facecolor='white',edgecolor='#D0D0D0',
        framealpha=1,fontsize=12,handlelength=2.7,columnspacing=2.5,borderpad=.4)
    legend.get_frame().set_linewidth(.8)
    return legend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'figures/balanced_spatial_comparison')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--layout',choices=['1x4','2x2'],default='1x4')
    args = parser.parse_args()
    output = args.output or ROOT/f'figures/balanced_spatial_comparison_{args.layout}'
    stem = output/'balanced_spatial_comparison'
    if stem.with_suffix('.png').exists():
        raise FileExistsError(stem)
    protocol = json.loads((args.input/'selection_protocol.json').read_text())
    for source in protocol['inputs'].values():
        digest = hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()
        if digest != source['sha256']:
            raise RuntimeError('Source data changed after case selection; regenerate in a new directory')
    table,projection = load_table()
    output.mkdir(parents=True,exist_ok=True)
    if output.resolve()!=args.input.resolve():
        for filename in ['selection_protocol.json','case_summary.csv','selected_samples.csv','candidate_pool.csv']:
            shutil.copyfile(args.input/filename,output/filename)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes = create_layout(args.layout)
    dimensions = fig.get_size_inches().tolist()
    labels = ['(a) Improvement case 1','(b) Improvement case 2',
              '(c) Near-tie case','(d) Failure case']
    panels = []
    for ax,case,label in zip(axes.flat,protocol['cases'],labels):
        frame = table.loc[case['ids']]
        panels.append(draw_case(ax,frame,projection,case,label))
    add_shared_legend(fig,axes)
    for extension in ['png','tiff','pdf','svg']:
        options = {'dpi':600} if extension in ('png','tiff') else {}
        if extension=='tiff':
            options['pil_kwargs'] = {'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+extension),**options)
    fig.savefig(output/'preview.png',dpi=120)
    plt.close(fig)
    (output/'plot_metadata.json').write_text(json.dumps(dict(dpi=600,dimensions_inches=dimensions,layout=args.layout,
        panels=panels,metric='Maximum-pose errors; no trajectory smoothing or optimization.',
        selection='Outcome-stratified illustrative cases; not unbiased aggregate evidence.'),indent=2)+'\n')
    write_explanation(output)
    print(stem.with_suffix('.png'),flush=True)


if __name__ == '__main__':
    main()

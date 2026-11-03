"""Compare identical spatial samples in two method rows using measured predictions."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.balanced_spatial_comparison import ROOT, load_table
from evaluation.plot_balanced_spatial_comparison import draw_case, add_shared_legend


def main():
    source = ROOT/'figures/balanced_spatial_comparison'
    output = ROOT/'figures/spatial_comparison_method_rows'
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((source/'selection_protocol.json').read_text())
    for item in protocol['inputs'].values():
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256']
    table, projection = load_table()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2,4,figsize=(24,11))
    fig.subplots_adjust(left=.045,right=.995,bottom=.025,top=.90,wspace=.035,hspace=.08)
    labels = ['(a) Improvement case 1','(b) Improvement case 2',
              '(c) Near-tie case','(d) Failure case']
    panels = []
    for row, method in enumerate(['baseline','ours']):
        for column, case in enumerate(protocol['cases']):
            panel = draw_case(axes[row,column],table.loc[case['ids']],projection,case,
                              labels[column] if row==0 else '',method=method)
            panel.update(method=method,case=column)
            panels.append(panel)
    for column in range(4):
        assert panels[column]['viewport']==panels[4+column]['viewport']
    add_shared_legend(fig,axes)
    fig.canvas.draw()
    for row, label in enumerate(['VecMapLocNet','TopVecLoc']):
        box = axes[row,0].get_position()
        fig.text(.017,(box.y0+box.y1)/2,label,rotation=90,ha='center',va='center',
                 fontsize=19,fontweight='bold')
    output.mkdir(parents=True)
    for filename in ['selection_protocol.json','case_summary.csv','selected_samples.csv','candidate_pool.csv']:
        shutil.copyfile(source/filename,output/filename)
    stem = output/'spatial_comparison_method_rows'
    for extension in ['png','tiff','pdf','svg']:
        options = {'dpi':600} if extension in ('png','tiff') else {}
        if extension=='tiff':
            options['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+extension),**options)
    fig.savefig(output/'preview.png',dpi=100)
    plt.close(fig)
    (output/'plot_metadata.json').write_text(json.dumps({
        'dpi':600,'dimensions_inches':[24,11],'panels':panels,
        'scope':'Spatial ordering of independent maximum-pose predictions; no temporal smoothing.',
        'selection':'Two improvements, one near tie and one failure; outcome-stratified illustrations.'
    },indent=2)+'\n')
    print(stem.with_suffix('.png'))


if __name__=='__main__':
    main()

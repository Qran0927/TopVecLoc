"""Four-panel fixed-checkpoint sensitivity comparison, without retraining."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.plot_recall_comparison import ROOT, PANELS, build_figure, read_source

VARIANTS = ('none','no_attention','no_pe','no_context','no_edges','random_edges')
STYLES = [
    ('Full TopVecLoc','#179EAA','-','D'),
    ('w/o global attention','#D85C5C','-','o'),
    ('w/o geographic PE','#527BAD','-','s'),
    ('w/o POI/region context','#BC9141','-','^'),
    ('w/o road edges','#9272AB','-','v'),
    ('Rewired road edges','#738A60','-','P'),
]


def read_data(path):
    return read_source(path,VARIANTS)


def make_figure(data):
    return build_figure(data,styles=STYLES,figsize=(16,5.8),legend_columns=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'experiments/inference_interventions/full_v1/comparison.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'figures/intervention_recalls_1x4')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = read_data(args.source)
    fig,_ = make_figure(data)
    args.output.mkdir(parents=True)
    data.to_csv(args.output/'source_data.csv',index=False)
    stem = args.output/'topvecloc_intervention_recalls'
    for extension in ['png','tiff','pdf','svg']:
        options = {'dpi':600} if extension in ('png','tiff') else {}
        if extension == 'tiff':
            options['pil_kwargs'] = {'compression':'tiff_lzw'}
        fig.savefig(stem.with_suffix('.'+extension),**options)
    fig.savefig(args.output/'preview.png',dpi=150)
    plt.close(fig)
    metadata = dict(backend='Python/matplotlib',dpi=600,dimensions_inches=[16,5.8],
        settings=dict(zip(VARIANTS,[s[0] for s in STYLES])),
        panels=[p[2] for p in PANELS],n_per_setting=10000,test_city='SanFrancisco',
        estimator='maximum-probability pose',threshold_comparison='strict <',
        scientific_scope='Fixed-checkpoint inference interventions; not retrained ablations.',
        uncertainty='Point estimates only; no confidence intervals or multi-seed claims.',
        limitation='Existing checkpoint was previously selected with test-subset feedback.',
        integrity='Measured values unchanged; no jitter, smoothing, truncated axes, or invented values.')
    (args.output/'figure_notes.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()

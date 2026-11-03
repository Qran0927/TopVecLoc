import numpy as np


def test_box_statistics_keep_outliers_in_mean_and_sample_count():
    from evaluation.plot_error_boxplots import describe_box
    stats = describe_box(np.array([0.,1.,2.,3.,100.]))
    assert stats['n'] == 5 and stats['median'] == 2
    assert stats['q1'] == 1 and stats['q3'] == 3
    assert stats['whisker_low'] == 0 and stats['whisker_high'] == 3
    assert stats['outlier_count'] == 1
    assert stats['mean'] == 21.2 and stats['maximum'] == 100


def test_boxplots_use_measured_samples_and_produce_four_panels():
    import matplotlib.pyplot as plt
    from evaluation.plot_error_boxplots import load_samples, make_figure, INTERVENTIONS
    samples = load_samples(INTERVENTIONS)
    assert all(len(f)==10000 for f in samples.values())
    fig,axes = make_figure(samples,'interventions',show_outliers=False)
    assert len(axes) == 4
    fig.canvas.draw()
    for ax in axes:
        assert len(ax.patches) == 6
        assert ax.get_ylim()[0] == 0
    plt.close(fig)

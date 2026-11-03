from pathlib import Path


def test_six_intervention_curves_have_exact_values_and_shared_legend():
    import matplotlib.pyplot as plt
    from evaluation.plot_intervention_recalls import make_figure, read_data, VARIANTS
    data = read_data(Path('experiments/inference_interventions/full_v1/comparison.csv'))
    assert data['variant'].tolist() == list(VARIANTS)
    fig,axes = make_figure(data)
    fig.canvas.draw()
    assert len(axes) == 4 and len(fig.legends) == 1
    legend_box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
    assert legend_box.y0 > max(ax.get_window_extent().y1 for ax in axes)
    for ax,family,unit in zip(axes,['position','yaw','longitudinal','lateral'],['m','deg','m','m']):
        assert len(ax.lines) == 6 and ax.get_ylim() == (0,100)
        for j,line in enumerate(ax.lines):
            expected = data.iloc[j][[f'{family}_recall_{t}{unit}' for t in [1,3,5,10]]].to_numpy(float)
            assert all(abs(a-b)<1e-10 for a,b in zip(line.get_ydata(),expected))
            assert line.get_linestyle() == '-'
    plt.close(fig)

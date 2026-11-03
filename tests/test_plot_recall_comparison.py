from pathlib import Path


def test_recall_figure_has_requested_order_and_untruncated_scales():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.plot_recall_comparison import build_figure, read_source
    source = Path('experiments/inference_interventions/full_v1/comparison.csv')
    data = read_source(source)
    assert data['variant'].tolist() == ['VecMapLocNet_released','none']
    fig, axes = build_figure(data)
    fig.canvas.draw()
    assert all(ax.yaxis.label.get_window_extent(fig.canvas.get_renderer()).x0 >= 0 for ax in axes)
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in axes)
    legend_box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
    assert legend_box.y0 > max(ax.get_window_extent().y1 for ax in axes)
    assert len(axes) == 4
    for ax,family,unit in zip(axes,['position','yaw','longitudinal','lateral'],['m','deg','m','m']):
        assert ax.get_ylim() == (0,100)
        assert len(ax.lines) == 2
        assert all(line.get_linestyle() == '-' for line in ax.lines)
        assert [line.get_color() for line in ax.lines] == ['#D85C5C','#179EAA']
        for j in range(2):
            expected = data.iloc[j][[f'{family}_recall_{t}{unit}' for t in [1,3,5,10]]].to_numpy(float)
            assert all(abs(a-b)<1e-10 for a,b in zip(ax.lines[j].get_ydata(),expected))
    plt.close(fig)

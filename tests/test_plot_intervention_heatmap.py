from pathlib import Path
import numpy as np


def test_delta_heatmaps_use_actual_subtraction_and_one_shared_zero_centered_scale():
    import matplotlib.pyplot as plt
    from evaluation.plot_intervention_heatmap import read_data, calculate_deltas, make_figure
    data = read_data(Path('experiments/inference_interventions/full_v1/comparison.csv'))
    delta = calculate_deltas(data)
    assert delta.shape == (5,16)
    assert np.isclose(delta.loc['no_attention','position_recall_5m'],-1.12)
    assert np.isclose(delta.loc['no_context','yaw_recall_5deg'],.04)
    fig,axes = make_figure(delta)
    fig.canvas.draw()
    assert len(axes) == 4
    assert all('\n' not in label.get_text() for label in axes[0].get_yticklabels())
    assert len(fig.texts) == 0
    norm = axes[0].images[0].norm
    assert norm.vcenter == 0 and norm.vmin == -norm.vmax
    for ax in axes:
        assert ax.images[0].norm is norm
        assert len(ax.texts) == 20
        values = np.asarray(ax.images[0].get_array()).flatten()
        labels = [float(t.get_text().replace('−','-')) for t in ax.texts]
        np.testing.assert_allclose(labels,np.round(values,2),atol=1e-10)
    plt.close(fig)


def test_idw_and_origin_deltas_use_full_k3_as_reference():
    import matplotlib.pyplot as plt
    from evaluation.plot_intervention_heatmap import load_group, calculate_deltas, make_figure, GROUPS
    variants,labels = GROUPS['idw_origin']
    data = load_group(Path('experiments/inference_interventions/full_v1/comparison.csv'),'idw_origin')
    delta = calculate_deltas(data,variants)
    assert delta.index.tolist() == ['k1','k3','k5','k8','shift_origin']
    np.testing.assert_allclose(delta.loc['k3'].to_numpy(float),0)
    assert np.isclose(delta.loc['k1','position_recall_5m'],.07)
    assert np.isclose(delta.loc['shift_origin','position_recall_10m'],-3.41)
    fig,axes = make_figure(delta,row_labels=labels)
    assert axes[0].get_yticklabels()[1].get_text() == 'IDW K = 3 (reference)'
    assert all(ax.images[0].norm is axes[0].images[0].norm for ax in axes)
    plt.close(fig)

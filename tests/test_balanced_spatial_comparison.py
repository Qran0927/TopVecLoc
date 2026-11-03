import numpy as np
import pandas as pd


def example_table():
    ids = np.arange(100)
    return pd.DataFrame({'east_m':(ids//25)*800+(ids%25)*3+10,
                         'north_m':(ids%25)*2+10,
                         'baseline_position_error_m':10.,'ours_position_error_m':5.,
                         'baseline_yaw_error_deg':10.,'ours_yaw_error_deg':5.},index=ids)


def test_candidate_membership_and_order_are_independent_of_prediction_errors():
    from evaluation.balanced_spatial_comparison import build_candidates
    table = example_table()
    a = build_candidates(table)
    table['ours_position_error_m'] = np.arange(100)*100
    b = build_candidates(table)
    assert [g['ids'] for g in a] == [g['ids'] for g in b]
    assert len(a) == 4 and all(len(g['ids'])==20 for g in a)
    assert len(set(sum([g['ids'] for g in a],[]))) == 80


def test_selection_covers_improvement_tie_and_failure_not_only_wins():
    from evaluation.balanced_spatial_comparison import choose_cases
    groups = []
    for i,(pg,yg) in enumerate([(5,6),(10,10),(18,20),(.1,-.1),(-8,-10),(-20,-30)]):
        groups.append(dict(ids=list(range(i*20,(i+1)*20)),center_east_m=i*1000.,
            center_north_m=0.,position_gain_m=pg,yaw_gain_deg=yg,
            ours_position_error_m=20.,ours_yaw_error_deg=25.,joint_gain=pg/25+yg/22))
    chosen = choose_cases(groups,25,22)
    assert [c['category'] for c in chosen] == ['improvement','improvement','near_tie','failure']
    assert all(c['position_gain_m']>=3 and c['yaw_gain_deg']>=3 for c in chosen[:2])
    assert abs(chosen[2]['position_gain_m'])<=1 and abs(chosen[2]['yaw_gain_deg'])<=1
    assert chosen[3]['position_gain_m']<0 and chosen[3]['yaw_gain_deg']<0
    assert len(set(sum([g['ids'] for g in chosen],[])))==80


def test_actual_metric_offsets_preserve_saved_errors_and_selected_case_definitions():
    from evaluation.balanced_spatial_comparison import load_table, build_candidates, choose_cases
    table,_ = load_table()
    for model in ['baseline','ours']:
        offsets = table[[model+'_east_m',model+'_north_m']].to_numpy()-table[['east_m','north_m']].to_numpy()
        np.testing.assert_allclose(np.linalg.norm(offsets,axis=1),table[model+'_position_error_m'],atol=1e-9)
    groups = build_candidates(table)
    selected = choose_cases(groups,table.baseline_position_error_m.mean(),table.baseline_yaw_error_deg.mean())
    assert len(set(sum([g['ids'] for g in selected],[])))==80


def test_single_row_layout_has_four_aligned_panels():
    import matplotlib.pyplot as plt
    from evaluation.plot_balanced_spatial_comparison import create_layout
    fig,axes = create_layout('1x4')
    assert len(axes.flat)==4
    boxes = [ax.get_position() for ax in axes.flat]
    assert all(abs(box.y0-boxes[0].y0)<1e-10 for box in boxes)
    assert all(boxes[i].x1<boxes[i+1].x0 for i in range(3))
    assert fig.get_size_inches()[0]>3*fig.get_size_inches()[1]
    plt.close(fig)


def test_shared_legend_is_framed_compact_and_centered_without_notes():
    import matplotlib.pyplot as plt
    from evaluation.plot_balanced_spatial_comparison import create_layout, add_shared_legend
    fig,axes = create_layout('1x4')
    legend = add_shared_legend(fig,axes)
    fig.canvas.draw()
    assert len(fig.texts)==0 and len(fig.legends)==1
    assert legend.get_frame_on()
    bounds = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    left = min(ax.get_position().x0 for ax in axes.flat)
    right = max(ax.get_position().x1 for ax in axes.flat)
    assert bounds.width < .6*(right-left)
    assert abs((bounds.x0+bounds.x1)/2-(left+right)/2)<1e-5
    plt.close(fig)


def test_map_panel_title_is_bold_and_enlarged():
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    from evaluation.balanced_spatial_comparison import load_table
    from evaluation.plot_balanced_spatial_comparison import draw_case
    table,projection = load_table()
    protocol = json.loads(Path('figures/balanced_spatial_comparison/selection_protocol.json').read_text())
    case = protocol['cases'][0]
    fig,ax = plt.subplots()
    draw_case(ax,table.loc[case['ids']],projection,case,'(a) Improvement case 1')
    assert ax.title.get_fontweight() == 'bold'
    assert ax.title.get_fontsize() == 16
    assert 'DejaVu Sans' in ax.title.get_fontfamily()
    summary = next(t for t in ax.texts if t.get_text().startswith('Position MAE:'))
    assert summary.get_fontsize() == 12
    plt.close(fig)

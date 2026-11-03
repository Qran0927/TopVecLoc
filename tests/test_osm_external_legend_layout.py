from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_osm_uses_a_compact_fixed_map_axis_and_internal_legend():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert 'plt.figure(figsize=(8.0, 6.4))' in source
    assert 'fig.add_axes([0.025, 0.025, 0.95, 0.95])' in source
    legend_block = source[source.index("    ax.legend(") : source.index("\n\n    output_stem")]
    assert 'loc="upper right"' in legend_block
    assert "bbox_to_anchor" not in legend_block

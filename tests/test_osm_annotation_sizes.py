from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_osm_legend_and_error_summary_use_larger_text():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert 'fontsize=9.5' in source
    legend_block = source[source.index("    ax.legend(") : source.index("\n\n    output_stem")]
    assert 'fontsize=10' in legend_block
    assert 'handlelength=2.8' in legend_block
    assert 'borderpad=0.6' in legend_block

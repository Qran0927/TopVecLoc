from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_osm_legend_is_a_single_vertical_column():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")
    legend_block = source[source.index("    ax.legend(") : source.index("\n\n    output_stem")]

    assert "ncols=1" in legend_block
    assert "ncols=3" not in legend_block

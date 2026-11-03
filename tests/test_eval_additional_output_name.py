from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_additional_candidate_evaluation_uses_a_non_overwriting_suffix():
    source = (ROOT / "eval_uav.py").read_text(encoding="utf-8")

    assert '"_additional_selected_per_sample.csv"' in source

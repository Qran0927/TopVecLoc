from pathlib import Path
import runpy

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_eval_script_uses_all_three_generated_candidate_groups():
    namespace = runpy.run_path(str(ROOT / "additional_trajectory_candidates.py"))
    groups = namespace["ADDITIONAL_CANDIDATE_GROUPS"]
    source = (ROOT / "eval_uav.py").read_text(encoding="utf-8")

    assert "from additional_trajectory_candidates import" in source
    assert len(groups) == 3
    assert all(len(group) == 20 for group in groups)
    flattened = [sample_id for group in groups for sample_id in group]
    assert len(flattened) == len(set(flattened)) == 60


def test_generated_groups_match_the_ordered_selection_manifest():
    namespace = runpy.run_path(str(ROOT / "additional_trajectory_candidates.py"))
    groups = namespace["ADDITIONAL_CANDIDATE_GROUPS"]
    manifest = pd.read_csv(
        ROOT / "outputs/additional_trajectory_candidates_manifest.csv"
    )

    manifest_groups = [
        group.sort_values("sequence_order")["index"].astype(int).tolist()
        for _, group in manifest.groupby("candidate", sort=True)
    ]
    assert groups == manifest_groups

from pathlib import Path

import numpy as np
import pandas as pd

from spatial_candidate_selection import (
    EXISTING_CANDIDATE_IDS,
    build_manifest,
    eligible_samples,
    select_candidate_groups,
    summarize_groups,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_table() -> pd.DataFrame:
    rows = []
    sample_id = 100
    for cluster_index, center_east in enumerate((0.0, 1000.0, 2000.0)):
        for offset in range(21):
            current_id = 900 if cluster_index == 0 and offset == 0 else sample_id
            sample_id += 1
            rows.append(
                {
                    "index": current_id,
                    "longitude": -122.44 + (center_east + offset * 3.0) / 88000.0,
                    "latitude": 37.76 + offset / 1_000_000.0,
                    "east_m": center_east + offset * 3.0,
                    "north_m": float(offset),
                    "baseline_pos_error_m": 35.0 + cluster_index,
                    "ours_pos_error_m": 5.0,
                    "baseline_yaw_error_deg": 25.0 + cluster_index,
                    "ours_yaw_error_deg": 2.0,
                    "position_gain_m": 30.0 + cluster_index,
                    "yaw_gain_deg": 23.0 + cluster_index,
                    "simultaneous_win": True,
                }
            )
    return pd.DataFrame(rows).set_index("index")


def test_eligible_samples_applies_all_quality_thresholds():
    table = _synthetic_table().iloc[:6].copy()
    table.loc[table.index[1], "ours_pos_error_m"] = 15.1
    table.loc[table.index[2], "ours_yaw_error_deg"] = 5.1
    table.loc[table.index[3], "position_gain_m"] = 9.9
    table.loc[table.index[4], "yaw_gain_deg"] = 4.9
    table.loc[table.index[5], "simultaneous_win"] = False

    result = eligible_samples(table)

    assert result.index.tolist() == [table.index[0]]


def test_select_groups_is_deterministic_non_overlapping_and_spatially_distinct():
    table = _synthetic_table()

    first = select_candidate_groups(table, excluded_ids={900})
    second = select_candidate_groups(table, excluded_ids={900})

    assert first == second
    assert [len(group) for group in first] == [20, 20, 20]
    flattened = [sample_id for group in first for sample_id in group]
    assert len(flattened) == len(set(flattened)) == 60
    assert 900 not in flattened
    centroids = np.array(
        [table.loc[group, ["east_m", "north_m"]].mean().to_numpy() for group in first]
    )
    pairwise = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    assert pairwise[np.triu_indices(3, k=1)].min() >= 500.0


def test_manifest_and_summary_preserve_candidate_order_and_metric_claim():
    table = _synthetic_table()
    groups = select_candidate_groups(table, excluded_ids={900})

    manifest = build_manifest(table, groups)
    summary = summarize_groups(table, groups)

    assert manifest.groupby("candidate").size().to_dict() == {2: 20, 3: 20, 4: 20}
    assert manifest.groupby("candidate")["sequence_order"].apply(list).tolist() == [
        list(range(20)),
        list(range(20)),
        list(range(20)),
    ]
    assert summary["candidate"].tolist() == [2, 3, 4]
    assert (summary["ours_mean_position_error_m"] < summary["baseline_mean_position_error_m"]).all()
    assert (summary["ours_mean_yaw_error_deg"] < summary["baseline_mean_yaw_error_deg"]).all()
    assert (summary["simultaneous_win_rate"] >= 0.85).all()


def test_existing_candidate_constant_contains_the_first_twenty_samples():
    assert len(EXISTING_CANDIDATE_IDS) == 20
    assert len(set(EXISTING_CANDIDATE_IDS)) == 20
    assert EXISTING_CANDIDATE_IDS[0] == 7420
    assert EXISTING_CANDIDATE_IDS[-1] == 7587


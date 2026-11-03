import pandas as pd

from spatial_candidate_selection import select_candidate_groups, summarize_groups


def test_select_groups_rejects_a_high_gain_sequence_with_a_long_spatial_jump():
    rows = []
    for offset in range(20):
        rows.append(
            {
                "index": 1000 + offset,
                "longitude": -122.44,
                "latitude": 37.76,
                "east_m": offset * 500.0,
                "north_m": 0.0,
                "baseline_pos_error_m": 205.0,
                "ours_pos_error_m": 5.0,
                "baseline_yaw_error_deg": 202.0,
                "ours_yaw_error_deg": 2.0,
                "position_gain_m": 200.0,
                "yaw_gain_deg": 200.0,
                "simultaneous_win": True,
            }
        )
        rows.append(
            {
                "index": 2000 + offset,
                "longitude": -122.42,
                "latitude": 37.78,
                "east_m": offset * 3.0,
                "north_m": 10_000.0 + offset,
                "baseline_pos_error_m": 35.0,
                "ours_pos_error_m": 5.0,
                "baseline_yaw_error_deg": 32.0,
                "ours_yaw_error_deg": 2.0,
                "position_gain_m": 30.0,
                "yaw_gain_deg": 30.0,
                "simultaneous_win": True,
            }
        )
    table = pd.DataFrame(rows).set_index("index")

    [selected] = select_candidate_groups(table, excluded_ids=set(), count=1)

    assert set(selected) == set(range(2000, 2020))
    summary = summarize_groups(table, [selected])
    assert summary.at[0, "max_step_m"] < 10.0
    assert summary.at[0, "max_step_limit_m"] == 400.0

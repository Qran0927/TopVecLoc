"""Deterministically select additional spatial sequences for trajectory figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXISTING_CANDIDATE_IDS = (
    7420,
    118,
    539,
    8408,
    7064,
    2269,
    6983,
    2848,
    3315,
    8572,
    5336,
    1661,
    7190,
    3719,
    8127,
    7683,
    8363,
    8192,
    7066,
    7587,
)

SELECTION_LIMITS = {
    "ours_position_max_m": 15.0,
    "ours_yaw_max_deg": 5.0,
    "position_gain_min_m": 10.0,
    "yaw_gain_min_deg": 5.0,
    "group_size": 20,
    "candidate_count": 3,
    "simultaneous_win_rate_min": 0.85,
    "centroid_distance_min_m": 500.0,
    "max_step_limit_m": 400.0,
}


def load_selection_table(root: Path) -> pd.DataFrame:
    """Join full evaluation results with ground-truth geographic coordinates."""
    baseline = pd.read_csv(root / "last-step-checkpointing-v1_per_sample.csv").set_index(
        "index"
    )
    ours = pd.read_csv(root / "checkpoint-best-step-150000_per_sample.csv").set_index(
        "index"
    )
    info = pd.read_csv(
        root / "UAV2MAP_v1.0/UAV2MAP/SanFrancisco/info.csv"
    ).set_index("id")

    common = baseline.index.intersection(ours.index).intersection(info.index)
    baseline = baseline.loc[common]
    ours = ours.loc[common]
    info = info.loc[common]
    latitude = info["uav_lat"].astype(float)
    longitude = info["uav_long"].astype(float)
    latitude_origin = float(latitude.mean())
    longitude_origin = float(longitude.mean())

    table = pd.DataFrame(index=common.astype(int))
    table.index.name = "index"
    table["longitude"] = longitude.to_numpy()
    table["latitude"] = latitude.to_numpy()
    table["east_m"] = (
        (longitude.to_numpy() - longitude_origin)
        * 111_320.0
        * np.cos(np.deg2rad(latitude_origin))
    )
    table["north_m"] = (latitude.to_numpy() - latitude_origin) * 110_540.0
    table["baseline_pos_error_m"] = baseline["pos_error_m"].to_numpy()
    table["ours_pos_error_m"] = ours["pos_error_m"].to_numpy()
    table["baseline_yaw_error_deg"] = baseline["yaw_error_deg"].to_numpy()
    table["ours_yaw_error_deg"] = ours["yaw_error_deg"].to_numpy()
    table["position_gain_m"] = (
        table["baseline_pos_error_m"] - table["ours_pos_error_m"]
    )
    table["yaw_gain_deg"] = (
        table["baseline_yaw_error_deg"] - table["ours_yaw_error_deg"]
    )
    table["simultaneous_win"] = (table["position_gain_m"] > 0) & (
        table["yaw_gain_deg"] > 0
    )
    return table.sort_index()


def eligible_samples(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed quality contract used for all candidate sequences."""
    mask = (
        table["simultaneous_win"].astype(bool)
        & (table["ours_pos_error_m"] <= SELECTION_LIMITS["ours_position_max_m"])
        & (table["ours_yaw_error_deg"] <= SELECTION_LIMITS["ours_yaw_max_deg"])
        & (table["position_gain_m"] >= SELECTION_LIMITS["position_gain_min_m"])
        & (table["yaw_gain_deg"] >= SELECTION_LIMITS["yaw_gain_min_deg"])
    )
    return table.loc[mask].sort_index()


def order_nearest_neighbour(group: pd.DataFrame) -> list[int]:
    """Order a spatial group into a deterministic nearest-neighbour path."""
    if group.empty:
        return []
    remaining = set(int(value) for value in group.index)
    start = min(
        remaining,
        key=lambda sample_id: (
            float(group.at[sample_id, "east_m"]),
            float(group.at[sample_id, "north_m"]),
            sample_id,
        ),
    )
    ordered = [start]
    remaining.remove(start)
    while remaining:
        current = ordered[-1]
        current_xy = group.loc[current, ["east_m", "north_m"]].to_numpy(dtype=float)
        next_id = min(
            remaining,
            key=lambda sample_id: (
                float(
                    np.linalg.norm(
                        group.loc[sample_id, ["east_m", "north_m"]].to_numpy(
                            dtype=float
                        )
                        - current_xy
                    )
                ),
                sample_id,
            ),
        )
        ordered.append(next_id)
        remaining.remove(next_id)
    return ordered


def _path_statistics(table: pd.DataFrame, ordered_ids: list[int]) -> tuple[float, float]:
    points = table.loc[ordered_ids, ["east_m", "north_m"]].to_numpy(dtype=float)
    if len(points) < 2:
        return 0.0, 0.0
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(steps.sum()), float(steps.max())


def select_candidate_groups(
    table: pd.DataFrame,
    excluded_ids: set[int],
    group_size: int = 20,
    count: int = 3,
) -> list[list[int]]:
    """Select high-gain, non-overlapping, geographically distinct sequences."""
    eligible = eligible_samples(table).drop(
        index=list(set(int(value) for value in excluded_ids)), errors="ignore"
    )
    if len(eligible) < group_size * count:
        raise ValueError(
            f"need at least {group_size * count} eligible samples, found {len(eligible)}"
        )

    coordinates = eligible[["east_m", "north_m"]].to_numpy(dtype=float)
    sample_ids = eligible.index.to_numpy(dtype=int)
    proposals = []
    seen_sets: set[frozenset[int]] = set()
    for seed_position, _seed_id in enumerate(sample_ids):
        distances = np.linalg.norm(coordinates - coordinates[seed_position], axis=1)
        nearest_positions = np.lexsort((sample_ids, distances))[:group_size]
        unordered_ids = sample_ids[nearest_positions].tolist()
        identity = frozenset(unordered_ids)
        if identity in seen_sets:
            continue
        seen_sets.add(identity)
        ordered_ids = order_nearest_neighbour(eligible.loc[unordered_ids])
        group = eligible.loc[ordered_ids]
        _path_length, max_step = _path_statistics(eligible, ordered_ids)
        if max_step > SELECTION_LIMITS["max_step_limit_m"]:
            continue
        position_gain = float(group["position_gain_m"].mean())
        yaw_gain = float(group["yaw_gain_deg"].mean())
        score = position_gain + yaw_gain - 0.02 * max_step
        centroid = group[["east_m", "north_m"]].mean().to_numpy(dtype=float)
        proposals.append((score, max_step, tuple(ordered_ids), centroid))

    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    accepted: list[list[int]] = []
    accepted_sets: list[set[int]] = []
    accepted_centroids: list[np.ndarray] = []
    minimum_centroid_distance = SELECTION_LIMITS["centroid_distance_min_m"]
    for _score, _max_step, ordered_ids, centroid in proposals:
        current_set = set(ordered_ids)
        if any(current_set & previous for previous in accepted_sets):
            continue
        if any(
            np.linalg.norm(centroid - previous) < minimum_centroid_distance
            for previous in accepted_centroids
        ):
            continue
        accepted.append(list(ordered_ids))
        accepted_sets.append(current_set)
        accepted_centroids.append(centroid)
        if len(accepted) == count:
            break

    if len(accepted) != count:
        raise ValueError(
            f"could only form {len(accepted)} spatially distinct groups; requested {count}"
        )
    return accepted


def build_manifest(table: pd.DataFrame, groups: list[list[int]]) -> pd.DataFrame:
    """Build one traceable row per selected sample in sequence order."""
    records = []
    for candidate, ordered_ids in enumerate(groups, start=2):
        for sequence_order, sample_id in enumerate(ordered_ids):
            row = table.loc[sample_id].to_dict()
            records.append(
                {
                    "candidate": candidate,
                    "sequence_order": sequence_order,
                    "index": int(sample_id),
                    **row,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_groups(table: pd.DataFrame, groups: list[list[int]]) -> pd.DataFrame:
    """Summarize the evidence and spatial geometry for each accepted sequence."""
    records = []
    for candidate, ordered_ids in enumerate(groups, start=2):
        group = table.loc[ordered_ids]
        path_length, max_step = _path_statistics(table, ordered_ids)
        records.append(
            {
                "candidate": candidate,
                "sample_count": len(group),
                "baseline_mean_position_error_m": group[
                    "baseline_pos_error_m"
                ].mean(),
                "ours_mean_position_error_m": group["ours_pos_error_m"].mean(),
                "mean_position_gain_m": group["position_gain_m"].mean(),
                "baseline_mean_yaw_error_deg": group[
                    "baseline_yaw_error_deg"
                ].mean(),
                "ours_mean_yaw_error_deg": group["ours_yaw_error_deg"].mean(),
                "mean_yaw_gain_deg": group["yaw_gain_deg"].mean(),
                "simultaneous_win_rate": group["simultaneous_win"].mean(),
                "path_length_m": path_length,
                "max_step_m": max_step,
                "centroid_longitude": group["longitude"].mean(),
                "centroid_latitude": group["latitude"].mean(),
                **SELECTION_LIMITS,
            }
        )
    return pd.DataFrame.from_records(records)


def _format_groups_module(groups: list[list[int]]) -> str:
    lines = [
        '"""Generated additional spatial candidate IDs; do not reorder."""',
        "",
        "ADDITIONAL_CANDIDATE_GROUPS = [",
    ]
    for candidate, group in enumerate(groups, start=2):
        lines.append(f"    [  # Candidate {candidate}")
        lines.extend(f"        {sample_id}," for sample_id in group)
        lines.append("    ],")
    lines.extend(["]", ""])
    return "\n".join(lines)


def write_selection_outputs(root: Path, table: pd.DataFrame, groups: list[list[int]]) -> None:
    """Save the manifest, summary, and importable grouped-ID module."""
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(table, groups)
    summary = summarize_groups(table, groups)
    manifest.to_csv(output_dir / "additional_trajectory_candidates_manifest.csv", index=False)
    summary.to_csv(output_dir / "additional_trajectory_candidates_summary.csv", index=False)
    (root / "additional_trajectory_candidates.py").write_text(
        _format_groups_module(groups), encoding="utf-8"
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    table = load_selection_table(root)
    groups = select_candidate_groups(table, set(EXISTING_CANDIDATE_IDS))
    write_selection_outputs(root, table, groups)
    print(f"Eligible samples: {len(eligible_samples(table))}")
    for candidate, group in enumerate(groups, start=2):
        print(f"Candidate {candidate}: {group}")
    print(summarize_groups(table, groups).to_string(index=False))


if __name__ == "__main__":
    main()

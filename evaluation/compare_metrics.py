"""Compare per-sample localization CSVs against a fixed baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ERROR_METRICS = {
    "position_error_mean",
    "position_error_median",
    "yaw_error_mean",
    "yaw_error_median",
}


def summarize_samples(frame: pd.DataFrame) -> dict[str, float]:
    required = {"index", "pos_error_m", "yaw_error_deg"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"per-sample CSV is missing columns: {sorted(missing)}")
    if frame["index"].duplicated().any():
        raise ValueError("per-sample CSV contains duplicate indices")
    values = frame[["pos_error_m", "yaw_error_deg"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("per-sample CSV contains non-finite errors")

    position = frame["pos_error_m"].astype(float)
    yaw = frame["yaw_error_deg"].astype(float)
    summary = {
        "position_error_mean": float(position.mean()),
        "position_error_median": float(position.median()),
        "yaw_error_mean": float(yaw.mean()),
        "yaw_error_median": float(yaw.median()),
    }
    for threshold in (1, 3, 5):
        summary[f"position_recall_{threshold}m"] = float(
            (position <= threshold).mean()
        )
        summary[f"yaw_recall_{threshold}deg"] = float((yaw <= threshold).mean())
    return summary


def compare_csvs(baseline_path, candidate_path) -> dict[str, object]:
    baseline = pd.read_csv(baseline_path).sort_values("index").reset_index(drop=True)
    candidate = pd.read_csv(candidate_path).sort_values("index").reset_index(drop=True)
    if len(baseline) != len(candidate) or not baseline["index"].equals(candidate["index"]):
        raise ValueError("baseline and candidate sample indices do not match")

    baseline_summary = summarize_samples(baseline)
    candidate_summary = summarize_samples(candidate)
    metrics = {}
    for name, baseline_value in baseline_summary.items():
        candidate_value = candidate_summary[name]
        lower_is_better = name in ERROR_METRICS
        passed = (
            candidate_value <= baseline_value
            if lower_is_better
            else candidate_value >= baseline_value
        )
        improvement = (
            baseline_value - candidate_value
            if lower_is_better
            else candidate_value - baseline_value
        )
        metrics[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
            "improvement": improvement,
            "passed": bool(passed),
        }

    return {
        "baseline_csv": str(baseline_path),
        "candidate_csv": str(candidate_path),
        "sample_count": len(baseline),
        "eligible": all(metric["passed"] for metric in metrics.values()),
        "strictly_improved_count": sum(
            metric["improvement"] > 0 for metric in metrics.values()
        ),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_csv")
    parser.add_argument("candidate_csv")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    report = compare_csvs(args.baseline_csv, args.candidate_csv)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_path:
        Path(args.json_path).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Compare UAV2MAP paper-style R@d metrics with the published TopVeLoc row."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOPVELOC_BASELINE = {
    "longitudinal_recall_1m": 28.56,
    "longitudinal_recall_3m": 67.02,
    "longitudinal_recall_5m": 78.87,
    "longitudinal_recall_10m": 82.75,
    "lateral_recall_1m": 29.07,
    "lateral_recall_3m": 68.02,
    "lateral_recall_5m": 78.84,
    "lateral_recall_10m": 82.76,
    "position_recall_1m": 8.18,
    "position_recall_3m": 50.37,
    "position_recall_5m": 73.48,
    "position_recall_10m": 80.15,
}

ERROR_COLUMNS = {
    "longitudinal": "longitudinal_max_error_m",
    "lateral": "lateral_max_error_m",
    "position": "position_max_error_m",
}


def summarize_paper_recalls(frame: pd.DataFrame) -> dict[str, float]:
    required = {"index", *ERROR_COLUMNS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"paper evaluation CSV is missing columns: {sorted(missing)}")
    if frame["index"].duplicated().any():
        raise ValueError("paper evaluation CSV contains duplicate indices")
    values = frame[list(ERROR_COLUMNS.values())].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("paper evaluation CSV contains non-finite errors")

    result = {}
    for family, column in ERROR_COLUMNS.items():
        errors = frame[column].astype(float)
        for threshold in (1, 3, 5, 10):
            result[f"{family}_recall_{threshold}m"] = float(
                (errors < threshold).mean() * 100.0
            )
    return result


def compare_with_topveloc(candidate: dict[str, float]) -> dict[str, object]:
    if set(candidate) != set(TOPVELOC_BASELINE):
        missing = sorted(set(TOPVELOC_BASELINE).difference(candidate))
        extra = sorted(set(candidate).difference(TOPVELOC_BASELINE))
        raise ValueError(f"candidate metric mismatch: missing={missing}, extra={extra}")
    metrics = {}
    for name, baseline in TOPVELOC_BASELINE.items():
        value = float(candidate[name])
        improvement = value - baseline
        metrics[name] = {
            "baseline": baseline,
            "candidate": value,
            "improvement": improvement,
            "passed": improvement > 0,
        }
    return {
        "baseline": "TopVeLoc Table III",
        "eligible": all(item["passed"] for item in metrics.values()),
        "strictly_improved_count": sum(item["passed"] for item in metrics.values()),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_csv")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    candidate = summarize_paper_recalls(pd.read_csv(args.candidate_csv))
    report = compare_with_topveloc(candidate)
    report["candidate_csv"] = args.candidate_csv
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_path:
        Path(args.json_path).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

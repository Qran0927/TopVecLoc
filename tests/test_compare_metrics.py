from pathlib import Path

import pandas as pd
import pytest

from evaluation.compare_metrics import compare_csvs, summarize_samples


def _write(path: Path, position, yaw, indices=None):
    if indices is None:
        indices = list(range(len(position)))
    pd.DataFrame(
        {
            "index": indices,
            "pos_error_m": position,
            "yaw_error_deg": yaw,
        }
    ).to_csv(path, index=False)
    return path


def test_summarize_samples_returns_all_ten_metrics():
    frame = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "pos_error_m": [0.5, 2.0, 4.0, 6.0],
            "yaw_error_deg": [0.5, 2.0, 4.0, 6.0],
        }
    )

    summary = summarize_samples(frame)

    assert len(summary) == 10
    assert summary["position_error_mean"] == pytest.approx(3.125)
    assert summary["position_recall_1m"] == pytest.approx(0.25)
    assert summary["yaw_recall_5deg"] == pytest.approx(0.75)


def test_candidate_is_ineligible_when_one_recall_regresses(tmp_path):
    baseline = _write(
        tmp_path / "baseline.csv",
        [0.5, 2.0, 4.0, 6.0],
        [0.5, 2.0, 4.0, 6.0],
    )
    candidate = _write(
        tmp_path / "candidate.csv",
        [1.5, 1.5, 3.5, 5.5],
        [0.5, 2.0, 4.0, 6.0],
    )

    report = compare_csvs(baseline, candidate)

    assert report["eligible"] is False
    assert report["metrics"]["position_recall_1m"]["passed"] is False
    assert report["metrics"]["position_error_mean"]["passed"] is True


def test_equal_candidate_is_eligible(tmp_path):
    baseline = _write(tmp_path / "baseline.csv", [1.0, 2.0], [1.0, 2.0])
    candidate = _write(tmp_path / "candidate.csv", [1.0, 2.0], [1.0, 2.0])

    assert compare_csvs(baseline, candidate)["eligible"] is True


def test_comparison_rejects_different_sample_indices(tmp_path):
    baseline = _write(tmp_path / "baseline.csv", [1.0, 2.0], [1.0, 2.0])
    candidate = _write(
        tmp_path / "candidate.csv", [1.0, 2.0], [1.0, 2.0], indices=[0, 3]
    )

    with pytest.raises(ValueError, match="indices"):
        compare_csvs(baseline, candidate)

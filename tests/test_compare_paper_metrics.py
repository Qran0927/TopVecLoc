import pandas as pd
import pytest

from evaluation.compare_paper_metrics import (
    TOPVELOC_BASELINE,
    compare_with_topveloc,
    summarize_paper_recalls,
)


def test_summarize_paper_recalls_returns_all_twelve_metrics():
    frame = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "longitudinal_max_error_m": [0.5, 2.0, 4.0, 11.0],
            "lateral_max_error_m": [0.5, 2.0, 4.0, 11.0],
            "position_max_error_m": [0.5, 2.0, 4.0, 11.0],
        }
    )

    summary = summarize_paper_recalls(frame)

    assert len(summary) == 12
    assert summary["position_recall_1m"] == pytest.approx(25.0)
    assert summary["position_recall_5m"] == pytest.approx(75.0)
    assert summary["position_recall_10m"] == pytest.approx(75.0)


def test_topveloc_constants_match_paper_table():
    assert TOPVELOC_BASELINE["longitudinal_recall_1m"] == 28.56
    assert TOPVELOC_BASELINE["lateral_recall_3m"] == 68.02
    assert TOPVELOC_BASELINE["position_recall_10m"] == 80.15


def test_candidate_must_improve_all_twelve_metrics():
    candidate = {name: value + 0.01 for name, value in TOPVELOC_BASELINE.items()}
    report = compare_with_topveloc(candidate)
    assert report["eligible"] is True
    assert report["strictly_improved_count"] == 12

    candidate["position_recall_3m"] = TOPVELOC_BASELINE["position_recall_3m"]
    report = compare_with_topveloc(candidate)
    assert report["eligible"] is False
    assert report["metrics"]["position_recall_3m"]["passed"] is False

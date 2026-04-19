"""Unit tests for scripts/aggregate_grounding_results.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_grounding_results",
    ROOT / "scripts" / "aggregate_grounding_results.py",
)
agg = importlib.util.module_from_spec(SPEC)
sys.modules["aggregate_grounding_results"] = agg
SPEC.loader.exec_module(agg)


def _write_eval_json(path: Path, hits1: list[bool], hits5: list[bool], l2s: list[float]) -> None:
    """Write a synthetic eval_grounding.py-shaped JSON."""
    per_case = [
        {
            "query": f"q{i}",
            "hit@1": h1,
            "hit@5": h5,
            "rank_of_hit": 1 if h5 else None,
            "top1_centroid_l2_m": l2,
            "top1_score": 0.5,
            "latency_s": 0.05,
            "num_results": 5,
        }
        for i, (h1, h5, l2) in enumerate(zip(hits1, hits5, l2s, strict=True))
    ]
    summary = {
        "num_cases": len(per_case),
        "hit@1": sum(hits1) / len(hits1),
        "hit@5": sum(hits5) / len(hits5),
        "mean_top1_l2_m": sum(l2s) / len(l2s),
        "mean_top1_l2_m_hit_only": sum(l2s) / len(l2s),
        "mean_latency_s": 0.05,
    }
    path.write_text(json.dumps({"summary": summary, "per_case": per_case}) + "\n")


def test_aggregate_two_scenes_unweighted_mean(tmp_path: Path) -> None:
    j1 = tmp_path / "a.json"
    j2 = tmp_path / "b.json"
    _write_eval_json(j1, [True, True, False, False], [True, True, True, True], [0.1, 0.2, 0.3, 0.4])
    _write_eval_json(j2, [True, False], [True, True], [0.5, 0.6])

    out = agg.aggregate(
        scene_results=[("scene_a", j1), ("scene_b", j2)],
    )

    # per-scene
    assert out["per_scene"]["scene_a"]["hit@1"] == 0.5
    assert out["per_scene"]["scene_a"]["hit@5"] == 1.0
    assert out["per_scene"]["scene_a"]["n_queries"] == 4
    assert out["per_scene"]["scene_b"]["hit@1"] == 0.5
    assert out["per_scene"]["scene_b"]["n_queries"] == 2

    # aggregate is unweighted mean of per-scene metrics
    assert out["aggregate"]["mean_hit@1"] == 0.5  # (0.5 + 0.5) / 2
    assert out["aggregate"]["mean_hit@5"] == 1.0  # (1.0 + 1.0) / 2
    assert out["aggregate"]["n_scenes"] == 2
    assert out["aggregate"]["total_queries"] == 6


def test_aggregate_markdown_table_has_one_row_per_scene_plus_aggregate(tmp_path: Path) -> None:
    j1 = tmp_path / "a.json"
    j2 = tmp_path / "b.json"
    _write_eval_json(j1, [True], [True], [0.1])
    _write_eval_json(j2, [False], [True], [0.5])

    out = agg.aggregate(scene_results=[("scene_a", j1), ("scene_b", j2)])
    md = agg.format_markdown_table(out)

    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|") and "---" not in ln]
    # Header + 2 scene rows + aggregate row = 4
    assert len(lines) == 4
    assert "scene_a" in md
    assert "scene_b" in md
    assert "**aggregate**" in md

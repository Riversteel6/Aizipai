"""Lightweight action-value model tests."""

from pathlib import Path

import numpy as np

from ai.action_value_model import (
    ActionValueModel,
    evaluate_action_ranking,
    feature_matrix,
    fit_fixed_action_value_model,
)


def _row(action: dict) -> dict:
    return {
        "game": {
            "players": 3,
            "wildcard_enabled": True,
            "candidate_seat": 0,
            "dealer": 1,
        },
        "phase": "response_root",
        "turn": 8,
        "seat": 0,
        "action": action,
        "state_features": {
            "hand_counts": {"四": 1, "六": 1, "王": 1},
            "remaining_counts": {"四": 3, "五": 2, "六": 3},
            "own_meld_count": 1,
            "opponent_meld_counts": [2, 1],
            "all_melds": [
                [{"type": "peng", "labels": ["五", "五", "五"]}],
                [{"type": "wei", "labels": ["柒", "柒", "柒"]}],
                [],
            ],
            "discards": [["九"], ["贰", "拾"], ["一"]],
            "discard_counts": [3, 2, 4],
            "stock_count": 15,
            "hand_sizes": [12, 11, 10],
            "pending_card": "五",
            "pending_source_seat": 2,
        },
    }


def test_chi_options_have_distinct_action_value_features():
    first = _row(
        {
            "type": "CHI",
            "consumed_from_hand": ["四", "六"],
            "meld_groups": [["四", "五", "六"]],
        }
    )
    second = _row(
        {
            "type": "CHI",
            "consumed_from_hand": ["贰", "柒"],
            "meld_groups": [["贰", "柒", "拾"]],
        }
    )

    matrix, names = feature_matrix([first, second])

    assert matrix.shape == (2, len(names))
    assert not np.array_equal(matrix[0], matrix[1])


def test_features_do_not_encode_the_current_policy_choice():
    selected = _row({"type": "PASS"})
    selected["selected"] = True
    alternative = _row({"type": "PASS"})
    alternative["selected"] = False

    matrix, names = feature_matrix([selected, alternative])

    assert "production_selected" not in names
    assert np.array_equal(matrix[0], matrix[1])


def test_action_value_model_round_trips_without_prediction_drift(tmp_path: Path):
    rows = [
        _row({"type": "PASS"}),
        _row(
            {
                "type": "CHI",
                "consumed_from_hand": ["四", "六"],
                "meld_groups": [["四", "五", "六"]],
            }
        ),
    ]
    matrix, names = feature_matrix(rows)
    model = ActionValueModel(
        feature_names=names,
        feature_mean=np.zeros(matrix.shape[1]),
        feature_scale=np.ones(matrix.shape[1]),
        projection=np.ones((matrix.shape[1], 2)) * 0.01,
        projection_bias=np.zeros(2),
        output_weights=np.linspace(0.0, 1.0, matrix.shape[1] + 3),
        activation="tanh",
    )
    path = tmp_path / "model.npz"

    before = model.predict_rows(rows)
    model.save(path, metadata={"name": "test"})
    loaded, metadata = ActionValueModel.load(path)

    assert metadata == {"name": "test"}
    assert np.allclose(loaded.predict_rows(rows), before)


def test_ranking_evaluation_does_not_use_evidence_order_to_break_score_ties():
    rows = []
    for index, reward in enumerate((0.0, 1.0)):
        row = _row({"type": "PASS", "heuristic_value": 0.0})
        row.update(
            {
                "split": "test",
                "group_id": "same-decision",
                "selected": index == 0,
                "targets": {"mean_reward": reward},
            }
        )
        rows.append(row)

    report = evaluate_action_ranking(rows, [0.0, 0.0], split="test")

    assert report["raw_model"]["mean_reward"] == 0.5
    assert report["raw_model"]["mean_regret"] == 0.5
    assert report["raw_model"]["top1_accuracy"] == 0.5
    assert report["model"]["mean_reward"] == 0.0
    assert report["model_overrides"] == 0


def test_fixed_action_value_fit_ranks_the_selected_action() -> None:
    rows = []
    for group_index in range(8):
        for selected, label in ((True, "一"), (False, "十")):
            row = _row({"type": "DISCARD", "label": label})
            row.update(
                {
                    "group_id": f"group-{group_index}",
                    "selected": selected,
                    "targets": {"mean_reward": float(selected)},
                    "audit_health": {"paired_worlds": 8},
                }
            )
            rows.append(row)
    model, report = fit_fixed_action_value_model(
        rows,
        seed=7,
        hidden_dimension=8,
        ridge=0.01,
        activation="tanh",
    )
    scores = model.predict_rows(rows[:2])

    assert scores[0] > scores[1]
    assert report["decision_groups"] == 8

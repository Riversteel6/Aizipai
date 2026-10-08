"""Lightweight action-value model trained from full-game counterfactuals."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL


LABELS = (*SMALL_LABELS, *BIG_LABELS, WILD_LABEL)
MODES = ("2p_no_wang", "2p_wang", "3p_no_wang", "3p_wang")
PHASES = (
    "discard",
    "response_root",
    "self_hu",
    "post_auto_hu",
    "post_action_hu",
)
ACTION_TYPES = ("DISCARD", "CHI", "PENG", "HU", "PASS")
MELD_TYPES = (
    "wei",
    "peng",
    "mixed_same_rank_triplet",
    "special_123",
    "normal_sequence",
    "special_2710",
    "ti",
    "pao",
)


@dataclass(frozen=True)
class ActionValueModel:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    projection: np.ndarray
    projection_bias: np.ndarray
    output_weights: np.ndarray
    activation: str

    def predict_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        matrix, names = feature_matrix(rows)
        if names != self.feature_names:
            raise ValueError("action_value_feature_schema_mismatch")
        return self.predict_matrix(matrix)

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        normalized = (matrix - self.feature_mean) / self.feature_scale
        hidden = _activation(
            normalized @ self.projection + self.projection_bias,
            self.activation,
        )
        design = np.column_stack(
            (
                np.ones(len(normalized), dtype=np.float64),
                normalized,
                hidden,
            )
        )
        return design @ self.output_weights

    def save(self, path: Path, *, metadata: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_names=np.asarray(self.feature_names),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            projection=self.projection,
            projection_bias=self.projection_bias,
            output_weights=self.output_weights,
            activation=np.asarray(self.activation),
            metadata=np.asarray(
                json.dumps(dict(metadata), ensure_ascii=False)
            ),
        )

    @classmethod
    def load(cls, path: Path) -> tuple["ActionValueModel", dict[str, Any]]:
        with np.load(path, allow_pickle=False) as payload:
            model = cls(
                feature_names=tuple(
                    str(value) for value in payload["feature_names"]
                ),
                feature_mean=payload["feature_mean"].astype(np.float64),
                feature_scale=payload["feature_scale"].astype(np.float64),
                projection=payload["projection"].astype(np.float64),
                projection_bias=payload["projection_bias"].astype(np.float64),
                output_weights=payload["output_weights"].astype(np.float64),
                activation=str(payload["activation"]),
            )
            metadata = json.loads(str(payload["metadata"]))
        return model, metadata


def feature_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    encoded = [encode_action_value_features(row) for row in rows]
    if not encoded:
        return np.empty((0, 0), dtype=np.float64), ()
    names = tuple(name for name, _value in encoded[0])
    matrix = np.asarray(
        [[value for _name, value in features] for features in encoded],
        dtype=np.float64,
    )
    if any(
        tuple(name for name, _value in features) != names
        for features in encoded[1:]
    ):
        raise ValueError("inconsistent_action_value_feature_schema")
    return matrix, names


def encode_action_value_features(
    row: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    game = dict(row.get("game") or {})
    state = dict(row.get("state_features") or {})
    action = dict(row.get("action") or {})
    hand = _label_counts(state.get("hand_counts"))
    remaining = _label_counts(state.get("remaining_counts"))
    consumed = Counter(str(label) for label in action.get("consumed_from_hand") or ())
    meld_cards = Counter(
        str(label)
        for group in action.get("meld_groups") or ()
        for label in group
    )
    action_type = str(action.get("type") or "")
    action_label = str(action.get("label") or "")
    followup = str(action.get("followup_discard") or "")
    pending = str(state.get("pending_card") or "")
    all_melds = list(state.get("all_melds") or ())
    discards = list(state.get("discards") or ())
    post_hand = Counter(hand)
    post_hand.subtract(consumed)
    if action_type == "DISCARD" and action_label:
        post_hand[action_label] -= 1
    mode = _mode_key(game)
    values: list[tuple[str, float]] = []

    def add(name: str, value: float | int | bool) -> None:
        values.append((name, float(value)))

    for label in LABELS:
        add(f"hand:{label}", hand[label] / 4.0)
    for label in LABELS:
        add(f"remaining:{label}", remaining[label] / 4.0)
    for label in LABELS:
        add(f"consumed:{label}", consumed[label] / 4.0)
    for label in LABELS:
        add(f"meld_cards:{label}", meld_cards[label] / 4.0)
    for label in LABELS:
        add(f"post_hand:{label}", max(0, post_hand[label]) / 4.0)
    _add_one_hot(values, "mode", mode, MODES)
    _add_one_hot(values, "phase", str(row.get("phase") or ""), PHASES)
    _add_one_hot(values, "action", action_type, ACTION_TYPES)
    _add_one_hot(values, "pending", pending, LABELS, include_none=True)
    _add_one_hot(
        values,
        "action_label",
        action_label,
        LABELS,
        include_none=True,
    )
    for seat in range(3):
        seat_melds = all_melds[seat] if seat < len(all_melds) else ()
        seat_meld_labels = Counter(
            str(label)
            for meld in seat_melds or ()
            for label in dict(meld).get("labels") or ()
        )
        seat_meld_types = Counter(
            str(dict(meld).get("type") or "")
            for meld in seat_melds or ()
        )
        seat_discards = [
            str(label)
            for label in (discards[seat] if seat < len(discards) else ())
        ]
        seat_discard_counts = Counter(seat_discards)
        for label in LABELS:
            add(
                f"seat_meld_label:{seat}:{label}",
                seat_meld_labels[label] / 4.0,
            )
        for meld_type in MELD_TYPES:
            add(
                f"seat_meld_type:{seat}:{meld_type}",
                seat_meld_types[meld_type] / 7.0,
            )
        for label in LABELS:
            add(
                f"seat_discard_label:{seat}:{label}",
                seat_discard_counts[label] / 4.0,
            )
        for recent_index in range(3):
            recent = (
                seat_discards[-(recent_index + 1)]
                if len(seat_discards) > recent_index
                else ""
            )
            _add_one_hot(
                values,
                f"seat_recent_discard:{seat}:{recent_index}",
                recent,
                LABELS,
                include_none=True,
            )
    _add_one_hot(
        values,
        "followup",
        followup,
        LABELS,
        include_none=True,
    )

    opponent_melds = _pad_numeric(state.get("opponent_meld_counts"), 2)
    discard_counts = _pad_numeric(state.get("discard_counts"), 3)
    hand_sizes = _pad_numeric(state.get("hand_sizes"), 3)
    add("turn", min(80, int(row.get("turn") or 0)) / 80.0)
    add("stock_count", int(state.get("stock_count") or 0) / 43.0)
    add("own_meld_count", int(state.get("own_meld_count") or 0) / 7.0)
    for index, value in enumerate(opponent_melds):
        add(f"opponent_meld_count:{index}", value / 7.0)
    for index, value in enumerate(discard_counts):
        add(f"discard_count:{index}", value / 30.0)
    for index, value in enumerate(hand_sizes):
        add(f"hand_size:{index}", value / 21.0)
    add(
        "is_dealer",
        int(game.get("candidate_seat") or 0)
        == int(game.get("dealer") or 0),
    )
    add("seat", int(row.get("seat") or 0) / 2.0)
    add("pending_source_known", state.get("pending_source_seat") is not None)
    add("consumed_count", sum(consumed.values()) / 6.0)
    add("meld_group_count", len(action.get("meld_groups") or ()) / 2.0)
    add("followup_present", bool(followup))
    add("heuristic_value", float(action.get("heuristic_value") or 0.0) / 1500.0)
    add("action_hand_count", hand[action_label] / 4.0)
    add("action_remaining_count", remaining[action_label] / 4.0)
    add("action_matches_pending", bool(action_label and action_label == pending))
    add(
        "consumed_red_fraction",
        _red_fraction(consumed),
    )
    add(
        "meld_red_fraction",
        _red_fraction(meld_cards),
    )
    return tuple(values)


def train_action_value_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260728,
    hidden_dimensions: Sequence[int] = (64, 128),
    ridge_values: Sequence[float] = (0.1, 1.0, 10.0),
    activations: Sequence[str] = ("relu", "tanh"),
) -> tuple[ActionValueModel, dict[str, Any]]:
    if not rows:
        raise ValueError("action_value_training_rows_required")
    matrix, feature_names = feature_matrix(rows)
    targets = np.asarray(
        [
            float(dict(row.get("targets") or {}).get("mean_reward") or 0.0)
            for row in rows
        ],
        dtype=np.float64,
    )
    split_indices = {
        split: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row.get("split") == split
            ],
            dtype=np.int64,
        )
        for split in ("train", "validation", "test")
    }
    if any(not len(indices) for indices in split_indices.values()):
        raise ValueError("train_validation_test_rows_required")
    train_indices = split_indices["train"]
    feature_mean = matrix[train_indices].mean(axis=0)
    feature_scale = matrix[train_indices].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (matrix - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    maximum_hidden = max(int(value) for value in hidden_dimensions)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(max(1, normalized.shape[1])),
        size=(normalized.shape[1], maximum_hidden),
    )
    projection_bias = rng.uniform(-0.5, 0.5, size=maximum_hidden)
    sample_weights = _group_balanced_weights(rows, train_indices)
    candidates: list[dict[str, Any]] = []
    best: tuple[
        tuple[float, float, float],
        ActionValueModel,
        dict[str, Any],
    ] | None = None
    for activation in activations:
        hidden_all = _activation(
            normalized @ projection + projection_bias,
            activation,
        )
        for hidden_dimension in hidden_dimensions:
            width = int(hidden_dimension)
            design_all = np.column_stack(
                (
                    np.ones(len(rows), dtype=np.float64),
                    normalized,
                    hidden_all[:, :width],
                )
            )
            centered_design, centered_targets = _center_training_groups(
                design_all,
                targets,
                rows,
                train_indices,
            )
            for ridge in ridge_values:
                output_weights = _weighted_ridge(
                    centered_design,
                    centered_targets,
                    sample_weights,
                    float(ridge),
                )
                predictions = design_all @ output_weights
                override_thresholds = tune_override_thresholds(
                    rows,
                    predictions,
                    split="validation",
                )
                validation = evaluate_action_ranking(
                    rows,
                    predictions,
                    split="validation",
                    override_thresholds=override_thresholds,
                )
                candidate = {
                    "activation": activation,
                    "hidden_dimension": width,
                    "ridge": float(ridge),
                    "override_thresholds": override_thresholds,
                    "validation": validation,
                }
                candidates.append(candidate)
                model = ActionValueModel(
                    feature_names=feature_names,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    projection=projection[:, :width],
                    projection_bias=projection_bias[:width],
                    output_weights=output_weights,
                    activation=activation,
                )
                rank = (
                    float(validation["model"]["mean_regret"]),
                    -float(validation["model"]["top1_accuracy"]),
                    float(validation["model"]["rmse"]),
                )
                if best is None or rank < best[0]:
                    best = (rank, model, candidate)
    if best is None:
        raise RuntimeError("no_action_value_model_candidate")
    model = best[1]
    selected_candidate = best[2]
    override_thresholds = dict(selected_candidate["override_thresholds"])
    predictions = model.predict_matrix(matrix)
    metrics = {
        split: evaluate_action_ranking(
            rows,
            predictions,
            split=split,
            override_thresholds=override_thresholds,
        )
        for split in ("train", "validation", "test")
    }
    report = {
        "schema_version": "action-value-model-training-v1",
        "seed": seed,
        "features": len(feature_names),
        "samples": len(rows),
        "decision_groups": len({str(row.get("group_id")) for row in rows}),
        "selected_config": {
            "activation": model.activation,
            "hidden_dimension": model.projection.shape[1],
            "ridge": selected_candidate["ridge"],
            "override_thresholds": override_thresholds,
        },
        "candidate_count": len(candidates),
        "training_objective": "within_decision_centered_action_advantage",
        "candidates": candidates,
        "metrics": metrics,
        "accepted_as_search_prior": _prior_acceptance(metrics),
    }
    return model, report


def fit_fixed_action_value_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    hidden_dimension: int,
    ridge: float,
    activation: str,
    training_indices: Sequence[int] | None = None,
) -> tuple[ActionValueModel, dict[str, Any]]:
    """Fit one preregistered model specification without model selection."""
    if not rows:
        raise ValueError("fixed_action_value_training_rows_required")
    matrix, feature_names = feature_matrix(rows)
    indices = np.asarray(
        list(training_indices) if training_indices is not None else range(len(rows)),
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError("fixed_action_value_training_indices_required")
    targets = np.asarray(
        [
            float(dict(row.get("targets") or {}).get("mean_reward") or 0.0)
            for row in rows
        ],
        dtype=np.float64,
    )
    feature_mean = matrix[indices].mean(axis=0)
    feature_scale = matrix[indices].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (matrix - feature_mean) / feature_scale
    resolved_hidden = max(1, int(hidden_dimension))
    rng = np.random.default_rng(int(seed))
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(max(1, normalized.shape[1])),
        size=(normalized.shape[1], resolved_hidden),
    )
    projection_bias = rng.uniform(-0.5, 0.5, size=resolved_hidden)
    hidden = _activation(
        normalized @ projection + projection_bias,
        activation,
    )
    design = np.column_stack(
        (np.ones(len(rows), dtype=np.float64), normalized, hidden)
    )
    centered_design, centered_targets = _center_training_groups(
        design,
        targets,
        rows,
        indices,
    )
    weights = _group_balanced_weights(rows, indices)
    output_weights = _weighted_ridge(
        centered_design,
        centered_targets,
        weights,
        float(ridge),
    )
    model = ActionValueModel(
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        projection=projection,
        projection_bias=projection_bias,
        output_weights=output_weights,
        activation=activation,
    )
    return model, {
        "schema_version": "fixed-action-value-model-fit-v1",
        "seed": int(seed),
        "features": len(feature_names),
        "samples": len(indices),
        "decision_groups": len(
            {str(rows[index].get("group_id")) for index in indices}
        ),
        "activation": activation,
        "hidden_dimension": resolved_hidden,
        "ridge": float(ridge),
        "training_objective": "within_decision_centered_action_advantage",
    }


def evaluate_action_ranking(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
    *,
    split: str,
    override_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if row.get("split") == split:
            grouped[str(row.get("group_id"))].append((row, float(prediction)))
    model_records = []
    raw_model_records = []
    selected_records = []
    heuristic_records = []
    model_by_mode: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    selected_by_mode: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    heuristic_by_mode: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    model_by_phase: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    selected_by_phase: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    heuristic_by_phase: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    squared_errors: list[float] = []
    model_overrides = 0
    for group_rows in grouped.values():
        actual = np.asarray(
            [
                float(dict(row.get("targets") or {}).get("mean_reward") or 0.0)
                for row, _prediction in group_rows
            ]
        )
        predicted = np.asarray(
            [prediction for _row, prediction in group_rows]
        )
        heuristic = np.asarray(
            [
                float(dict(row.get("action") or {}).get("heuristic_value") or 0.0)
                for row, _prediction in group_rows
            ]
        )
        best_value = float(actual.max())
        selected_index = next(
            (
                index
                for index, (row, _prediction) in enumerate(group_rows)
                if row.get("selected")
            ),
            0,
        )
        raw_model_index = int(np.argmax(predicted))
        phase = str(group_rows[0][0].get("phase") or "")
        mode = _mode_key(dict(group_rows[0][0].get("game") or {}))
        threshold_key = f"{mode}:{phase}"
        threshold = float(
            dict(override_thresholds or {}).get(
                threshold_key,
                1_000_000_000.0,
            )
        )
        model_index = selected_index
        if (
            raw_model_index != selected_index
            and predicted[raw_model_index] - predicted[selected_index]
            >= threshold
        ):
            model_index = raw_model_index
            model_overrides += 1
        raw_model_record = _score_choice_record(
            actual,
            predicted,
            best_value,
        )
        model_record = _choice_record(actual, model_index, best_value)
        selected_record = _choice_record(actual, selected_index, best_value)
        heuristic_record = _score_choice_record(
            actual,
            heuristic,
            best_value,
        )
        model_records.append(model_record)
        raw_model_records.append(raw_model_record)
        selected_records.append(selected_record)
        heuristic_records.append(heuristic_record)
        model_by_mode[mode].append(model_record)
        selected_by_mode[mode].append(selected_record)
        heuristic_by_mode[mode].append(heuristic_record)
        model_by_phase[phase].append(model_record)
        selected_by_phase[phase].append(selected_record)
        heuristic_by_phase[phase].append(heuristic_record)
        squared_errors.extend((predicted - actual) ** 2)
    return {
        "groups": len(grouped),
        "model": _summarize_choices(model_records, squared_errors),
        "raw_model": _summarize_choices(raw_model_records, squared_errors),
        "model_overrides": model_overrides,
        "override_thresholds": dict(override_thresholds or {}),
        "current_policy": _summarize_choices(selected_records),
        "heuristic": _summarize_choices(heuristic_records),
        "model_by_mode": {
            mode: _summarize_choices(records)
            for mode, records in sorted(model_by_mode.items())
        },
        "current_policy_by_mode": {
            mode: _summarize_choices(records)
            for mode, records in sorted(selected_by_mode.items())
        },
        "heuristic_by_mode": {
            mode: _summarize_choices(records)
            for mode, records in sorted(heuristic_by_mode.items())
        },
        "model_by_phase": {
            phase: _summarize_choices(records)
            for phase, records in sorted(model_by_phase.items())
        },
        "current_policy_by_phase": {
            phase: _summarize_choices(records)
            for phase, records in sorted(selected_by_phase.items())
        },
        "heuristic_by_phase": {
            phase: _summarize_choices(records)
            for phase, records in sorted(heuristic_by_phase.items())
        },
    }


def tune_override_thresholds(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
    *,
    split: str,
) -> dict[str, float]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if row.get("split") == split:
            grouped[str(row.get("group_id"))].append((row, float(prediction)))
    by_mode_phase: dict[
        str,
        list[tuple[float, float, float]],
    ] = defaultdict(list)
    for group_rows in grouped.values():
        actual = np.asarray(
            [
                float(dict(row.get("targets") or {}).get("mean_reward") or 0.0)
                for row, _prediction in group_rows
            ]
        )
        predicted = np.asarray(
            [prediction for _row, prediction in group_rows]
        )
        selected_index = next(
            (
                index
                for index, (row, _prediction) in enumerate(group_rows)
                if row.get("selected")
            ),
            0,
        )
        model_index = int(np.argmax(predicted))
        phase = str(group_rows[0][0].get("phase") or "")
        mode = _mode_key(dict(group_rows[0][0].get("game") or {}))
        by_mode_phase[f"{mode}:{phase}"].append(
            (
                float(predicted[model_index] - predicted[selected_index]),
                float(actual[model_index]),
                float(actual[selected_index]),
            )
        )
    thresholds: dict[str, float] = {}
    never_override = 1_000_000_000.0
    for mode_phase, records in by_mode_phase.items():
        advantages = sorted(
            {
                max(0.0, advantage)
                for advantage, model_reward, selected_reward in records
                if advantage > 0.0 and model_reward != selected_reward
            }
        )
        candidates = [0.0, *advantages, never_override]
        best: tuple[float, int, float] | None = None
        best_threshold = never_override
        for threshold in candidates:
            regrets = []
            overrides = 0
            for advantage, model_reward, selected_reward in records:
                chosen = selected_reward
                if advantage >= threshold and advantage > 0.0:
                    chosen = model_reward
                    overrides += 1
                regrets.append(max(model_reward, selected_reward) - chosen)
            rank = (
                float(np.mean(regrets)) if regrets else 0.0,
                overrides,
                -threshold,
            )
            if best is None or rank < best:
                best = rank
                best_threshold = threshold
        thresholds[mode_phase] = round(float(best_threshold), 12)
    return thresholds


def _weighted_ridge(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    weighted_targets = targets * root_weights
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    return np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets,
    )


def _center_training_groups(
    design: np.ndarray,
    targets: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered_design = design[indices].copy()
    centered_targets = targets[indices].copy()
    positions: dict[str, list[int]] = defaultdict(list)
    for position, row_index in enumerate(indices):
        positions[str(rows[row_index].get("group_id"))].append(position)
    for group_positions in positions.values():
        centered_design[group_positions] -= centered_design[
            group_positions
        ].mean(axis=0)
        centered_targets[group_positions] -= centered_targets[
            group_positions
        ].mean()
    return centered_design, centered_targets


def _group_balanced_weights(
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> np.ndarray:
    counts = Counter(str(rows[index].get("group_id")) for index in indices)
    quality_by_group = {
        str(rows[index].get("group_id")): np.sqrt(
            max(
                1.0,
                float(
                    dict(rows[index].get("audit_health") or {}).get(
                        "paired_worlds"
                    )
                    or 1
                )
                / 8.0,
            )
        )
        for index in indices
    }
    weights = np.asarray(
        [
            quality_by_group[str(rows[index].get("group_id"))]
            / counts[str(rows[index].get("group_id"))]
            for index in indices
        ],
        dtype=np.float64,
    )
    return weights * (len(weights) / weights.sum())


def _activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "tanh":
        return np.tanh(values)
    raise ValueError(f"unknown_action_value_activation:{name}")


def _choice_record(
    actual: np.ndarray,
    index: int,
    best_value: float,
) -> tuple[float, float, float]:
    chosen = float(actual[index])
    return (
        chosen,
        max(0.0, best_value - chosen),
        float(np.isclose(chosen, best_value, atol=1e-9)),
    )


def _score_choice_record(
    actual: np.ndarray,
    scores: np.ndarray,
    best_value: float,
) -> tuple[float, float, float]:
    tied = np.flatnonzero(
        np.isclose(scores, float(scores.max()), atol=1e-12)
    )
    chosen = float(actual[tied].mean())
    top1_probability = float(
        np.isclose(actual[tied], best_value, atol=1e-9).mean()
    )
    return (
        chosen,
        max(0.0, best_value - chosen),
        top1_probability,
    )


def _summarize_choices(
    records: Sequence[tuple[float, float, float]],
    squared_errors: Sequence[float] = (),
) -> dict[str, Any]:
    if not records:
        return {
            "groups": 0,
            "mean_reward": 0.0,
            "mean_regret": 0.0,
            "p95_regret": 0.0,
            "top1_accuracy": 0.0,
            "rmse": 0.0,
        }
    array = np.asarray(records, dtype=np.float64)
    return {
        "groups": len(records),
        "mean_reward": round(float(array[:, 0].mean()), 6),
        "mean_regret": round(float(array[:, 1].mean()), 6),
        "p95_regret": round(float(np.percentile(array[:, 1], 95)), 6),
        "top1_accuracy": round(float(array[:, 2].mean()), 6),
        "rmse": round(
            float(np.sqrt(np.mean(squared_errors)))
            if squared_errors
            else 0.0,
            6,
        ),
    }


def _prior_acceptance(metrics: Mapping[str, Any]) -> bool:
    for split in ("validation", "test"):
        split_metrics = dict(metrics.get(split) or {})
        model = dict(split_metrics.get("model") or {})
        current = dict(split_metrics.get("current_policy") or {})
        if not model or not current:
            return False
        if float(model["mean_regret"]) >= float(current["mean_regret"]):
            return False
        mode_metrics = dict(split_metrics.get("model_by_mode") or {})
        current_by_mode = dict(
            split_metrics.get("current_policy_by_mode") or {}
        )
        if set(mode_metrics) != set(MODES) or set(current_by_mode) != set(MODES):
            return False
        if any(
            float(mode_metrics[mode]["mean_regret"])
            > float(current_by_mode[mode]["mean_regret"])
            for mode in MODES
        ):
            return False
    return True


def _add_one_hot(
    output: list[tuple[str, float]],
    prefix: str,
    value: str,
    options: Sequence[str],
    *,
    include_none: bool = False,
) -> None:
    for option in options:
        output.append((f"{prefix}:{option}", float(value == option)))
    if include_none:
        output.append((f"{prefix}:none", float(not value)))


def _label_counts(payload: Any) -> Counter[str]:
    if isinstance(payload, Mapping):
        return Counter(
            {
                str(label): int(amount)
                for label, amount in payload.items()
            }
        )
    return Counter()


def _pad_numeric(payload: Any, size: int) -> tuple[float, ...]:
    values = [float(value) for value in payload or ()][:size]
    values.extend([0.0] * (size - len(values)))
    return tuple(values)


def _red_fraction(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    red = sum(
        amount
        for label, amount in counts.items()
        if label in {"二", "七", "十", "贰", "柒", "拾"}
    )
    return red / total


def _mode_key(game: Mapping[str, Any]) -> str:
    players = int(game.get("players") or 0)
    wildcard = "wang" if game.get("wildcard_enabled") else "no_wang"
    return f"{players}p_{wildcard}"


__all__ = [
    "ACTION_TYPES",
    "ActionValueModel",
    "encode_action_value_features",
    "evaluate_action_ranking",
    "feature_matrix",
    "fit_fixed_action_value_model",
    "tune_override_thresholds",
    "train_action_value_model",
]

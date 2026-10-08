"""Frozen rule configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = WORKSPACE_ROOT / "config" / "rules.yaml"


DEFAULT_XI = {
    "ti": {"small": 9, "big": 12},
    "pao": {"small": 6, "big": 9},
    "wei": {"small": 3, "big": 6},
    "peng": {"small": 1, "big": 3},
    "special_123": {"small": 3, "big": 6},
    "special_2710": {"small": 3, "big": 6},
}

DEFAULT_WEIGHTS = {
    "direct_hu": 10000,
    "wildcard_keep": 100,
    "reach_min_xi": 500,
    "special_2710_small": 120,
    "special_2710_big": 160,
    "special_123_small": 100,
    "special_123_big": 140,
    "pair": 40,
    "red_card": 25,
    "big_card": 10,
    "normal_sequence_potential": 15,
    "isolated_penalty": -30,
    "break_pair_penalty": -60,
    "break_triplet_penalty": -180,
    "break_exact_quad_penalty": -420,
    "break_mixed_same_rank_triplet_penalty": -320,
    "break_special_2710_penalty": -260,
    "break_special_123_penalty": -220,
    "break_sequence_penalty": -150,
    "break_unknown_complete_meld_penalty": -80,
    "break_2710_penalty": -120,
    "break_123_penalty": -100,
    "discard_wildcard_penalty": -500,
    "late_red_danger_penalty": -80,
    "possible_hu_danger_penalty": -300,
    "possible_pao_danger_penalty": -200,
    "break_wildcard_related_penalty": -150,
    "immediate_hu": 10000,
    "xi_enough": 1000,
    "xi_per_point": 40,
    "ting_ready": 500,
    "ting_improvement": 120,
    "wildcard_discard_penalty": 10000,
    "wildcard_use_low_value_penalty": 200,
    "hard_protected_break_penalty": 10000,
    "exact_triplet_break_penalty": 3000,
    "exact_quad_break_penalty": 5000,
    "mixed_same_rank_break_penalty": 1000,
    "special_2710_break_penalty": 1200,
    "special_123_break_penalty": 1000,
    "normal_sequence_break_penalty": 500,
    "pair_break_penalty": 300,
    "orphan_discard_bonus": 300,
    "weak_potential_discard_bonus": 120,
    "free_card_discard_bonus": 200,
    "red_card_keep_bonus": 80,
    "red_card_late_danger": 180,
    "danger_base_penalty": 100,
    "opponent_pao_risk_penalty": 500,
    "opponent_hu_risk_penalty": 1000,
    "chi_min_ev_gain": 20,
    "peng_min_ev_gain": 20,
    "information_set_hu_draw_value": 1000,
    "information_set_hu_score_scale": 0.2,
    "information_set_improve_draw_value": 220,
}

DEFAULT_STRATEGY = {
    "hard_protection_enabled": True,
    "soft_protection_enabled": True,
    "resource_allocation_enabled": True,
    "local_runtime_only": True,
}

DEFAULT_CHI = {
    "enabled": True,
    "pass_if_uncertain": True,
    "min_confidence": 0.75,
    "min_ev_gain": 20,
    "allow_break_soft_protection_if_gain": 80,
    "forbid_break_hard_protection": True,
    "forbid_use_wildcard_unless_immediate_hu_or_big_gain": True,
    "simulate_followup_discard": True,
}

DEFAULT_PENG = {
    "min_ev_gain": 20,
    "simulate_followup_discard": True,
    "pass_if_uncertain": True,
}

DEFAULT_SAFETY = {
    "forbid_action_plan_reselect": True,
    "halt_on_conflict": True,
    "halt_on_uncertain_recognition": True,
    "halt_if_only_hard_protected_clickable_but_free_cards_exist": True,
}

DEFAULT_LOGGING = {
    "enabled": True,
    "save_raw_screenshot": True,
    "save_debug_screenshot": True,
    "save_every_frame": False,
    "save_decision_frames": True,
    "save_error_frames": True,
    "save_before_after_action": True,
    "max_sessions_keep": 20,
    "compress_round_on_end": True,
}

SUPPORTED_PLAYER_COUNTS = frozenset({2, 3})
ROOM_MODE_PRESETS: dict[str, tuple[int, bool]] = {
    "1v1-no-wang": (2, False),
    "1v1-wang": (2, True),
    "1v1v1-no-wang": (3, False),
    "1v1v1-wang": (3, True),
}


@dataclass(frozen=True)
class RuleConfig:
    raw: dict[str, Any]

    @property
    def min_xi(self) -> int:
        return int(self.raw.get("rules", {}).get("min_xi", 9))

    @property
    def wildcard_label(self) -> str:
        wildcard = self.raw.get("wildcard", {})
        return str(wildcard.get("label") or wildcard.get("name") or "王")

    @property
    def red_black_mode(self) -> str:
        return str(self.raw.get("rules", {}).get("red_black_mode", "red_black_mingtang"))

    @property
    def xi_to_tun(self) -> str:
        return str(self.raw.get("rules", {}).get("xi_to_tun", "3_to_1"))

    @property
    def base_tun_at_9_xi(self) -> int:
        return int(self.raw.get("rules", {}).get("base_tun_at_9_xi", 1))

    @property
    def zimo_double(self) -> bool:
        return bool(self.raw.get("rules", {}).get("zimo_double", False))

    @property
    def piao_mode(self) -> str:
        return str(self.raw.get("rules", {}).get("piao_mode", "none"))

    @property
    def weights(self) -> dict[str, int | float]:
        return self.raw.get("weights") or self.raw.get("ai_weights") or {}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_PATH) -> "RuleConfig":
        return cls(load_rules(path))


@lru_cache(maxsize=8)
def load_rules(path: str | Path = "config/rules.yaml") -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = WORKSPACE_ROOT / resolved
    return normalize_rules(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {})


def normalize_rules(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("_normalized_rules") is True:
        return raw
    data = deepcopy(raw)
    data.setdefault("xi", DEFAULT_XI)
    for key, value in DEFAULT_XI.items():
        data["xi"].setdefault(key, value)
    data.setdefault("ai_weights", DEFAULT_WEIGHTS)
    for key, value in DEFAULT_WEIGHTS.items():
        data["ai_weights"].setdefault(key, value)
    data.setdefault("rules", {})
    data["rules"].setdefault("min_xi", 9)
    data["rules"].setdefault("required_meld_groups", 7)
    data["rules"].setdefault("allow_1510", False)
    data["rules"].setdefault("red_black_mode", "red_black_mingtang")
    data.setdefault("wildcard", {})
    data["wildcard"].setdefault("enabled", True)
    data["wildcard"].setdefault("name", "王")
    data["wildcard"].setdefault("label", data["wildcard"].get("name", "王"))
    data["wildcard"].setdefault("copies", data.get("game", {}).get("deck_copies", 4))
    for action in ("chi", "peng", "wei", "pao", "ti"):
        data["wildcard"].setdefault(f"can_form_{action}", False)
    data.setdefault("scoring", {})
    data["scoring"].setdefault("red_hu_min_red", 13)
    data["scoring"].setdefault("black_hu_red_count", 0)
    data["scoring"].setdefault("one_red_hu_red_count", 1)
    data["scoring"].setdefault("red_black_special_point", 1)
    data["scoring"].setdefault("red_black_point_multiplier", 1)
    data.setdefault("ting", {})
    data["ting"].setdefault("max_exact_draw_enum_cards", 21)
    data.setdefault("strategy", {})
    for key, value in DEFAULT_STRATEGY.items():
        data["strategy"].setdefault(key, value)
    data.setdefault("chi", {})
    for key, value in DEFAULT_CHI.items():
        data["chi"].setdefault(key, value)
    data.setdefault("peng", {})
    for key, value in DEFAULT_PENG.items():
        data["peng"].setdefault(key, value)
    data.setdefault("safety", {})
    for key, value in DEFAULT_SAFETY.items():
        data["safety"].setdefault(key, value)
    data.setdefault("logging", {})
    for key, value in DEFAULT_LOGGING.items():
        data["logging"].setdefault(key, value)
    data.setdefault("weights", {})
    for key, value in DEFAULT_WEIGHTS.items():
        data["weights"].setdefault(key, value)
    data["_normalized_rules"] = True
    return data


def rules_for_room(
    path: str | Path = "config/rules.yaml",
    *,
    wildcard_enabled: bool | None = None,
    players: int | None = None,
    room_mode: str | None = None,
) -> dict[str, Any]:
    """Return an isolated 9-xi red/black room profile with room switches applied."""
    data = deepcopy(load_rules(path))
    if room_mode is not None:
        try:
            preset_players, preset_wildcard = ROOM_MODE_PRESETS[room_mode]
        except KeyError as exc:
            raise ValueError(f"unknown room mode: {room_mode}") from exc
        if players is not None and int(players) != preset_players:
            raise ValueError(
                f"room mode {room_mode} conflicts with players={players}"
            )
        if wildcard_enabled is not None and bool(wildcard_enabled) != preset_wildcard:
            raise ValueError(
                f"room mode {room_mode} conflicts with wildcard_enabled={wildcard_enabled}"
            )
        players = preset_players
        wildcard_enabled = preset_wildcard
    game = data.setdefault("game", {})
    player_count = int(game.get("players", 3) if players is None else players)
    if player_count not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError(f"unsupported player count: {player_count}; expected 2 or 3")
    game["players"] = player_count
    data.setdefault("rules", {})["min_xi"] = 9
    data["rules"]["required_meld_groups"] = 7
    data["rules"]["red_black_mode"] = "red_black_mingtang"
    if wildcard_enabled is not None:
        data.setdefault("wildcard", {})["enabled"] = bool(wildcard_enabled)
    resolved_wildcard = bool(data.get("wildcard", {}).get("enabled", False))
    resolved_mode = next(
        mode
        for mode, preset in ROOM_MODE_PRESETS.items()
        if preset == (player_count, resolved_wildcard)
    )
    data["room_profile"] = {
        "mode": resolved_mode,
        "players": player_count,
        "min_xi": 9,
        "red_black": True,
        "wildcard_enabled": resolved_wildcard,
    }
    return data


def min_xi(config: dict[str, Any]) -> int:
    return int(config.get("rules", {}).get("min_xi", 9))


def weights(config: dict[str, Any]) -> dict[str, int | float]:
    return config["ai_weights"]

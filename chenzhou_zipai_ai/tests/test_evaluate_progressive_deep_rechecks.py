import gzip
import json

from tools.evaluate_progressive_deep_rechecks import (
    _candidate_key,
    _deep_override_advantage,
    _load_cases,
    _progressive_search,
    _response_candidate,
    _summarize,
)


def test_load_cases_deduplicates_same_public_decision_by_deepest_evidence(
    tmp_path,
):
    evidence = tmp_path / "evidence.jsonl.gz"
    base = {
        "complete": True,
        "state_before_hash": "state-1",
        "phase": "discard",
        "seat": 0,
        "selected_key": "DISCARD:二",
        "best_key": "DISCARD:三",
        "legal_action_count": 2,
        "public_state": {"seat": 0, "hand": ["二", "三"]},
        "candidate_stats": [
            {"label": "二", "average_reward": 0.0},
            {"label": "三", "average_reward": 1.0},
        ],
        "game": {"players": 2, "wildcard_enabled": False},
    }
    with gzip.open(evidence, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({**base, "completed_paired_worlds": 24}) + "\n"
        )
        handle.write(
            json.dumps({**base, "completed_paired_worlds": 64}) + "\n"
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"entries": [{"output_evidence": str(evidence)}]}
        ),
        encoding="utf-8",
    )

    cases, source_rows = _load_cases(manifest)

    assert source_rows == 2
    assert len(cases) == 1
    assert cases[0]["completed_paired_worlds"] == 64


def test_load_cases_prefers_deeper_extra_evidence(tmp_path):
    base = {
        "complete": True,
        "state_before_hash": "state-1",
        "phase": "discard",
        "seat": 0,
        "selected_key": "DISCARD:二",
        "best_key": "DISCARD:三",
        "legal_action_count": 2,
        "public_state": {"seat": 0, "hand": ["二", "三"]},
        "candidate_stats": [
            {"label": "二", "average_reward": 0.0},
            {"label": "三", "average_reward": 1.0},
        ],
        "game": {"players": 2, "wildcard_enabled": False},
    }
    original = tmp_path / "original.jsonl.gz"
    extra = tmp_path / "extra.jsonl.gz"
    with gzip.open(original, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({**base, "completed_paired_worlds": 64}) + "\n"
        )
    with gzip.open(extra, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({**base, "completed_paired_worlds": 256}) + "\n"
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"entries": [{"output_evidence": str(original)}]}),
        encoding="utf-8",
    )

    cases, source_rows = _load_cases(
        manifest,
        extra_evidence=[extra],
    )

    assert source_rows == 2
    assert len(cases) == 1
    assert cases[0]["completed_paired_worlds"] == 256
    assert cases[0]["source_evidence"] == str(extra.resolve())


def test_response_candidate_preserves_committed_chi_structure():
    candidate = _response_candidate(
        {
            "key": "CHI:option",
            "action_type": "CHI",
            "heuristic_value": 2.5,
            "option_id": "option",
            "consumed_from_hand": ["二", "十"],
            "meld_groups": [["二", "七", "十"]],
            "followup_discard": "四",
        }
    )

    assert candidate.key == "CHI:option"
    assert candidate.consumed_from_hand == ("二", "十")
    assert candidate.meld_groups == (("二", "七", "十"),)
    assert candidate.followup_discard == "四"
    assert _candidate_key("response_root", {"key": "PASS"}) == "PASS"
    assert _candidate_key("discard", {"label": "二"}) == "DISCARD:二"


def test_summary_reports_paired_regret_reduction_and_harmful_overrides(
    tmp_path,
):
    rows = [
        {
            "error": None,
            "all_candidates_covered": True,
            "mode": "2p_off",
            "phase": "discard",
            "baseline_regret": 0.5,
            "runtime_regret": 0.1,
            "regret_reduction": 0.4,
            "baseline_matches_deep_best": False,
            "runtime_matches_deep_best": False,
            "runtime_override": True,
            "elapsed_ms": 100.0,
        },
        {
            "error": None,
            "all_candidates_covered": True,
            "mode": "2p_off",
            "phase": "discard",
            "baseline_regret": 0.0,
            "runtime_regret": 0.2,
            "regret_reduction": -0.2,
            "baseline_matches_deep_best": True,
            "runtime_matches_deep_best": False,
            "runtime_override": True,
            "elapsed_ms": 200.0,
        },
    ]

    report = _summarize(
        rows,
        manifest=tmp_path / "manifest.json",
        source_rows=2,
    )

    assert report["ok"]
    assert report["baseline_mean_regret"] == 0.25
    assert report["runtime_mean_regret"] == 0.15
    assert report["mean_regret_reduction"] == 0.1
    assert report["supported_overrides"] == 1
    assert report["harmful_overrides"] == 1
    assert not report["development_safety_gate_passed"]
    assert not report["accepted_as_runtime_candidate"]


def test_summary_does_not_call_zero_improvement_an_advantage():
    rows = [
        {
            "error": None,
            "all_candidates_covered": True,
            "mode": mode,
            "phase": "discard",
            "baseline_regret": 0.2,
            "runtime_regret": 0.2,
            "regret_reduction": 0.0,
            "baseline_matches_deep_best": False,
            "runtime_matches_deep_best": False,
            "runtime_override": False,
            "elapsed_ms": 100.0,
        }
        for mode in ("2p_off", "2p_on", "3p_off", "3p_on")
    ]

    report = _summarize(
        rows,
        manifest=__file__,
        source_rows=4,
    )

    assert report["development_safety_gate_passed"]
    assert report["required_modes_present"]
    assert not report["all_modes_positive_regret_reduction_95ci"]
    assert not report["development_advantage_gate_passed"]
    assert not report["accepted_as_runtime_candidate"]


def test_summary_requires_all_four_modes_for_development_advantage_gate(
    tmp_path,
):
    rows = [
        {
            "error": None,
            "all_candidates_covered": True,
            "mode": mode,
            "phase": "discard",
            "baseline_regret": 0.2,
            "runtime_regret": 0.1,
            "regret_reduction": 0.1,
            "baseline_matches_deep_best": False,
            "runtime_matches_deep_best": True,
            "runtime_override": True,
            "deep_override_confidently_supported": True,
            "deep_override_confidently_harmful": False,
            "elapsed_ms": 100.0,
        }
        for mode in ("2p_off", "2p_on", "3p_off", "3p_on")
    ]

    report = _summarize(
        rows,
        manifest=tmp_path / "manifest.json",
        source_rows=4,
    )

    assert report["required_modes_present"]
    assert report["all_modes_positive_regret_reduction_95ci"]
    assert report["development_advantage_gate_passed"]
    assert report["accepted_as_runtime_candidate"]


def test_progressive_search_uses_requested_refinement_budget():
    search = _progressive_search(
        {
            "refinement_candidates": 4,
            "refinement_worlds": 12,
            "refinement_time_budget_ms": 4_000,
            "confirmation_worlds": 48,
        }
    )

    assert search.refinement_candidates == 4
    assert search.refinement.config.max_candidates == 4
    assert search.refinement.config.max_iterations == 48
    assert search.refinement.config.time_budget_ms == 4_000
    assert search.discard_selection.config.max_candidates == 4
    assert search.discard_selection.config.max_iterations == 64
    assert search.confirmation.config.max_iterations == 144


def test_deep_override_advantage_normalizes_discard_keys():
    advantage = _deep_override_advantage(
        {
            "phase": "discard",
            "deep_paired_advantages": [
                {
                    "preferred_key": "五",
                    "candidate_key": "六",
                    "lower_confidence_bound": 0.1,
                }
            ],
        },
        baseline_key="DISCARD:五",
        runtime_key="DISCARD:六",
    )

    assert advantage is not None
    assert advantage["candidate_key"] == "六"

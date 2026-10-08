from tools.compare_paired_leagues import compare_reports


def test_compare_reports_requires_identical_jobs_and_summarizes_paired_delta():
    baseline = _report("professional_brain", scores=(1.0, -2.0), wins=(True, False))
    candidate = _report("candidate", scores=(2.0, -2.0), wins=(True, False))

    report = compare_reports(baseline, candidate)

    assert report["ok"]
    assert report["games"] == 2
    assert report["paired_summary"]["improved"] == 1
    assert report["paired_summary"]["equal"] == 1
    assert report["paired_summary"]["candidate_win_delta"] == 0
    assert report["paired_summary"]["mean_score_delta"] == 0.5


def test_compare_reports_rejects_mismatched_seed():
    baseline = _report("professional_brain", scores=(1.0,), wins=(True,))
    candidate = _report("candidate", scores=(1.0,), wins=(True,))
    candidate["seed"] += 1

    report = compare_reports(baseline, candidate)

    assert not report["ok"]
    assert report["field_mismatches"] == ["seed"]


def test_compare_reports_detects_significant_paired_outcome_gain():
    baseline_wins = (False,) * 20 + (True,) * 7
    candidate_wins = (True,) * 20 + (False,) * 7
    baseline = _report(
        "professional_brain",
        scores=tuple(1.0 if won else -1.0 for won in baseline_wins),
        wins=baseline_wins,
    )
    candidate = _report(
        "candidate",
        scores=tuple(1.0 if won else -1.0 for won in candidate_wins),
        wins=candidate_wins,
    )

    summary = compare_reports(baseline, candidate)["paired_summary"]

    assert summary["outcome_improved"] == 20
    assert summary["outcome_regressed"] == 7
    assert summary["outcome_equal"] == 0
    assert summary["mean_outcome_utility_delta_95"][0] > 0.0
    assert summary["outcome_sign_test_p"] < 0.05
    assert summary["outcome_significant_positive"]


def _report(candidate: str, *, scores: tuple[float, ...], wins: tuple[bool, ...]):
    rows = []
    for index, (score, won) in enumerate(zip(scores, wins, strict=True)):
        rows.append(
            {
                "candidate": candidate,
                "opponents": ["baseline"],
                "candidate_seat": index % 2,
                "dealer": index % 2,
                "seed": 100 + index,
                "candidate_won": won,
                "draw": False,
                "candidate_outcome_score": score,
            }
        )
    return {
        "ok": True,
        "candidate": candidate,
        "players": 2,
        "wildcard_enabled": True,
        "seed": 100,
        "deals_per_matchup": 1,
        "games": len(rows),
        "candidate_wins": sum(wins),
        "losses": len(rows) - sum(wins),
        "draws": 0,
        "candidate_win_share_all": sum(wins) / len(rows),
        "candidate_win_share_decisive": sum(wins) / len(rows),
        "mean_candidate_outcome_score": sum(scores) / len(rows),
        "invariant_violations": 0,
        "coverage_failures": 0,
        "rows": rows,
    }

from tools.fit_anchored_teacher_discard_proxies import dense_teacher_targets


def test_dense_teacher_targets_preserve_shortlist_and_reject_prefiltered() -> None:
    shortlist = {
        "一": (10.0, None),
        "二": (5.0, None),
    }
    all_scores = {
        **shortlist,
        "三": (50.0, None),
    }

    targets = dense_teacher_targets(shortlist, all_scores)

    assert max(targets, key=lambda label: (targets[label], label)) == "一"
    assert targets["三"] < targets["二"] < targets["一"]

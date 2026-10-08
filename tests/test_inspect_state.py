from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
SRC = Path(__file__).resolve().parents[1] / "src"
for path in (PROJECT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aizipai.state.models import Color, Suit
from tools.inspect_state import (
    _expected_total_from_surface,
    _low_confidence_warning,
    _select_pending_action_card,
    card_from_label,
    inspect_screenshot,
)
from vision.button_detector import ButtonDetection
from vision.hand_recognizer import HandCard
from vision.pending_card_recognizer import PendingCard


def test_card_from_label_maps_card_metadata():
    small_two = card_from_label("二")
    big_nine = card_from_label("玖")
    wildcard = card_from_label("王")

    assert small_two.rank == 2
    assert small_two.suit == Suit.SMALL
    assert small_two.color == Color.RED
    assert big_nine.rank == 9
    assert big_nine.suit == Suit.BIG
    assert big_nine.color == Color.BLACK
    assert wildcard.is_wildcard


def test_inspect_state_reports_meld_summaries_and_sanity_checks(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_105209.png"
    if not screenshot.exists():
        return
    monkeypatch.chdir(root)

    result = inspect_screenshot(
        "data/screenshots/screenshot_20260527_105209.png",
        config_path="config/screen_1080x2400.yaml",
        expected_total=20,
    )

    assert result["sanity_checks"]["ok"]
    assert result["sanity_checks"]["controlled_card_count"] == 20
    assert result["meld_group_summaries"]["my_melds"][0]["type"] == "hidden_triplet"


def test_response_buttons_prefer_opponent_pending_card():
    center = PendingCard("pending_action_card", "五", 0.56, 100, 100, 80, 120, "五_001.png")
    opponent = PendingCard("opponent_pending_card", "十", 0.99, 20, 100, 80, 120, "十_001.png")
    buttons = [ButtonDetection("peng", 0.86, 1800, 400, 120, 120)]

    selected = _select_pending_action_card(center, opponent, buttons)

    assert selected is opponent


def test_no_response_buttons_keep_center_pending_card():
    center = PendingCard("pending_action_card", "五", 0.56, 100, 100, 80, 120, "五_001.png")
    opponent = PendingCard("opponent_pending_card", "十", 0.99, 20, 100, 80, 120, "十_001.png")

    selected = _select_pending_action_card(center, opponent, [])

    assert selected is center


def test_surface_count_uses_21_only_for_own_discard_turn():
    discard_button = object()
    response = ButtonDetection("pass", 0.99, 1800, 400, 120, 120)

    assert _expected_total_from_surface(
        buttons=[],
        discard_button=discard_button,
        options=[],
    ) == 21
    assert _expected_total_from_surface(
        buttons=[response],
        discard_button=discard_button,
        options=[],
    ) == 20
    assert _expected_total_from_surface(
        buttons=[],
        discard_button=discard_button,
        options=[object()],
    ) == 20


def test_independent_hybrid_agreement_is_not_rejected_by_legacy_threshold():
    agreed = {
        "name": "壹",
        "confidence": 0.67,
        "template": "hybrid:independent_agreement",
        "recognition_source": "hybrid",
        "hybrid_reason": "independent_agreement",
    }
    single_source = {
        **agreed,
        "recognition_source": "template",
        "hybrid_reason": "",
    }

    assert _low_confidence_warning(agreed, threshold=0.70) is None
    assert _low_confidence_warning(single_source, threshold=0.70) == "low_confidence:壹:0.67"


def test_missing_row_recovery_never_exceeds_expected_total(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = (
        root
        / "data"
        / "debug_previews"
        / "live_halt_20260723_1520"
        / "phone_current.png"
    )
    if not screenshot.exists():
        return
    calls = 0

    def fake_recognize_hand(*args, **kwargs):
        nonlocal calls
        calls += 1
        count = 18 if calls == 1 else 22
        return [
            HandCard("一", 0.99, 500 + index * 10, 700, 145, 128, "一.png")
            for index in range(count)
        ]

    monkeypatch.setattr("tools.inspect_state.recognize_hand", fake_recognize_hand)

    result = inspect_screenshot(
        screenshot,
        config_path=root / "config" / "screen_1080x2400.yaml",
        expected_total=21,
    )

    assert calls == 2
    assert result["hand_count"] == 18
    assert result["hand_recognition"].get("recovery_pass") is None

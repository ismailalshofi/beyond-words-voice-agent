import math
from array import array

import pytest

from app.listening.listener import Listener
from app.listening.policy import Policy, Rule
from app.listening.style import SpeakingStyle
from app.listening.turn_end import TextTurnEnd
from app.listening.vad import VoiceActivity, frame_level_db


def _tone(amplitude, samples=1600):
    return array("h", (int(amplitude * math.sin(i / 5)) for i in range(samples))).tobytes()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ---------- Voice activity ----------


def test_frame_level_of_silence_and_loud_tone():
    assert frame_level_db(bytes(3200)) < -90
    assert frame_level_db(b"") == -100.0
    assert frame_level_db(_tone(16000)) > -10


def test_voice_activity_separates_speech_from_room_noise():
    vad = VoiceActivity(hangover_frames=0)
    for _ in range(20):
        assert not vad.is_speech(_tone(30))  # quiet room
    assert vad.is_speech(_tone(4000))
    assert not vad.is_speech(_tone(30))


def test_voice_activity_hangover_bridges_a_short_gap():
    vad = VoiceActivity(hangover_frames=1)
    vad.is_speech(_tone(4000))
    assert vad.is_speech(_tone(10))
    assert not vad.is_speech(_tone(10))


# ---------- Style ----------


def test_baseline_needs_a_few_pauses_then_uses_the_median():
    style = SpeakingStyle(default_pause_ms=700)
    style.add_pause(1500)
    style.add_pause(1600)
    assert style.baseline_pause_ms == 700
    style.add_pause(1400)
    assert style.baseline_pause_ms == 1500


def test_baseline_is_clamped():
    style = SpeakingStyle(max_ms=2500)
    for _ in range(3):
        style.add_pause(9000)
    assert style.baseline_pause_ms == 2500


# ---------- End of turn ----------


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("", 0.0, 0.0),
        ("What time do you open?", 0.9, 1.0),
        ("That's all.", 0.8, 0.9),
        ("So I was thinking that", 0.0, 0.2),
        ("I went there and, um", 0.0, 0.2),
        ("Well,", 0.2, 0.3),
        ("I'd like to book", 0.4, 0.6),
    ],
)
def test_text_turn_end(text, low, high):
    assert low <= TextTurnEnd().probability(text) <= high


# ---------- Listener ----------


def test_silence_counts_from_the_last_person_who_stopped():
    clock = Clock()
    listener = Listener(clock=clock)
    listener.set_agent_speaking(True)
    clock.now = 2.0
    assert listener.state().silence_ms == 0

    listener.set_agent_speaking(False)
    clock.now = 2.5
    assert listener.state().silence_ms == pytest.approx(500)


def test_pauses_inside_a_turn_teach_the_baseline():
    clock = Clock()
    listener = Listener(clock=clock)
    for start in (0.0, 3.0, 6.0, 9.0):
        clock.now = start
        listener.set_user_speaking(True)
        listener.on_final("word")
        clock.now = start + 1.0
        listener.set_user_speaking(False)
    # Three pauses of 2 s between the four bursts
    assert listener.state().baseline_pause_ms == 2000


def test_overlap_is_dropped_when_the_agent_finishes_but_kept_when_interrupted():
    listener = Listener(clock=Clock())
    listener.set_agent_speaking(True)
    listener.on_final("yeah")
    listener.set_agent_speaking(False)
    assert listener.state().pending_text == ""

    listener.set_agent_speaking(True)
    listener.on_final("wait, stop")
    listener.set_agent_speaking(False, interrupted=True)
    assert listener.state().pending_text == "wait, stop"


def test_take_turn_hands_over_the_text_and_resets():
    listener = Listener(clock=Clock())
    listener.on_final("Hello")
    listener.on_final("there.")
    listener.mark_reaction("backchannel")

    assert listener.take_turn() == "Hello there."
    state = listener.state()
    assert state.pending_text == ""
    assert state.reactions == frozenset()
    assert state.turns == 1


def test_speech_without_words_is_tracked():
    clock = Clock()
    listener = Listener(clock=clock)
    listener.set_user_speaking(True)
    clock.now = 1.2
    listener.set_user_speaking(False)
    assert listener.state().unheard_speech_ms == pytest.approx(1200)

    listener.on_interim("hello")
    assert listener.state().unheard_speech_ms == 0


# ---------- Policy ----------


def test_first_matching_rule_wins():
    policy = Policy(
        [
            Rule("talking", "wait", {"user_speaking": True}),
            Rule("anything", "respond", {}),
        ]
    )
    listener = Listener(clock=Clock())
    listener.set_user_speaking(True)
    assert policy.decide(listener.state()).rule == "talking"
    listener.set_user_speaking(False)
    assert policy.decide(listener.state()).action == "respond"


def test_policy_rejects_unknown_conditions_and_actions(tmp_path):
    bad_condition = tmp_path / "a.toml"
    bad_condition.write_text('[[rule]]\nname = "x"\naction = "wait"\nwhen = { silence_ms_ish = 3 }\n')
    with pytest.raises(ValueError, match="silence_ms_ish"):
        Policy.load(bad_condition)

    bad_action = tmp_path / "b.toml"
    bad_action.write_text('[[rule]]\nname = "x"\naction = "dance"\n')
    with pytest.raises(ValueError, match="dance"):
        Policy.load(bad_action)


def test_policy_lines():
    policy = Policy([], {"nudge": ["Take your time."]})
    assert policy.line("nudge") == "Take your time."
    assert policy.line("missing") is None

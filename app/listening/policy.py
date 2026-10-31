import random
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.listening.state import CallState

Action = Literal["stop", "wait", "backchannel", "nudge", "repair", "respond"]
ACTIONS = {"stop", "wait", "backchannel", "nudge", "repair", "respond"}
# Short spoken lines with no LLM behind them. They have to be fast, so they play from clips.
FAST_REACTIONS = {"backchannel", "nudge", "repair"}

# Every condition a rule can use. A rule matches when all of its conditions hold.
CONDITIONS: dict[str, Callable[[CallState, object], bool]] = {
    "agent_speaking": lambda s, v: s.agent_speaking == v,
    "user_speaking": lambda s, v: s.user_speaking == v,
    "has_pending_text": lambda s, v: bool(s.pending_text) == v,
    "interim_words_at_least": lambda s, v: len(s.interim_text.split()) >= v,
    "end_of_turn_at_least": lambda s, v: s.end_of_turn >= v,
    "end_of_turn_below": lambda s, v: s.end_of_turn < v,
    "silence_at_least_ms": lambda s, v: s.silence_ms >= v,
    "silence_at_least_x_baseline": lambda s, v: s.silence_ms >= v * s.baseline_pause_ms,
    "silence_below_x_baseline": lambda s, v: s.silence_ms < v * s.baseline_pause_ms,
    "words_quiet_at_least_ms": lambda s, v: s.words_quiet_ms >= v,
    "unheard_speech_at_least_ms": lambda s, v: s.unheard_speech_ms >= v,
    "not_reacted": lambda s, v: v not in s.reactions,
}


@dataclass(frozen=True)
class Rule:
    name: str
    action: Action
    when: dict = field(default_factory=dict)
    # Extra instruction for the LLM when this rule makes the agent respond
    guidance: str = ""

    def matches(self, state: CallState) -> bool:
        return all(CONDITIONS[key](state, value) for key, value in self.when.items())


@dataclass(frozen=True)
class Decision:
    action: Action
    rule: str
    guidance: str = ""


WAIT = Decision("wait", "no rule matched")


class Policy:
    """Reads the state and decides what the agent does: wait, say "mm", stop, or answer.

    It is plain code driven by a rules file, so it is fast, it does the same thing every time,
    and it can be tested. The LLM can break a prompt rule, but it can't break a policy rule.
    """

    def __init__(self, rules: list[Rule], lines: dict[str, list[str]] | None = None):
        self.rules = rules
        self.lines = lines or {}

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        rules = [Rule(**raw) for raw in data.get("rule", [])]
        for rule in rules:
            # Fail when the file loads, not in the middle of a call
            if rule.action not in ACTIONS:
                raise ValueError(f"Rule '{rule.name}': unknown action '{rule.action}'")
            unknown = set(rule.when) - set(CONDITIONS)
            if unknown:
                raise ValueError(f"Rule '{rule.name}': unknown condition(s) {sorted(unknown)}")
        lines = {kind: [text] if isinstance(text, str) else list(text) for kind, text in data.get("lines", {}).items()}
        return cls(rules, lines)

    def decide(self, state: CallState) -> Decision:
        for rule in self.rules:
            if rule.matches(state):
                return Decision(rule.action, rule.name, rule.guidance)
        return WAIT

    def line(self, kind: str) -> str | None:
        options = self.lines.get(kind)
        return random.choice(options) if options else None

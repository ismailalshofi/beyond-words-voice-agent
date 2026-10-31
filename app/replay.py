"""Replay scripted moments of a call through the listener and the policy, with a fake clock.

    python -m app.replay                    # every scenario in scenarios/
    python -m app.replay scenarios/x.toml   # one scenario
    python -m app.replay --policy my.toml   # try a different rules file

A scenario is a timeline of what the caller did (talked, went quiet, what the STT heard) and what
the agent is expected to do, and when. No audio, no API keys, the same result every run, so a
change to a rule is checked against every scenario before it reaches a real call.
"""

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from app.listening.listener import Listener
from app.listening.policy import FAST_REACTIONS, Policy
from app.settings import settings

STEP_MS = 50


@dataclass
class Result:
    name: str
    path: Path
    timeline: list[tuple[int, str, str]]
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def report(self) -> str:
        lines = [f"{'PASS' if self.passed else 'FAIL'}  {self.name}  ({self.path.name})"]
        lines += [f"        {at:>6} ms  {action:<12} {rule}" for at, action, rule in self.timeline]
        lines += [f"     !  {failure}" for failure in self.failures]
        return "\n".join(lines)


def run_scenario(path: Path, policy: Policy) -> Result:
    scenario = tomllib.loads(path.read_text(encoding="utf-8"))
    events = sorted(scenario.get("event", []), key=lambda e: e["at"])
    expects = scenario.get("expect", [])
    duration = scenario.get("duration_ms", max([e["at"] for e in events] + [0]) + 10_000)
    # How long the agent talks when it answers
    reply_ms = scenario.get("reply_ms", 2000)

    now = 0.0
    listener = Listener(clock=lambda: now / 1000)
    timeline: list[tuple[int, str, str]] = []
    reply_ends_at: int | None = None

    t = 0
    while t <= duration:
        now = t
        if reply_ends_at is not None and t >= reply_ends_at:
            listener.set_agent_speaking(False)
            reply_ends_at = None
        while events and events[0]["at"] <= t:
            _apply(listener, events.pop(0))

        decision = policy.decide(listener.state())
        if decision.action != "wait":
            timeline.append((t, decision.action, decision.rule))
            # What the call would do, minus the audio
            if decision.action == "respond":
                listener.take_turn()
                listener.set_agent_speaking(True)
                reply_ends_at = t + reply_ms
            elif decision.action == "stop":
                listener.set_agent_speaking(False, interrupted=True)
                reply_ends_at = None
            elif decision.action in FAST_REACTIONS:
                listener.mark_reaction(decision.action)
        t += STEP_MS

    result = Result(scenario.get("name", path.stem), path, timeline)
    result.failures = [f for expect in expects if (f := _check(expect, timeline))]
    return result


def _apply(listener: Listener, event: dict) -> None:
    if "speech" in event:
        listener.set_user_speaking(event["speech"])
    if "interim" in event:
        listener.on_interim(event["interim"])
    if "final" in event:
        listener.on_final(event["final"])
    if "agent" in event:
        listener.set_agent_speaking(event["agent"])


def _check(expect: dict, timeline: list[tuple[int, str, str]]) -> str | None:
    if "never" in expect:
        before = expect.get("before", float("inf"))
        hits = [at for at, action, _ in timeline if action == expect["never"] and at < before]
        if hits:
            limit = f" before {before} ms" if before != float("inf") else ""
            return f"expected no '{expect['never']}'{limit}, got one at {hits[0]} ms"
        return None

    action = expect["action"]
    hits = [at for at, a, _ in timeline if a == action]
    if "between" in expect:
        low, high = expect["between"]
        if not any(low <= at <= high for at in hits):
            got = ", ".join(f"{at} ms" for at in hits) or "never"
            return f"expected '{action}' between {low} and {high} ms, got: {got}"
    if "times" in expect and len(hits) != expect["times"]:
        return f"expected '{action}' {expect['times']} time(s), got {len(hits)}"
    return None


def find_scenarios(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        found += sorted(path.glob("*.toml")) if path.is_dir() else [path]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay call scenarios through the policy.")
    parser.add_argument("paths", nargs="*", default=["scenarios"])
    parser.add_argument("--policy", default=settings.POLICY_FILE)
    args = parser.parse_args()

    policy = Policy.load(args.policy)
    results = [run_scenario(path, policy) for path in find_scenarios(args.paths)]
    for result in results:
        print(result.report() + "\n")

    passed = sum(r.passed for r in results)
    print(f"{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

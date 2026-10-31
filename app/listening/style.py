from collections import deque
from statistics import median


class SpeakingStyle:
    """How this caller talks. For now, one number: their usual pause in the middle of a turn.

    Three seconds of silence means one thing from a fast talker and another from a slow one,
    so every timing rule in the policy is measured against this baseline.
    """

    def __init__(self, default_pause_ms: float = 700, window: int = 20, min_ms: float = 300, max_ms: float = 2500):
        self.default_pause_ms = default_pause_ms
        self.min_ms = min_ms
        self.max_ms = max_ms
        self._pauses: deque[float] = deque(maxlen=window)

    def add_pause(self, pause_ms: float) -> None:
        self._pauses.append(pause_ms)

    @property
    def baseline_pause_ms(self) -> float:
        # A couple of pauses could be anything; wait for a few before trusting them
        if len(self._pauses) < 3:
            return self.default_pause_ms
        return min(self.max_ms, max(self.min_ms, median(self._pauses)))

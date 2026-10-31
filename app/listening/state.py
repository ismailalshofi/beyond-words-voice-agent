from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CallState:
    """One snapshot of the call right now.

    Facts only, no decisions. The listener writes it, the policy reads it.
    """

    call_ms: float
    turns: int
    agent_speaking: bool
    user_speaking: bool
    # Time since the last person stopped talking, 0 while anyone is talking
    silence_ms: float
    baseline_pause_ms: float
    # Final STT segments of the turn in progress, and the segment being recognised now
    pending_text: str
    interim_text: str
    end_of_turn: float
    # Time since the STT last gave us any words
    words_quiet_ms: float
    # Speech the mic picked up that the STT turned into no words at all
    unheard_speech_ms: float
    # Fast reactions already used in this pause, so the agent doesn't say "mm" twice
    reactions: frozenset[str]

    def snapshot(self) -> dict:
        data = asdict(self)
        data["reactions"] = sorted(self.reactions)
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 2) if key == "end_of_turn" else round(value)
        return data

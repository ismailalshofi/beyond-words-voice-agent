import time
from collections.abc import Callable

from app.listening.state import CallState
from app.listening.style import SpeakingStyle
from app.listening.turn_end import TextTurnEnd, TurnEndModel
from app.listening.vad import VoiceActivity


class Listener:
    """Turns the raw signals of a call into the state.

    It hears the audio, the transcript and the agent's own playback, and keeps the facts up to
    date. It never decides anything: what to do with a 2-second pause is the policy's job.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        vad: VoiceActivity | None = None,
        style: SpeakingStyle | None = None,
        turn_end: TurnEndModel | None = None,
    ):
        self._clock = clock
        self.vad = vad or VoiceActivity()
        self.style = style or SpeakingStyle()
        self.turn_end = turn_end or TextTurnEnd()

        self._started_at = clock()
        self.turns = 0
        self._user_speaking = False
        self._agent_speaking = False
        self._quiet_since = self._started_at
        self._user_stopped_at: float | None = None

        self._pending: list[str] = []
        self._interim = ""
        # Segments heard while the agent talked; dropped if the agent wasn't interrupted ("yeah", "mm")
        self._overlap_from = 0

        self._last_words_at = self._started_at
        self._speech_started_at: float | None = None
        self._words_since_speech = False
        self._unheard_ms = 0.0

        self._reactions: set[str] = set()

    # ---------- Signals in ----------

    def on_audio(self, frame: bytes) -> None:
        self.set_user_speaking(self.vad.is_speech(frame))

    def set_user_speaking(self, speaking: bool) -> None:
        if speaking == self._user_speaking:
            return
        now = self._clock()
        self._user_speaking = speaking

        if speaking:
            # A pause inside a turn is how we learn this caller's rhythm
            if self._pending and self._user_stopped_at is not None:
                self.style.add_pause((now - self._user_stopped_at) * 1000)
            self._speech_started_at = now
            self._words_since_speech = False
            self._reactions.clear()
        else:
            self._user_stopped_at = now
            if self._speech_started_at is not None and not self._words_since_speech:
                self._unheard_ms += (now - self._speech_started_at) * 1000
            if not self._agent_speaking:
                self._quiet_since = now

    def on_interim(self, text: str) -> None:
        self._interim = text.strip()
        if self._interim:
            self._heard_words()

    def on_final(self, text: str) -> None:
        self._interim = ""
        text = text.strip()
        if text:
            self._pending.append(text)
            self._heard_words()

    def set_agent_speaking(self, speaking: bool, *, interrupted: bool = False) -> None:
        if speaking == self._agent_speaking:
            return
        self._agent_speaking = speaking
        if speaking:
            self._overlap_from = len(self._pending)
            return
        if not interrupted:
            del self._pending[self._overlap_from :]
        self._reactions.clear()
        if not self._user_speaking:
            self._quiet_since = self._clock()

    def mark_reaction(self, kind: str) -> None:
        self._reactions.add(kind)

    def take_turn(self) -> str:
        """Hand the finished turn to the agent and start a new one."""
        text = " ".join(self._pending)
        self._pending.clear()
        self._interim = ""
        self._unheard_ms = 0.0
        self._reactions.clear()
        self.turns += 1
        return text

    # ---------- State out ----------

    def state(self) -> CallState:
        now = self._clock()
        pending = " ".join(self._pending)
        talking = self._user_speaking or self._agent_speaking

        unheard_ms = self._unheard_ms
        if self._user_speaking and not self._words_since_speech and self._speech_started_at is not None:
            unheard_ms += (now - self._speech_started_at) * 1000

        return CallState(
            call_ms=(now - self._started_at) * 1000,
            turns=self.turns,
            agent_speaking=self._agent_speaking,
            user_speaking=self._user_speaking,
            silence_ms=0.0 if talking else (now - self._quiet_since) * 1000,
            baseline_pause_ms=self.style.baseline_pause_ms,
            pending_text=pending,
            interim_text=self._interim,
            end_of_turn=self.turn_end.probability(pending),
            words_quiet_ms=(now - self._last_words_at) * 1000,
            unheard_speech_ms=unheard_ms,
            reactions=frozenset(self._reactions),
        )

    def _heard_words(self) -> None:
        self._last_words_at = self._clock()
        self._words_since_speech = True
        self._unheard_ms = 0.0

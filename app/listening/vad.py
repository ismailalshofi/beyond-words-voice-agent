import math
from array import array

FRAME_MS = 100  # the browser sends 100 ms frames


class VoiceActivity:
    """Energy-based voice activity on PCM16 frames.

    It runs on the raw audio, so it knows the caller stopped talking ~300 ms before the STT does.
    A frame counts as speech when it is clearly louder than the background noise. The noise floor
    follows the quiet frames slowly, and drops at once when the room gets quieter.
    """

    def __init__(self, threshold_db: float = 12.0, adapt: float = 0.05, hangover_frames: int = 1):
        self.threshold_db = threshold_db
        self.adapt = adapt
        self.hangover_frames = hangover_frames
        self.noise_floor_db = -50.0
        self._hangover = 0

    def is_speech(self, frame: bytes) -> bool:
        level = frame_level_db(frame)
        if level > self.noise_floor_db + self.threshold_db:
            # A short hangover keeps quiet consonants and tiny gaps inside a word as speech
            self._hangover = self.hangover_frames
            return True

        if level < self.noise_floor_db:
            self.noise_floor_db = level
        else:
            self.noise_floor_db += self.adapt * (level - self.noise_floor_db)

        if self._hangover > 0:
            self._hangover -= 1
            return True
        return False


def frame_level_db(frame: bytes) -> float:
    samples = array("h", frame[: len(frame) - len(frame) % 2])
    if not samples:
        return -100.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    return 20 * math.log10(max(rms, 1.0) / 32768)

PCM16_16K_BYTES_PER_MS = 32  # 16 000 samples/s * 2 bytes / 1000


class SpokenTimeline:
    """Maps the audio of one response back to the text it contains.

    ElevenLabs sends character timings with each audio chunk, relative to that chunk.
    Laying the chunks end to end gives every character an absolute start time, so after
    a barge-in we can say which words the caller actually heard.
    """

    def __init__(self, bytes_per_ms: float = PCM16_16K_BYTES_PER_MS):
        self._bytes_per_ms = bytes_per_ms
        self._chars: list[tuple[float, str]] = []
        self.audio_ms = 0.0

    def add_audio(self, n_bytes: int, alignment: dict | None) -> None:
        offset = self.audio_ms
        if alignment:
            chars = alignment.get("chars") or []
            starts = alignment.get("charStartTimesMs") or []
            for char, start_ms in zip(chars, starts, strict=False):
                self._chars.append((offset + start_ms, char))
        self.audio_ms += n_bytes / self._bytes_per_ms

    def text_before(self, played_ms: float) -> str:
        """The text spoken in the first `played_ms`, cut back to the last whole word."""
        heard = "".join(char for start, char in self._chars if start < played_ms)
        cut_mid_word = len(heard) < len(self._chars) and not self._chars[len(heard)][1].isspace()
        if cut_mid_word and " " in heard:
            heard = heard.rsplit(" ", 1)[0]
        elif cut_mid_word:
            heard = ""
        return " ".join(heard.split())

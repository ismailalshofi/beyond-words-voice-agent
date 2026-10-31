import random
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol

from loguru import logger


class ClipVoice(Protocol):
    def stream(self, text_iterator: AsyncIterator[str]) -> AsyncIterator[bytes]: ...
    async def disconnect(self) -> None: ...


class Clips:
    """Short lines ("Mm-hmm.", "Take your time.") rendered once per call in the agent's voice.

    A fast reaction has to land in under ~300 ms, which leaves no room for a TTS round trip.
    Rendering the lines at call start means they play at once and sound like the same speaker.
    They use their own TTS connection so they never queue behind a real reply.
    """

    def __init__(self, voice: ClipVoice, lines: dict[str, list[str]]):
        self.voice = voice
        self.lines = lines
        self._clips: dict[str, list[bytes]] = {}

    async def warm(self) -> None:
        try:
            for kind, texts in self.lines.items():
                for text in texts:
                    audio = b"".join([chunk async for chunk in self.voice.stream(_single(text))])
                    if audio:
                        self._clips.setdefault(kind, []).append(audio)
            logger.info(f"Clips ready: {sorted(self._clips)}")
        except Exception as e:
            # Fast reactions are a nice-to-have; the call works without them
            logger.warning(f"Rendering clips failed: {e}")
        finally:
            await self.close()

    def get(self, kind: str) -> bytes | None:
        options = self._clips.get(kind)
        return random.choice(options) if options else None

    async def close(self) -> None:
        with suppress(Exception):
            await self.voice.disconnect()


async def _single(text: str) -> AsyncIterator[str]:
    yield text

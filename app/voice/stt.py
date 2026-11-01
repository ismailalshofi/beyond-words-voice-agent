import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Protocol
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import ClientConnection, connect
from websockets.protocol import State

from app.settings import settings

SAMPLE_RATE = 16000

# Called with the words being recognised right now (not stable yet)
OnSpeech = Callable[[str], Awaitable[None]]
# Called with each stable segment. The STT doesn't decide when the turn ends, the policy does.
OnFinal = Callable[[str], Awaitable[None]]


class SpeechToText(Protocol):
    async def connect(self) -> None: ...
    async def send_audio(self, chunk: bytes) -> None: ...
    async def keep_alive(self) -> None: ...
    async def close(self) -> None: ...


def create_stt(provider: str, *, language: str, on_speech: OnSpeech, on_final: OnFinal) -> SpeechToText:
    if provider == "scribe":
        return ScribeSTT(language=language, on_speech=on_speech, on_final=on_final)
    return DeepgramSTT(language=language, on_speech=on_speech, on_final=on_final)


class DeepgramSTT:
    """Deepgram streaming STT.

    - is_final=false -> interim words, used for barge-in and shown live.
    - is_final=true  -> a stable segment, added to the caller's turn.
    """

    URL = "wss://api.deepgram.com/v1/listen"

    def __init__(self, *, language: str, on_speech: OnSpeech, on_final: OnFinal):
        self.language = language or "en"
        self._on_speech = on_speech
        self._on_final = on_final
        self._ws: ClientConnection | None = None
        self._receiver: asyncio.Task | None = None

    async def connect(self) -> None:
        query = urlencode(
            {
                "model": "nova-3",
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "language": self.language,
                "smart_format": "false",
                "punctuate": "true",
                "interim_results": "true",
                # Short endpointing so segments become stable fast; the turn itself is the policy's call
                "endpointing": 300,
            }
        )
        self._ws = await connect(
            f"{self.URL}?{query}",
            additional_headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"},
        )
        self._receiver = asyncio.create_task(self._receive(), name="deepgram_receiver")
        logger.info("Deepgram connected")

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws and self._ws.state is State.OPEN:
            await self._ws.send(chunk)

    async def keep_alive(self) -> None:
        if self._ws and self._ws.state is State.OPEN:
            await self._ws.send(json.dumps({"type": "KeepAlive"}))

    async def close(self) -> None:
        if self._ws:
            with suppress(Exception):
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            with suppress(Exception):
                await self._ws.close()
        if self._receiver and not self._receiver.done():
            self._receiver.cancel()

    async def _receive(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                try:
                    await self.handle_message(json.loads(message))
                except Exception:
                    logger.exception("Error handling Deepgram message")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Deepgram receiver stopped: {e}")

    async def handle_message(self, data: dict) -> None:
        if data.get("type") != "Results":
            return

        alternatives = (data.get("channel") or {}).get("alternatives") or [{}]
        sentence = (alternatives[0].get("transcript") or "").strip()
        if not sentence:
            return

        if data.get("is_final"):
            await self._on_final(sentence)
        else:
            await self._on_speech(sentence)


class ScribeSTT:
    """ElevenLabs Scribe realtime STT. Scribe commits whole utterances with its own VAD."""

    URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(self, *, language: str, on_speech: OnSpeech, on_final: OnFinal):
        self.language = language
        self._on_speech = on_speech
        self._on_final = on_final
        self._ws: ClientConnection | None = None
        self._receiver: asyncio.Task | None = None

    async def connect(self) -> None:
        params = {
            "model_id": "scribe_v2_realtime",
            "audio_format": f"pcm_{SAMPLE_RATE}",
            "commit_strategy": "vad",
            # Short, like Deepgram's endpointing: the policy waits for the turn, not Scribe.
            # At 0.8 s every reply waited 0.8 s for the last words.
            "vad_silence_threshold_secs": 0.4,
            "vad_threshold": 0.45,
            "min_silence_duration_ms": 300,
            "min_speech_duration_ms": 200,
        }
        if self.language:
            params["language_code"] = self.language
        self._ws = await connect(
            f"{self.URL}?{urlencode(params)}",
            additional_headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
        )
        self._receiver = asyncio.create_task(self._receive(), name="scribe_receiver")
        logger.info("Scribe connected")

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws and self._ws.state is State.OPEN:
            await self._ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(chunk).decode(),
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            )

    async def keep_alive(self) -> None:
        # The browser streams audio continuously, which keeps Scribe alive
        pass

    async def close(self) -> None:
        if self._ws:
            with suppress(Exception):
                await self._ws.close()
        if self._receiver and not self._receiver.done():
            self._receiver.cancel()

    async def _receive(self) -> None:
        try:
            async for message in self._ws:
                data = json.loads(message)
                message_type = data.get("message_type")
                text = (data.get("text") or "").strip()

                if message_type == "partial_transcript" and text:
                    await self._on_speech(text)
                elif message_type == "committed_transcript" and text:
                    await self._on_final(text)
                elif message_type and "error" in message_type:
                    logger.warning(f"Scribe error: {data}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Scribe receiver stopped: {e}")

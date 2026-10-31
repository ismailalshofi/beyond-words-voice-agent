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

# Called with any recognised speech (partial or final) - used for barge-in
OnSpeech = Callable[[str], Awaitable[None]]
# Called once per completed user turn
OnUtterance = Callable[[str], Awaitable[None]]


class SpeechToText(Protocol):
    async def connect(self) -> None: ...
    async def send_audio(self, chunk: bytes) -> None: ...
    async def keep_alive(self) -> None: ...
    def reset(self) -> None: ...
    async def close(self) -> None: ...


def create_stt(provider: str, *, language: str, on_speech: OnSpeech, on_utterance: OnUtterance) -> SpeechToText:
    if provider == "scribe":
        return ScribeSTT(language=language, on_speech=on_speech, on_utterance=on_utterance)
    return DeepgramSTT(language=language, on_speech=on_speech, on_utterance=on_utterance)


class DeepgramSTT:
    """Deepgram streaming STT with utterance buffering.

    Buffering strategy:
    - is_final=true, speech_final=false -> stable segment, speaker still talking. Buffer it.
    - is_final=true, speech_final=true  -> stable segment + speaker paused. Buffer it, complete the turn.
    - is_final=false -> interim preview, used for barge-in but not buffered.
    - UtteranceEnd event -> backup when noise blocks speech_final detection.
    - Safety timer -> flushes the buffer if neither signal arrives.
    """

    URL = "wss://api.deepgram.com/v1/listen"

    def __init__(
        self,
        *,
        language: str,
        on_speech: OnSpeech,
        on_utterance: OnUtterance,
        safety_timeout_s: float = 2.0,
    ):
        self.language = language or "en"
        self._on_speech = on_speech
        self._on_utterance = on_utterance
        self._safety_timeout_s = safety_timeout_s

        self.utterance_buffer: list[str] = []
        self._safety_timer_task: asyncio.Task | None = None
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
                "endpointing": 300,
                "utterance_end_ms": 1000,
                "vad_events": "true",
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

    def reset(self) -> None:
        self.utterance_buffer.clear()
        self._clear_safety_timer()

    async def close(self) -> None:
        self.reset()
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
        if data.get("type") == "UtteranceEnd":
            if self.utterance_buffer:
                await self._complete_turn()
            return

        if data.get("type") != "Results":
            return

        alternatives = (data.get("channel") or {}).get("alternatives") or [{}]
        sentence = (alternatives[0].get("transcript") or "").strip()
        if not sentence:
            return

        # The call may barge in here, which resets this buffer
        await self._on_speech(sentence)

        if not data.get("is_final"):
            # Interim result: speaker still talking, push the safety timer back
            self._reset_safety_timer()
            return

        self.utterance_buffer.append(sentence)

        if data.get("speech_final"):
            await self._complete_turn()
        else:
            self._reset_safety_timer()

    async def _complete_turn(self) -> None:
        self._clear_safety_timer()
        utterance = " ".join(self.utterance_buffer).strip()
        self.utterance_buffer.clear()
        if utterance:
            await self._on_utterance(utterance)

    def _reset_safety_timer(self) -> None:
        self._clear_safety_timer()
        self._safety_timer_task = asyncio.create_task(self._safety_timer())

    async def _safety_timer(self) -> None:
        try:
            await asyncio.sleep(self._safety_timeout_s)
            if self.utterance_buffer:
                logger.info("[SAFETY_TIMER] Flushing buffer (speech_final never arrived)")
                await self._complete_turn()
        except asyncio.CancelledError:
            pass

    def _clear_safety_timer(self) -> None:
        task = self._safety_timer_task
        # Never cancel ourselves: the timer itself completes the turn through this path
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
        self._safety_timer_task = None


class ScribeSTT:
    """ElevenLabs Scribe realtime STT. Scribe does its own VAD and commits whole turns."""

    URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(self, *, language: str, on_speech: OnSpeech, on_utterance: OnUtterance):
        self.language = language
        self._on_speech = on_speech
        self._on_utterance = on_utterance
        self._ws: ClientConnection | None = None
        self._receiver: asyncio.Task | None = None

    async def connect(self) -> None:
        params = {
            "model_id": "scribe_v2_realtime",
            "audio_format": f"pcm_{SAMPLE_RATE}",
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": 0.8,
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

    def reset(self) -> None:
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
                    await self._on_utterance(text)
                elif message_type and "error" in message_type:
                    logger.warning(f"Scribe error: {data}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Scribe receiver stopped: {e}")

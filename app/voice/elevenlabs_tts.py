import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from app.settings import settings
from app.voice.call_logger import CallLogger
from app.voice.spoken_timeline import SpokenTimeline

RECEIVE_TIMEOUT_S = 30.0

OnComplete = Callable[[str], Awaitable[None]]


@dataclass
class TTSRequest:
    text_iterator: AsyncIterator[str]
    on_complete: OnComplete | None
    audio_queue: asyncio.Queue
    timeline: SpokenTimeline | None = None
    cancelled: bool = False

    async def finish(self) -> None:
        await self.audio_queue.put(None)


class ElevenLabsTTS:
    """Streams text into ElevenLabs' input-streaming websocket and yields raw audio chunks.

    Requests are serialized through a queue so overlapping responses never mix audio.
    ElevenLabs can't cancel mid-stream, so barge-in drops the socket and pre-warms a new one.
    Every request ends the socket (end-of-input), so a fresh one is pre-warmed after each
    request too, keeping the connect handshake off the next turn's critical path.
    """

    DEFAULT_VOICE_SETTINGS = {
        "stability": 0.5,
        "use_speaker_boost": True,
        "similarity_boost": 0.75,
        "style": 0,
        "speed": 1,
    }

    def __init__(
        self,
        voice_id: str,
        call_logger: CallLogger,
        model_id: str | None = None,
        output_format: str = "pcm_16000",
        voice_settings: dict | None = None,
    ):
        self.voice_id = voice_id
        self.model_id = model_id or settings.ELEVENLABS_MODEL_ID
        self.output_format = output_format
        self.voice_settings = voice_settings or self.DEFAULT_VOICE_SETTINGS
        self.call_logger = call_logger

        self._ws: ClientConnection | None = None
        self._connect_lock = asyncio.Lock()
        self._requests: asyncio.Queue[TTSRequest | None] = asyncio.Queue()
        self._current: TTSRequest | None = None
        self._processor: asyncio.Task | None = None
        self._prewarm_task: asyncio.Task | None = None
        self._closed = False

    @property
    def _uri(self) -> str:
        query = urlencode(
            {
                "model_id": self.model_id,
                "output_format": self.output_format,
                "inactivity_timeout": 180,
                # Character timings with every audio chunk, used to work out what a caller heard
                "sync_alignment": "true",
            }
        )
        return f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?{query}"

    async def connect(self) -> None:
        await self._ensure_connected()
        if self._processor is None:
            self._processor = asyncio.create_task(self._process_requests(), name="tts_processor")

    async def disconnect(self) -> None:
        self._closed = True
        await self.cancel()
        if self._processor and not self._processor.done():
            await self._requests.put(None)
            try:
                await asyncio.wait_for(self._processor, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._processor.cancel()
        if self._prewarm_task and not self._prewarm_task.done():
            self._prewarm_task.cancel()
        await self._close_ws()

    async def cancel(self) -> None:
        """Barge-in: drop the current and queued requests."""
        while True:
            try:
                request = self._requests.get_nowait()
            except asyncio.QueueEmpty:
                break
            if request is None:
                # Keep the shutdown sentinel for the processor
                self._requests.put_nowait(None)
                break
            request.cancelled = True
            await request.finish()

        if self._current:
            self._current.cancelled = True
            # Closing the socket unblocks the receiver for the current request
            await self._close_ws()
            self._prewarm()

    async def stream(
        self,
        text_iterator: AsyncIterator[str],
        on_complete: OnComplete | None = None,
        timeline: SpokenTimeline | None = None,
    ) -> AsyncIterator[bytes]:
        """Queue text for synthesis and yield audio chunks as they arrive.

        `on_complete` receives the full spoken text once all of it has been sent.
        `timeline` is filled with the character timings of the audio.
        """
        if self._processor is None:
            await self.connect()

        request = TTSRequest(
            text_iterator=text_iterator, on_complete=on_complete, audio_queue=asyncio.Queue(), timeline=timeline
        )
        await self._requests.put(request)

        try:
            while (chunk := await request.audio_queue.get()) is not None:
                yield chunk
        finally:
            # Consumer stopped (finished or cancelled): make sure the sender stops too
            request.cancelled = True

    async def _ensure_connected(self) -> ClientConnection:
        async with self._connect_lock:
            if self._ws is None or self._ws.state is not State.OPEN:
                self._ws = await connect(
                    self._uri,
                    additional_headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
                    ping_interval=10,
                    ping_timeout=5,
                    max_size=2**20,
                    compression=None,
                )
                await self._ws.send(
                    json.dumps(
                        {
                            "text": " ",
                            "voice_settings": self.voice_settings,
                            "xi_api_key": settings.ELEVENLABS_API_KEY,
                        }
                    )
                )
                logger.info(f"ElevenLabs connected | voice_id={self.voice_id} | model={self.model_id}")
            return self._ws

    async def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with suppress(Exception):
                await ws.close()

    def _prewarm(self) -> None:
        if self._closed:
            return

        async def _run():
            try:
                await self._ensure_connected()
            except Exception as e:
                logger.warning(f"ElevenLabs pre-warm failed: {e}")

        self._prewarm_task = asyncio.create_task(_run(), name="tts_prewarm")

    async def _process_requests(self) -> None:
        while True:
            request = await self._requests.get()
            if request is None:
                break
            if request.cancelled:
                await request.finish()
                continue

            self._current = request
            try:
                await self._process_single_request(request)
            except Exception as e:
                logger.error(f"ElevenLabs request failed: {e}")
                await request.finish()
            finally:
                self._current = None

            self._prewarm()

    async def _process_single_request(self, request: TTSRequest) -> None:
        ws = await self._ensure_connected()
        sender = asyncio.create_task(self._send_text(ws, request))
        receiver = asyncio.create_task(self._receive_audio(ws, request))

        try:
            await receiver
            # In the normal flow the sender finished before the final audio arrived.
            # If the receiver stopped early (cancel, error), don't wait on the LLM.
            if not sender.done():
                sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
        finally:
            # The socket is spent after end-of-input (or a cancel); close this one only,
            # not one that a barge-in may have already pre-warmed
            with suppress(Exception):
                await ws.close()

    async def _send_text(self, ws: ClientConnection, request: TTSRequest) -> None:
        parts: list[str] = []
        try:
            async for text in request.text_iterator:
                if request.cancelled:
                    return
                if not parts:
                    self.call_logger.add_log("tts", "first_text_chunk_sent", f"'{text[:50]}'")
                parts.append(text)
                await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))

            if request.cancelled:
                return

            # Empty text = end of input; ElevenLabs flushes, sends isFinal and closes
            await ws.send(json.dumps({"text": ""}))

            spoken = " ".join("".join(parts).split())
            self.call_logger.add_log("tts", "last_text_chunk_sent", f"'{spoken[:100]}'")
            if request.on_complete and spoken:
                await request.on_complete(spoken)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error sending text to ElevenLabs: {e}")
            # Unblock the receiver instead of letting it wait for an isFinal that won't come
            with suppress(Exception):
                await ws.close()
        finally:
            aclose = getattr(request.text_iterator, "aclose", None)
            if aclose:
                with suppress(Exception):
                    await aclose()

    async def _receive_audio(self, ws: ClientConnection, request: TTSRequest) -> None:
        first_audio = True
        try:
            while not request.cancelled:
                message = await asyncio.wait_for(ws.recv(), timeout=RECEIVE_TIMEOUT_S)
                data = json.loads(message)

                if data.get("audio"):
                    if first_audio:
                        self.call_logger.add_log("tts", "first_audio_received", "First audio chunk received")
                        first_audio = False
                    audio = base64.b64decode(data["audio"])
                    if request.timeline:
                        request.timeline.add_audio(len(audio), data.get("alignment"))
                    await request.audio_queue.put(audio)

                if data.get("isFinal"):
                    self.call_logger.add_log("tts", "last_audio_received", "Last audio chunk received")
                    break
        except ConnectionClosed:
            if not request.cancelled:
                logger.warning("ElevenLabs connection closed before the final audio")
        except TimeoutError:
            logger.warning("ElevenLabs receive timeout")
        except Exception as e:
            logger.error(f"Error receiving audio from ElevenLabs: {e}")
        finally:
            await request.finish()

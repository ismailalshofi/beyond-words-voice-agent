import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from starlette.websockets import WebSocketState

from app.models import Assistant
from app.services import llm
from app.settings import settings
from app.voice.call_logger import CallLogger
from app.voice.elevenlabs_tts import ElevenLabsTTS
from app.voice.spoken_timeline import SpokenTimeline
from app.voice.stt import SpeechToText, create_stt
from app.voice.text_chunker import chunk_for_tts

HISTORY_LIMIT = 30
KEEP_ALIVE_INTERVAL_S = 5
# The browser holds the first chunk in a small jitter buffer before playing it (see static/app.js)
PLAYBACK_START_DELAY_MS = 60


class CallSession:
    """One live voice call between a browser and an assistant.

    Browser -> server:
      binary  16 kHz mono PCM16 microphone audio
      {"type": "playback_done", "id": n}   all audio for response n has finished playing
      {"type": "hangup"}

    Server -> browser:
      binary  16 kHz mono PCM16 speech
      {"type": "ready"}
      {"type": "interim", "text"}           what the caller is saying right now
      {"type": "transcript", "role", "text"}
      {"type": "audio_end", "id": n}        no more audio for response n
      {"type": "clear"}                     caller barged in: drop queued audio now
      {"type": "error", "message"}
    """

    def __init__(
        self,
        assistant: Assistant,
        ws: WebSocket,
        *,
        tts: ElevenLabsTTS | None = None,
        stt: SpeechToText | None = None,
    ):
        self.assistant = assistant
        self.ws = ws
        self.call_logger = CallLogger()
        self.history: deque[dict] = deque(maxlen=HISTORY_LIMIT)

        self.tts = tts or ElevenLabsTTS(
            voice_id=assistant.voice_id or settings.DEFAULT_VOICE_ID,
            call_logger=self.call_logger,
        )
        self.stt = stt or create_stt(
            settings.STT_PROVIDER,
            language=assistant.language,
            on_speech=self.on_speech,
            on_utterance=self.on_utterance,
        )

        # True from the moment a response starts until the browser reports playback finished
        self.is_agent_speaking = False
        self.is_active = False

        self._response_id = 0
        self._respond_task: asyncio.Task | None = None
        # What the current response said, and when its audio started, so history records what was heard
        self._timeline = SpokenTimeline()
        self._reply_text: str | None = None
        self._first_audio_at: float | None = None
        self._reply_committed = True
        self._background_tasks: list[asyncio.Task] = []
        self._send_lock = asyncio.Lock()
        self._hung_up = False

    async def run(self) -> None:
        await self.ws.accept()
        try:
            await self.start()
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self.stt.send_audio(message["bytes"])
                elif message.get("text"):
                    if not await self._handle_control(json.loads(message["text"])):
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Call failed")
            await self._send_event({"type": "error", "message": "The call failed. Check the server logs."})
        finally:
            await self.hang_up()

    async def start(self) -> None:
        self.is_active = True
        # Open both provider sockets in parallel to keep call setup fast
        await asyncio.gather(self.tts.connect(), self.stt.connect())
        self._background_tasks.append(asyncio.create_task(self._keep_alive(), name="keep_alive"))
        await self._send_event({"type": "ready"})
        self._greet()

    async def _handle_control(self, event: dict) -> bool:
        """Returns False when the call should end."""
        if event.get("type") == "playback_done":
            # Ignore late acks for responses that were already replaced
            if event.get("id") == self._response_id:
                self.is_agent_speaking = False
                self._commit_reply(interrupted=False)
        elif event.get("type") == "hangup":
            return False
        return True

    def _greet(self) -> None:
        greeting = (self.assistant.call_greeting or "").strip()
        if greeting:
            self._start_response(_single(greeting))
        else:
            self.call_logger.add_log("llm", "call_initiated", "Generating greeting")
            self._start_response(self._llm_stream(llm.greeting_messages(self.assistant, "call")))

    async def on_speech(self, text: str) -> None:
        """Any recognised speech. Barge in if the assistant is talking.

        A single word ("yeah", "mm-hmm") or a noise misheard as a word is usually not an
        interruption, so it takes a few words. A short completed turn still interrupts
        through `on_utterance`.
        """
        await self._send_event({"type": "interim", "text": text})
        if self.is_agent_speaking and len(text.split()) >= settings.BARGE_IN_MIN_WORDS:
            await self._interrupt()

    async def on_utterance(self, text: str) -> None:
        """A completed user turn."""
        if self.is_agent_speaking:
            await self._interrupt()

        self.call_logger.add_log("transcription", "transcription_complete", f"Transcript: '{text}'")
        await self._send_event({"type": "transcript", "role": "user", "text": text})

        self.history.append({"role": "user", "content": text})
        messages = llm.build_messages(self.assistant, self.history, "call")

        self.call_logger.add_log("llm", "call_initiated", "LLM call started")
        self._start_response(self._llm_stream(messages))

    def _llm_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        return llm.stream_reply(
            messages,
            model=self.assistant.llm_model,
            max_tokens=settings.CALL_MAX_TOKENS,
        )

    def _start_response(self, text_stream: AsyncIterator[str]) -> None:
        self._response_id += 1
        self.is_agent_speaking = True
        self._timeline = SpokenTimeline()
        self._reply_text = None
        self._first_audio_at = None
        self._reply_committed = False
        self._respond_task = asyncio.create_task(self._respond(text_stream, self._response_id), name="respond")

    async def _respond(self, text_stream: AsyncIterator[str], response_id: int) -> None:
        async def on_complete(text: str) -> None:
            # History waits until playback finishes or the caller interrupts (see _commit_reply)
            self._reply_text = text
            await self._send_event({"type": "transcript", "role": "assistant", "text": text})

        try:
            audio = self.tts.stream(chunk_for_tts(self._track_llm(text_stream)), on_complete, self._timeline)
            first_chunk = True
            async for chunk in audio:
                if first_chunk:
                    self._first_audio_at = asyncio.get_running_loop().time()
                    self.call_logger.add_log("client", "first_chunk_sent", f"First chunk sent ({len(chunk)} bytes)")
                    first_chunk = False
                await self._send_bytes(chunk)
            self.call_logger.add_log("client", "last_chunk_sent", "Last chunk sent")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Response failed")

        await self._send_event({"type": "audio_end", "id": response_id})

    async def _track_llm(self, text_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        first = True
        async for chunk in text_stream:
            if first:
                self.call_logger.add_log("llm", "first_chunk_received", "First chunk received")
                first = False
            yield chunk
        self.call_logger.add_log("llm", "last_chunk_received", "Last chunk received")

    def _commit_reply(self, *, interrupted: bool) -> None:
        """Add the current response to history once, as the caller heard it."""
        if self._reply_committed:
            return
        self._reply_committed = True
        text = self._heard_text() if interrupted else self._reply_text
        if text:
            self.history.append({"role": "assistant", "content": text})

    def _heard_text(self) -> str:
        """Estimate what the caller heard before interrupting.

        The browser plays audio as it arrives, so playback position is roughly the time since
        the first chunk was sent, capped by how much audio was sent.
        """
        if self._first_audio_at is None:
            return ""
        elapsed_ms = (asyncio.get_running_loop().time() - self._first_audio_at) * 1000 - PLAYBACK_START_DELAY_MS
        played_ms = min(self._timeline.audio_ms, max(0.0, elapsed_ms))
        if played_ms >= self._timeline.audio_ms and self._reply_text:
            return self._reply_text
        heard = self._timeline.text_before(played_ms)
        self.call_logger.add_log("interruption", "heard_text", f"Caller heard: '{heard[:100]}'")
        return heard

    async def _interrupt(self) -> None:
        self.call_logger.add_log("interruption", "interrupt_triggered", "User interrupted the assistant")
        self._commit_reply(interrupted=True)

        task = self._respond_task
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self.is_agent_speaking = False
        await self.tts.cancel()
        # Drop stale buffered segments so they don't leak into the next turn
        self.stt.reset()
        await self._send_event({"type": "clear"})

    async def _keep_alive(self) -> None:
        while self.is_active:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_S)
            try:
                await self.stt.keep_alive()
            except Exception as e:
                logger.warning(f"STT keep-alive failed: {e}")

    async def hang_up(self) -> None:
        if self._hung_up:
            return
        self._hung_up = True
        self.is_active = False

        tasks = [t for t in (self._respond_task, *self._background_tasks) if t and not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        with suppress(Exception):
            await self.stt.close()
        with suppress(Exception):
            await self.tts.disconnect()

        self.call_logger.print_summary()

        if self.ws.client_state == WebSocketState.CONNECTED:
            with suppress(Exception):
                await self.ws.close()

    async def _send_event(self, event: dict) -> None:
        async with self._send_lock:
            with suppress(Exception):
                await self.ws.send_text(json.dumps(event))

    async def _send_bytes(self, chunk: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(chunk)


async def _single(text: str) -> AsyncIterator[str]:
    yield text

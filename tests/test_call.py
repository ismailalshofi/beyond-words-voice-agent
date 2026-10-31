import asyncio
import json

import pytest
from starlette.websockets import WebSocketState

from app.models import Assistant
from app.services import call as call_module
from app.services.call import CallSession
from app.voice.spoken_timeline import PCM16_16K_BYTES_PER_MS
from app.voice.stt import DeepgramSTT


class FakeWebSocket:
    client_state = WebSocketState.DISCONNECTED

    def __init__(self):
        self.events: list[dict] = []
        self.audio: list[bytes] = []

    async def send_text(self, text):
        self.events.append(json.loads(text))

    async def send_bytes(self, data):
        self.audio.append(data)

    def types(self):
        return [e["type"] for e in self.events]


class FakeTTS:
    """Turns each text chunk into one audio chunk; can be told to hang mid-stream."""

    def __init__(self, hang=False):
        self.hang = hang
        self.cancelled = 0

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def cancel(self):
        self.cancelled += 1

    async def stream(self, text_iterator, on_complete=None, timeline=None):
        parts = []
        async for text in text_iterator:
            parts.append(text)
            if timeline:
                # Pretend every character takes 10 ms to say
                alignment = {"chars": list(text), "charStartTimesMs": [i * 10 for i in range(len(text))]}
                timeline.add_audio(len(text) * 10 * PCM16_16K_BYTES_PER_MS, alignment)
            yield text.encode()
            if self.hang:
                await asyncio.Event().wait()
        if on_complete:
            await on_complete(" ".join("".join(parts).split()))


class FakeSTT:
    def __init__(self):
        self.resets = 0

    async def connect(self):
        pass

    async def send_audio(self, chunk):
        pass

    async def keep_alive(self):
        pass

    def reset(self):
        self.resets += 1

    async def close(self):
        pass


def _assistant(**overrides):
    fields = dict(
        id="a1",
        name="Nova",
        system_prompt="Be brief.",
        call_greeting="Hi, Nova here.",
        chat_greeting="",
        voice_id=None,
        language="en",
        llm_model=None,
    )
    fields.update(overrides)
    return Assistant(**fields)


@pytest.fixture
def fake_llm(monkeypatch):
    seen = []

    async def fake_stream(messages, *, model=None, max_tokens):
        seen.append(messages)
        for token in ["Sure", ",", " happy", " to", " help", "."]:
            yield token

    monkeypatch.setattr(call_module.llm, "stream_reply", fake_stream)
    return seen


async def _wait_for_response(session):
    await session._respond_task


async def test_greeting_is_spoken_and_added_to_history(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)

    await session.start()
    await _wait_for_response(session)

    assert ws.types() == ["ready", "transcript", "audio_end"]
    assert b"".join(ws.audio).decode().strip() == "Hi, Nova here."
    # History waits until the browser has played it
    assert list(session.history) == []
    await session._handle_control({"type": "playback_done", "id": 1})
    assert list(session.history) == [{"role": "assistant", "content": "Hi, Nova here."}]
    assert fake_llm == []
    await session.hang_up()


async def test_utterance_triggers_llm_reply(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    await _wait_for_response(session)
    await session._handle_control({"type": "playback_done", "id": 1})

    await session.on_utterance("Can you help me?")
    await _wait_for_response(session)

    assert fake_llm[0][0]["role"] == "system"
    assert "live voice call" in fake_llm[0][0]["content"]
    assert fake_llm[0][-1] == {"role": "user", "content": "Can you help me?"}
    assert {"type": "transcript", "role": "user", "text": "Can you help me?"} in ws.events
    assert {"type": "transcript", "role": "assistant", "text": "Sure, happy to help."} in ws.events
    assert ws.events[-1] == {"type": "audio_end", "id": 2}
    await session.hang_up()


async def test_speech_while_speaking_interrupts(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(hang=True), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    await asyncio.sleep(0)  # let the greeting start streaming
    assert session.is_agent_speaking

    await session.on_speech("wait a second")

    assert session._respond_task.cancelled()
    assert not session.is_agent_speaking
    assert tts.cancelled == 1
    assert stt.resets == 1
    assert ws.events[-1] == {"type": "clear"}
    # Interrupted before any audio played, so nothing was heard
    assert list(session.history) == []
    await session.hang_up()


async def test_single_word_does_not_barge_in(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(hang=True), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    await asyncio.sleep(0)

    await session.on_speech("yeah")

    assert session.is_agent_speaking
    assert tts.cancelled == 0
    await session.hang_up()


async def test_completed_short_turn_still_interrupts(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(hang=True), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    await asyncio.sleep(0)

    await session.on_utterance("stop")

    assert tts.cancelled == 1
    assert {"role": "user", "content": "stop"} in session.history
    await session.hang_up()


async def test_interrupted_reply_is_recorded_as_heard(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(hang=True), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    while not ws.audio:  # wait for the first audio chunk to go out
        await asyncio.sleep(0)
    # "Hi, Nova here." at 10 ms per character, and the caller has heard 75 ms of it
    session._first_audio_at -= (call_module.PLAYBACK_START_DELAY_MS + 75) / 1000

    await session.on_speech("hold on")

    assert list(session.history) == [{"role": "assistant", "content": "Hi, Nova"}]
    await session.hang_up()


async def test_speech_while_silent_does_not_interrupt(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    await _wait_for_response(session)
    await session._handle_control({"type": "playback_done", "id": 1})

    await session.on_speech("hello")

    assert tts.cancelled == 0
    assert "clear" not in ws.types()
    await session.hang_up()


async def test_stale_playback_ack_is_ignored(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(), FakeSTT()
    session = CallSession(_assistant(), ws, tts=tts, stt=stt)
    await session.start()
    await _wait_for_response(session)
    await session.on_utterance("Hi")
    await _wait_for_response(session)

    await session._handle_control({"type": "playback_done", "id": 1})
    assert session.is_agent_speaking
    await session._handle_control({"type": "playback_done", "id": 2})
    assert not session.is_agent_speaking
    await session.hang_up()


async def test_empty_call_greeting_is_generated(fake_llm):
    ws, tts, stt = FakeWebSocket(), FakeTTS(), FakeSTT()
    session = CallSession(_assistant(call_greeting=""), ws, tts=tts, stt=stt)
    await session.start()
    await _wait_for_response(session)

    assert fake_llm[0][-1]["content"] == call_module.llm.GREETING_INSTRUCTION
    await session._handle_control({"type": "playback_done", "id": 1})
    assert list(session.history) == [{"role": "assistant", "content": "Sure, happy to help."}]
    await session.hang_up()


# ---------- Deepgram utterance buffering ----------


def _deepgram(safety_timeout_s=2.0):
    speech, turns = [], []

    async def on_speech(text):
        speech.append(text)

    async def on_utterance(text):
        turns.append(text)

    stt = DeepgramSTT(language="en", on_speech=on_speech, on_utterance=on_utterance, safety_timeout_s=safety_timeout_s)
    return stt, speech, turns


def _result(text, is_final=True, speech_final=False):
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text}]},
    }


async def test_deepgram_buffers_segments_until_speech_final():
    stt, speech, turns = _deepgram()
    await stt.handle_message(_result("I want to book", is_final=False))
    await stt.handle_message(_result("I want to book"))
    await stt.handle_message(_result("a table for two", speech_final=True))

    assert turns == ["I want to book a table for two"]
    assert speech == ["I want to book", "I want to book", "a table for two"]
    assert stt.utterance_buffer == []


async def test_deepgram_utterance_end_flushes_buffer():
    stt, _, turns = _deepgram()
    await stt.handle_message(_result("hello there"))
    await stt.handle_message({"type": "UtteranceEnd"})
    assert turns == ["hello there"]


async def test_deepgram_safety_timer_flushes_buffer():
    stt, _, turns = _deepgram(safety_timeout_s=0.05)
    await stt.handle_message(_result("are you there"))
    await asyncio.sleep(0.1)
    assert turns == ["are you there"]


async def test_deepgram_safety_timer_survives_turn_that_resets_stt():
    """The timer completes the turn, and the turn handler resets the STT (as a barge-in does).
    The timer must not cancel itself before the handler finishes."""
    finished = []

    async def on_speech(_):
        pass

    async def on_utterance(text):
        stt.reset()
        await asyncio.sleep(0.01)
        finished.append(text)

    stt = DeepgramSTT(language="en", on_speech=on_speech, on_utterance=on_utterance, safety_timeout_s=0.05)
    await stt.handle_message(_result("one more thing"))
    await asyncio.sleep(0.15)
    assert finished == ["one more thing"]

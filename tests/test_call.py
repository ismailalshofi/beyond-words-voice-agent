import asyncio
import json
from pathlib import Path

import httpx
import openai
import pytest
from starlette.websockets import WebSocketState
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from app.listening.listener import Listener
from app.listening.policy import Policy
from app.models import Assistant
from app.services import call as call_module
from app.services.call import CallSession
from app.services.provider_errors import ProviderError
from app.voice.spoken_timeline import PCM16_16K_BYTES_PER_MS
from app.voice.stt import DeepgramSTT

POLICY = Policy.load(Path(__file__).resolve().parents[1] / "policy.toml")


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
        # The live state panel updates are tested separately
        return [e["type"] for e in self.events if e["type"] != "state"]


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
    async def connect(self):
        pass

    async def send_audio(self, chunk):
        pass

    async def keep_alive(self):
        pass

    async def close(self):
        pass


class FakeClips:
    """Every clip is just its kind, so the tests can see which line was played."""

    async def warm(self):
        pass

    def get(self, kind):
        return f"<{kind}>".encode()

    async def close(self):
        pass


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1000


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


def _session(tts=None, **overrides):
    clock = FakeClock()
    ws = FakeWebSocket()
    session = CallSession(
        _assistant(**overrides),
        ws,
        tts=tts or FakeTTS(),
        stt=FakeSTT(),
        policy=POLICY,
        clips=FakeClips(),
        listener=Listener(clock=clock),
    )
    return session, ws, clock


@pytest.fixture(autouse=True)
def slow_policy_loop(monkeypatch):
    # The tests drive the policy themselves with session.decide() and a fake clock
    monkeypatch.setattr(call_module.settings, "POLICY_TICK_MS", 60_000)


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


async def _first_audio(ws):
    while not ws.audio:
        await asyncio.sleep(0)


async def _greeted(session):
    """Start the call and let the greeting play out."""
    await session.start()
    await _wait_for_response(session)
    await session._handle_control({"type": "playback_done", "id": 1})


async def test_greeting_is_spoken_and_added_to_history_after_playback(fake_llm):
    session, ws, _ = _session()
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


async def test_finished_question_is_answered_after_a_normal_pause(fake_llm):
    session, ws, clock = _session()
    await _greeted(session)

    await session.on_final("Can you help me?")
    clock.advance(500)
    await session.decide()
    assert session._response_id == 1, "answered before a normal pause was over"

    clock.advance(300)
    await session.decide()
    await _wait_for_response(session)

    assert fake_llm[0][0]["role"] == "system"
    assert "live voice call" in fake_llm[0][0]["content"]
    assert fake_llm[0][-1] == {"role": "user", "content": "Can you help me?"}
    assert {"type": "transcript", "role": "user", "text": "Can you help me?"} in ws.events
    assert {"type": "transcript", "role": "assistant", "text": "Sure, happy to help."} in ws.events
    assert ws.events[-1] == {"type": "audio_end", "id": 2}
    await session.hang_up()


async def test_unfinished_sentence_gets_a_backchannel_not_an_answer(fake_llm):
    session, ws, clock = _session()
    await _greeted(session)

    await session.on_final("So I was thinking that")
    clock.advance(1100)
    await session.decide()

    assert ws.audio[-1] == b"<backchannel>"
    assert fake_llm == []
    await session.hang_up()


async def test_two_words_while_speaking_interrupt(fake_llm):
    session, ws, _ = _session(tts=FakeTTS(hang=True))
    await session.start()
    await _first_audio(ws)
    assert session.is_agent_speaking

    await session.on_speech("wait a second")

    assert session._respond_task.cancelled()
    assert not session.is_agent_speaking
    assert session.tts.cancelled == 1
    assert ws.events[-1] == {"type": "clear"}
    await session.hang_up()


async def test_single_word_does_not_barge_in(fake_llm):
    session, _, _ = _session(tts=FakeTTS(hang=True))
    await session.start()
    await asyncio.sleep(0)

    await session.on_speech("yeah")

    assert session.is_agent_speaking
    assert session.tts.cancelled == 0
    await session.hang_up()


async def test_interrupted_reply_is_recorded_as_heard(fake_llm):
    session, ws, _ = _session(tts=FakeTTS(hang=True))
    await session.start()
    await _first_audio(ws)
    # "Hi, Nova here." at 10 ms per character, and the caller has heard 75 ms of it
    session._first_audio_at -= (call_module.PLAYBACK_START_DELAY_MS + 75) / 1000

    await session.on_speech("hold on")

    assert list(session.history) == [{"role": "assistant", "content": "Hi, Nova"}]
    await session.hang_up()


async def test_speech_while_silent_does_not_interrupt(fake_llm):
    session, ws, _ = _session()
    await _greeted(session)

    await session.on_speech("hello there")

    assert session.tts.cancelled == 0
    assert "clear" not in ws.types()
    await session.hang_up()


async def test_stale_playback_ack_is_ignored(fake_llm):
    session, _, clock = _session()
    await _greeted(session)
    await session.on_final("Hi.")
    clock.advance(800)
    await session.decide()
    await _wait_for_response(session)

    await session._handle_control({"type": "playback_done", "id": 1})
    assert session.is_agent_speaking
    await session._handle_control({"type": "playback_done", "id": 2})
    assert not session.is_agent_speaking
    await session.hang_up()


async def test_empty_call_greeting_is_generated(fake_llm):
    session, _, _ = _session(call_greeting="")
    await _greeted(session)

    assert fake_llm[0][-1]["content"] == call_module.llm.GREETING_INSTRUCTION
    assert list(session.history) == [{"role": "assistant", "content": "Sure, happy to help."}]
    await session.hang_up()


async def test_llm_failure_plays_the_apology(monkeypatch):
    async def broken_stream(messages, *, model=None, max_tokens):
        raise RuntimeError("provider down")
        yield  # pragma: no cover

    monkeypatch.setattr(call_module.llm, "stream_reply", broken_stream)
    session, ws, _ = _session(call_greeting="")
    await session.start()
    await _wait_for_response(session)

    assert ws.audio == [b"<llm_failed>"]
    assert ws.events[-1] == {"type": "audio_end", "id": 1}
    await session.hang_up()


async def test_rejected_stt_key_is_named_not_a_traceback(monkeypatch):
    class RejectedSTT(FakeSTT):
        async def connect(self):
            raise InvalidStatus(Response(401, "Unauthorized", Headers()))

    monkeypatch.setattr(call_module.settings, "STT_PROVIDER", "deepgram")
    session, _, _ = _session()
    session.stt = RejectedSTT()

    with pytest.raises(ProviderError, match="Deepgram rejected the API key. Check DEEPGRAM_API_KEY in .env."):
        await session.start()
    await session.hang_up()


async def test_rejected_llm_key_tells_the_browser_which_key(monkeypatch):
    async def rejected_stream(messages, *, model=None, max_tokens):
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        raise openai.AuthenticationError("Incorrect API key provided", response=response, body=None)
        yield  # pragma: no cover

    monkeypatch.setattr(call_module.llm, "stream_reply", rejected_stream)
    session, ws, _ = _session(call_greeting="")
    await session.start()
    await _wait_for_response(session)

    assert ws.audio == [b"<llm_failed>"]
    error = {"type": "error", "message": "The LLM provider rejected the API key. Check OPENAI_API_KEY in .env."}
    assert error in ws.events
    await session.hang_up()


async def test_slow_llm_fills_the_silence(monkeypatch):
    async def slow_stream(messages, *, model=None, max_tokens):
        await asyncio.sleep(0.1)
        yield "Here you go."

    monkeypatch.setattr(call_module.llm, "stream_reply", slow_stream)
    monkeypatch.setattr(call_module.settings, "LLM_SLOW_MS", 20)
    session, ws, _ = _session(call_greeting="")
    await session.start()
    await _wait_for_response(session)

    assert ws.audio[0] == b"<llm_slow>"
    assert b"Here you go." in ws.audio[-1]
    await session.hang_up()


async def test_state_is_pushed_to_the_browser(fake_llm):
    session, ws, _ = _session()
    await _greeted(session)

    await session.on_final("Can you help me?")

    state = [e for e in ws.events if e["type"] == "state"][-1]
    assert state["state"]["pending_text"] == "Can you help me?"
    assert state["decision"] == {"action": "wait", "rule": "normal pause"}
    await session.hang_up()


# ---------- Deepgram ----------


def _deepgram():
    speech, finals = [], []

    async def on_speech(text):
        speech.append(text)

    async def on_final(text):
        finals.append(text)

    return DeepgramSTT(language="en", on_speech=on_speech, on_final=on_final), speech, finals


def _result(text, is_final=True):
    return {"type": "Results", "is_final": is_final, "channel": {"alternatives": [{"transcript": text}]}}


async def test_deepgram_splits_interim_and_final_results():
    stt, speech, finals = _deepgram()
    await stt.handle_message(_result("I want to", is_final=False))
    await stt.handle_message(_result("I want to book"))
    await stt.handle_message(_result("", is_final=False))
    await stt.handle_message({"type": "UtteranceEnd"})

    assert speech == ["I want to"]
    assert finals == ["I want to book"]

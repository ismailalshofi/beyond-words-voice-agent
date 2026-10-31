import asyncio
import base64
import json

import pytest
from websockets.asyncio.server import serve

from app.voice.call_logger import CallLogger
from app.voice.elevenlabs_tts import ElevenLabsTTS
from app.voice.spoken_timeline import SpokenTimeline


class FakeElevenLabs:
    """Mimics the stream-input socket: echoes each text chunk back as 'audio', then isFinal on end of input."""

    def __init__(self):
        self.connections = 0
        self.hang = False

    async def handler(self, ws):
        self.connections += 1
        await ws.recv()  # beginning-of-stream message with voice settings
        async for message in ws:
            if self.hang:
                continue  # never answer: no audio, no isFinal
            text = json.loads(message)["text"]
            if text == "":
                await ws.send(json.dumps({"isFinal": True}))
                await ws.close()
                return
            alignment = {"chars": list(text), "charStartTimesMs": [i * 10 for i in range(len(text))]}
            await ws.send(json.dumps({"audio": base64.b64encode(text.encode()).decode(), "alignment": alignment}))


@pytest.fixture
async def fake_server():
    fake = FakeElevenLabs()
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        fake.uri = f"ws://127.0.0.1:{port}"
        yield fake


def _tts(fake, monkeypatch):
    monkeypatch.setattr(ElevenLabsTTS, "_uri", property(lambda self: fake.uri))
    return ElevenLabsTTS(voice_id="v", call_logger=CallLogger())


async def _text(*parts, delay=0.0):
    for part in parts:
        if delay:
            await asyncio.sleep(delay)
        yield part


async def test_streams_audio_and_reports_spoken_text(fake_server, monkeypatch):
    tts = _tts(fake_server, monkeypatch)
    spoken = []

    async def on_complete(text):
        spoken.append(text)

    await tts.connect()
    audio = [chunk async for chunk in tts.stream(_text("Hello ", "there. "), on_complete)]

    assert b"".join(audio) == b"Hello there. "
    assert spoken == ["Hello there."]
    await tts.disconnect()


async def test_back_to_back_requests_reconnect(fake_server, monkeypatch):
    tts = _tts(fake_server, monkeypatch)
    await tts.connect()

    first = [c async for c in tts.stream(_text("One. "))]
    second = [c async for c in tts.stream(_text("Two. "))]

    assert first == [b"One. "]
    assert second == [b"Two. "]
    assert fake_server.connections >= 2
    await tts.disconnect()


async def test_cancel_stops_current_request_and_next_one_works(fake_server, monkeypatch):
    tts = _tts(fake_server, monkeypatch)
    await tts.connect()

    # A slow LLM that would keep talking for a long time
    async def endless():
        while True:
            yield "blah "
            await asyncio.sleep(0.01)

    received = []

    async def consume():
        async for chunk in tts.stream(endless()):
            received.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    assert received

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await tts.cancel()

    after = await asyncio.wait_for(_collect(tts.stream(_text("Fresh start. "))), timeout=2)
    assert after == [b"Fresh start. "]
    await tts.disconnect()


async def test_cancel_unblocks_request_waiting_for_audio(fake_server, monkeypatch):
    fake_server.hang = True
    tts = _tts(fake_server, monkeypatch)
    await tts.connect()

    task = asyncio.create_task(_collect(tts.stream(_text("Hello ", delay=0.01), None)))
    await asyncio.sleep(0.1)
    await tts.cancel()

    # The consumer sees the end of the stream instead of hanging for the receive timeout
    assert await asyncio.wait_for(task, timeout=1) == []
    await tts.disconnect()


async def _collect(stream):
    return [chunk async for chunk in stream]


async def test_fills_timeline_from_alignment(fake_server, monkeypatch):
    tts = _tts(fake_server, monkeypatch)
    timeline = SpokenTimeline(bytes_per_ms=0.1)  # the fake's "audio" is the text: 1 byte = 10 ms
    await tts.connect()

    await _collect(tts.stream(_text("Hello ", "there. "), timeline=timeline))

    assert timeline.audio_ms == 130
    assert timeline.text_before(130) == "Hello there."
    assert timeline.text_before(65) == "Hello"
    await tts.disconnect()

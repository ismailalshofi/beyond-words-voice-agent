# Realtime Voice Agent

[![CI](https://github.com/ismailalshofi/realtime-voice-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ismailalshofi/realtime-voice-agent/actions/workflows/ci.yml)

A cascaded voice agent (streaming STT → LLM → streaming TTS) that you can talk to from the browser and interrupt mid-sentence. Create assistants with a name, a system prompt, and call and chat greetings, then call them or chat with them.

The interesting part isn't the three APIs. It's everything between them: deciding when the caller has finished, starting speech before the LLM is done, stopping cleanly when the caller interrupts, and remembering what the caller actually heard.

```
Browser mic ──PCM16 16 kHz──▶ ┌───────────────┐ ──audio──▶ Deepgram / Scribe (streaming STT)
                              │  FastAPI      │ ◀─transcripts─┘
                              │  websocket    │ ──messages──▶ LLM (OpenAI-compatible, streaming)
Browser speaker ◀──PCM16───── │  CallSession  │ ◀─tokens──┘
                              └───────────────┘ ──text chunks──▶ ElevenLabs (input streaming)
                                                ◀─audio chunks──┘
```

## How a turn works

1. **Listening.** The browser streams 100 ms frames of 16 kHz PCM16 over the websocket. The server forwards them to Deepgram (or ElevenLabs Scribe).
2. **Deciding the user has finished.** Deepgram final segments are buffered until `speech_final`. `UtteranceEnd` acts as a backup for noisy rooms, and a 2-second safety timer flushes the buffer if neither signal arrives.
3. **Thinking.** The completed turn and the last 30 messages go to the LLM as a stream.
4. **Speaking early.** Tokens are grouped by the text chunker. The first chunk is released after about 3 words, so speech starts before the LLM has finished. Later chunks break on punctuation so the voice keeps natural phrasing. The chunks go straight into ElevenLabs' input-streaming socket, and the audio streams back to the browser as it arrives.
5. **Barge-in.** If the caller says at least two words while the assistant is talking (a completed turn of any length also counts), the server:
   - records what the caller actually heard (below)
   - cancels the response
   - drops the ElevenLabs socket and pre-warms a new one
   - clears the STT buffer
   - tells the browser to `clear`, which stops playback at once.
6. **Remembering what was heard.** A reply only goes into the conversation history when the browser reports it finished playing, or when the caller interrupts it. The text generated is not what the caller heard: TTS runs faster than real time, so most of a reply can be synthesized and sent before the caller has heard its first sentence. On barge-in, the server estimates the playback position from the time since the first audio chunk went out, and maps it to text using the character timings ElevenLabs sends with each chunk. History then holds the words the caller heard, cut at a word boundary, so the LLM doesn't assume it said things the caller never heard.
7. **Latency report.** When the call ends, the server prints a timing breakdown for each turn: LLM time to first token, TTS time to first audio, and end-to-end time.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
cp .env.example .env            # then fill in the keys
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t realtime-voice-agent .
docker run --env-file .env -p 8000:8000 realtime-voice-agent
```

Open http://localhost:8000, create an assistant, and start a call or a chat.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM key |
| `OPENAI_BASE_URL` | Optional. Any OpenAI-compatible API, e.g. Groq |
| `LLM_MODEL` | Default model when an assistant doesn't set one |
| `ELEVENLABS_API_KEY` | Text to speech (and Scribe STT) |
| `ELEVENLABS_MODEL_ID` | Defaults to `eleven_flash_v2_5` (lowest latency) |
| `DEFAULT_VOICE_ID` | Voice used when an assistant doesn't set one |
| `STT_PROVIDER` | `deepgram` (default) or `scribe` |
| `DEEPGRAM_API_KEY` | Needed when `STT_PROVIDER=deepgram` |

Data is stored in SQLite (`voice_agent.db`), which is created on first start.

**Speakers or headphones.** On speakers, the browser stops sending the mic while the assistant is talking, so it doesn't hear itself and interrupt. Tick "I'm using headphones" to stream the mic all the time, which lets you interrupt.

## API

REST:

| Method | Path | |
|---|---|---|
| `GET` | `/api/assistants` | List |
| `POST` | `/api/assistants` | Create `{name, system_prompt, call_greeting, chat_greeting, voice_id?, language, llm_model?}` |
| `GET` / `PATCH` / `DELETE` | `/api/assistants/{id}` | Read / update / delete |

If a greeting is left empty, the assistant writes its own opening line.

### Call websocket: `/api/assistants/{id}/call`

| Direction | Message |
|---|---|
| client → server | binary: 16 kHz mono PCM16 mic audio |
| client → server | `{"type": "playback_done", "id": n}` once all audio for response `n` has played |
| client → server | `{"type": "hangup"}` |
| server → client | binary: 16 kHz mono PCM16 speech |
| server → client | `{"type": "ready"}` |
| server → client | `{"type": "interim", "text"}` |
| server → client | `{"type": "transcript", "role", "text"}` |
| server → client | `{"type": "audio_end", "id": n}` |
| server → client | `{"type": "clear"}` |
| server → client | `{"type": "error", "message"}` |

### Chat websocket: `/api/assistants/{id}/chat`

| Direction | Message |
|---|---|
| client → server | `{"type": "message", "text"}` |
| server → client | `{"type": "delta", "text"}` while the reply streams |
| server → client | `{"type": "done", "text"}` with the full reply (the greeting arrives this way too) |
| server → client | `{"type": "error", "message"}` |

## Project layout

```
app/
  main.py              FastAPI app, static UI
  models.py            Assistant table
  routers/             REST CRUD and the call/chat websockets
  services/call.py     CallSession: turn-taking, barge-in, response streaming
  services/chat.py     ChatSession
  services/llm.py      Prompt building and streaming completions
  voice/stt.py         Deepgram (with utterance buffering) and Scribe
  voice/elevenlabs_tts.py  Queued input-streaming TTS with cancel and pre-warm
  voice/text_chunker.py    LLM tokens -> TTS chunks
  voice/spoken_timeline.py Audio position -> the words the caller heard
  voice/call_logger.py     Per-turn latency breakdown
static/                Browser client (AudioWorklet capture, WebAudio playback)
tests/
```

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

No API keys needed: every provider is faked. The tests cover:
- the chunker, including numbers and hyphenated words it must not split
- the REST API and chat websocket, with a fake LLM
- call turn-taking, barge-in, and the barge-in threshold, with fake STT and TTS
- what goes into history after playback and after an interruption
- Deepgram utterance buffering
- the ElevenLabs client and its character timings, against a local fake ElevenLabs socket

## Limitations

- **No auth or rate limiting.** It's built to run locally. Before you expose it publicly, add auth, or anyone who can reach it can spend your API keys.
- **Heard-text is an estimate.** It assumes the browser plays audio continuously from the first chunk. Network jitter or a playback underrun can shift it by a word or two.
- **Barge-in on speakers.** Without headphones, the browser mutes the mic while the assistant talks so it doesn't hear itself. Interrupting needs headphones (or real acoustic echo cancellation).
- **Turn detection is silence-based** (Deepgram endpointing). A semantic turn-detection model would cut fewer callers off mid-thought.

## License

[MIT](LICENSE)

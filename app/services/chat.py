from collections import deque

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from app.models import Assistant
from app.services import llm
from app.settings import settings

HISTORY_LIMIT = 30


class ChatSession:
    """A text chat over a websocket.

    Client -> server: {"type": "message", "text"}
    Server -> client: {"type": "delta", "text"} while streaming, then {"type": "done", "text"} with the full reply,
                      or {"type": "error", "message"}
    """

    def __init__(self, assistant: Assistant, ws: WebSocket):
        self.assistant = assistant
        self.ws = ws
        self.history: deque[dict] = deque(maxlen=HISTORY_LIMIT)

    async def run(self) -> None:
        await self.ws.accept()
        try:
            await self._greet()
            while True:
                event = await self.ws.receive_json()
                text = str(event.get("text", "")).strip() if event.get("type") == "message" else ""
                if not text:
                    continue
                self.history.append({"role": "user", "content": text})
                await self._reply(llm.build_messages(self.assistant, self.history, "chat"))
        except WebSocketDisconnect:
            pass

    async def _greet(self) -> None:
        greeting = (self.assistant.chat_greeting or "").strip()
        if greeting:
            self.history.append({"role": "assistant", "content": greeting})
            await self.ws.send_json({"type": "done", "text": greeting})
        else:
            await self._reply(llm.greeting_messages(self.assistant, "chat"))

    async def _reply(self, messages: list[dict]) -> None:
        parts: list[str] = []
        try:
            async for delta in llm.stream_reply(
                messages,
                model=self.assistant.llm_model,
                max_tokens=settings.CHAT_MAX_TOKENS,
            ):
                parts.append(delta)
                await self.ws.send_json({"type": "delta", "text": delta})
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.exception("Chat reply failed")
            await self.ws.send_json(
                {"type": "error", "message": "The assistant could not reply. Check the server logs."}
            )
            return

        reply = "".join(parts).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        await self.ws.send_json({"type": "done", "text": reply})

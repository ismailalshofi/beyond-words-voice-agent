from fastapi import APIRouter, WebSocket

from app.db import SessionFactory
from app.models import Assistant
from app.services.call import CallSession
from app.services.chat import ChatSession

router = APIRouter(prefix="/api/assistants", tags=["realtime"])

NOT_FOUND_CLOSE_CODE = 4404


async def _load_assistant(assistant_id: str) -> Assistant | None:
    # Short-lived session: don't hold a DB connection for the length of a call
    async with SessionFactory() as session:
        return await session.get(Assistant, assistant_id)


async def _reject(ws: WebSocket) -> None:
    await ws.accept()
    await ws.close(code=NOT_FOUND_CLOSE_CODE, reason="Assistant not found")


@router.websocket("/{assistant_id}/chat")
async def chat_ws(ws: WebSocket, assistant_id: str):
    assistant = await _load_assistant(assistant_id)
    if not assistant:
        return await _reject(ws)
    await ChatSession(assistant, ws).run()


@router.websocket("/{assistant_id}/call")
async def call_ws(ws: WebSocket, assistant_id: str):
    assistant = await _load_assistant(assistant_id)
    if not assistant:
        return await _reject(ws)
    await CallSession(assistant, ws).run()

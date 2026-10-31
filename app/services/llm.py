from collections.abc import AsyncIterator, Iterable
from typing import Literal

from loguru import logger
from openai import AsyncOpenAI

from app.models import Assistant
from app.settings import settings

Channel = Literal["call", "chat"]

CALL_STYLE = (
    "You are talking on a live voice call. Reply the way a person speaks: short, natural sentences, "
    "usually one to three. Never use markdown, lists, emojis or URLs. "
    "Say numbers, symbols and abbreviations the way you would pronounce them."
)

GREETING_INSTRUCTION = "[The conversation has just started. Greet the user in one short sentence, in character.]"

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    return _client


def system_prompt(assistant: Assistant, channel: Channel) -> str:
    prompt = f"Your name is {assistant.name}."
    if assistant.system_prompt.strip():
        prompt += f"\n\n{assistant.system_prompt.strip()}"
    if channel == "call":
        prompt += f"\n\n{CALL_STYLE}"
    return prompt


def build_messages(assistant: Assistant, history: Iterable[dict], channel: Channel) -> list[dict]:
    return [{"role": "system", "content": system_prompt(assistant, channel)}, *history]


def greeting_messages(assistant: Assistant, channel: Channel) -> list[dict]:
    return build_messages(assistant, [{"role": "user", "content": GREETING_INSTRUCTION}], channel)


async def stream_reply(messages: list[dict], *, model: str | None = None, max_tokens: int) -> AsyncIterator[str]:
    """Yield text deltas from the chat completion stream."""
    stream = await get_client().chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        stream=True,
        max_completion_tokens=max_tokens,
    )
    try:
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        logger.exception("LLM stream failed")
        raise
    finally:
        await stream.close()

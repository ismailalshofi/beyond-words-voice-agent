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

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Model and reasoning effort when .env doesn't set them.
# Gemini Flash thinks by default, which adds seconds before the first word: off for a voice call.
DEFAULTS = {
    "openai": {"model": "gpt-4.1-mini", "reasoning_effort": ""},
    "google": {"model": "gemini-2.5-flash", "reasoning_effort": "none"},
}

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if settings.LLM_PROVIDER == "google":
            _client = AsyncOpenAI(api_key=settings.GOOGLE_API_KEY, base_url=GOOGLE_BASE_URL)
        else:
            _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    return _client


def default_model() -> str:
    return settings.LLM_MODEL or DEFAULTS[settings.LLM_PROVIDER]["model"]


def reasoning_effort() -> str:
    return settings.LLM_REASONING_EFFORT or DEFAULTS[settings.LLM_PROVIDER]["reasoning_effort"]


def system_prompt(assistant: Assistant, channel: Channel) -> str:
    prompt = f"Your name is {assistant.name}."
    if assistant.system_prompt.strip():
        prompt += f"\n\n{assistant.system_prompt.strip()}"
    if channel == "call":
        prompt += f"\n\n{CALL_STYLE}"
    return prompt


def build_messages(assistant: Assistant, history: Iterable[dict], channel: Channel, guidance: str = "") -> list[dict]:
    """`guidance` comes from the policy rule that decided to respond, e.g. "keep it to one sentence"."""
    prompt = system_prompt(assistant, channel)
    if guidance:
        prompt += f"\n\nFor this reply: {guidance}"
    return [{"role": "system", "content": prompt}, *history]


def greeting_messages(assistant: Assistant, channel: Channel) -> list[dict]:
    return build_messages(assistant, [{"role": "user", "content": GREETING_INSTRUCTION}], channel)


async def stream_reply(messages: list[dict], *, model: str | None = None, max_tokens: int) -> AsyncIterator[str]:
    """Yield text deltas from the chat completion stream."""
    extra = {"reasoning_effort": reasoning_effort()} if reasoning_effort() else {}
    stream = await get_client().chat.completions.create(
        model=model or default_model(),
        messages=messages,
        stream=True,
        max_completion_tokens=max_tokens,
        **extra,
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

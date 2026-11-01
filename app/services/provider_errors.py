"""Turn a provider failure into one line the person running the agent can act on."""

import openai
from websockets.exceptions import InvalidStatus

from app.settings import settings

# (name shown to the user, the .env key it needs)
TTS = ("ElevenLabs", "ELEVENLABS_API_KEY")
LLM = ("The LLM provider", "OPENAI_API_KEY")
STT = {
    "deepgram": ("Deepgram", "DEEPGRAM_API_KEY"),
    "scribe": ("ElevenLabs Scribe", "ELEVENLABS_API_KEY"),
}


class ProviderError(Exception):
    """A provider failed in a way the person running the agent can fix."""


def stt_provider() -> tuple[str, str]:
    return STT.get(settings.STT_PROVIDER, (settings.STT_PROVIDER, "the STT key"))


def status_of(error: Exception) -> int | None:
    if isinstance(error, InvalidStatus):
        return error.response.status_code
    if isinstance(error, openai.APIStatusError):
        return error.status_code
    return None


def is_key_problem(error: Exception) -> bool:
    return status_of(error) in (401, 403)


def explain(provider: tuple[str, str], error: Exception) -> str:
    name, key = provider
    status = status_of(error)
    if status in (401, 403):
        return f"{name} rejected the API key. Check {key} in .env."
    if status == 402:
        return f"{name} says the account is out of credit."
    if status == 429:
        return f"{name} is rate limiting the requests. Try again in a minute."
    if isinstance(error, openai.APIConnectionError):
        return f"Could not reach {name}. Check the internet connection or OPENAI_BASE_URL."
    return f"{name} failed: {error}"

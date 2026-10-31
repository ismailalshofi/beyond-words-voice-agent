from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "sqlite+aiosqlite:///./voice_agent.db"

    # LLM (any OpenAI-compatible endpoint works, e.g. Groq via OPENAI_BASE_URL)
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str | None = None
    LLM_MODEL: str = "gpt-4.1-mini"
    CALL_MAX_TOKENS: int = 250
    CHAT_MAX_TOKENS: int = 1024

    # Text to speech
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_MODEL_ID: str = "eleven_flash_v2_5"
    DEFAULT_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"

    # Speech to text
    STT_PROVIDER: Literal["deepgram", "scribe"] = "deepgram"
    DEEPGRAM_API_KEY: str = ""

    # Words of interim speech needed to interrupt the assistant mid-sentence
    BARGE_IN_MIN_WORDS: int = 2


settings = Settings()

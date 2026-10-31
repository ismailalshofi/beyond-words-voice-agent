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

    # The policy: rules for when the agent speaks, waits, says "mm" or stops
    POLICY_FILE: str = "policy.toml"
    POLICY_TICK_MS: int = 100
    # No LLM token after this long: say a short "One moment." so the line isn't dead
    LLM_SLOW_MS: int = 1500

    # Every call's events and latencies are saved here as JSON. Empty to disable.
    CALLS_DIR: str = "calls"


settings = Settings()

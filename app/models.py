from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Assistant(Base):
    __tablename__ = "assistants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(100))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    call_greeting: Mapped[str] = mapped_column(Text, default="")
    chat_greeting: Mapped[str] = mapped_column(Text, default="")
    # ElevenLabs voice; falls back to settings.DEFAULT_VOICE_ID
    voice_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str] = mapped_column(String(16), default="en")
    # Falls back to settings.LLM_MODEL, then the provider's default
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AssistantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    system_prompt: str = ""
    call_greeting: str = ""
    chat_greeting: str = ""
    voice_id: str | None = None
    language: str = "en"
    llm_model: str | None = None


class AssistantUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    system_prompt: str | None = None
    call_greeting: str | None = None
    chat_greeting: str | None = None
    voice_id: str | None = None
    language: str | None = None
    llm_model: str | None = None


class AssistantOut(AssistantCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime

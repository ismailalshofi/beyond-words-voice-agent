import httpx
import openai
import pytest

from app.services import llm, provider_errors


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream()


class FakeStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self):
        pass


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    return fake.chat.completions


@pytest.fixture
def google(monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", "google")
    monkeypatch.setattr(llm.settings, "GOOGLE_API_KEY", "g-key")
    monkeypatch.setattr(llm, "_client", None)
    yield
    llm._client = None


async def _drain(**kwargs):
    async for _ in llm.stream_reply([{"role": "user", "content": "hi"}], max_tokens=50, **kwargs):
        pass


async def test_openai_by_default_with_no_reasoning_effort(client):
    await _drain()

    assert client.kwargs["model"] == "gpt-4.1-mini"
    assert "reasoning_effort" not in client.kwargs


async def test_google_uses_gemini_with_thinking_off(google, client):
    await _drain()

    assert client.kwargs["model"] == "gemini-2.5-flash"
    assert client.kwargs["reasoning_effort"] == "none"


async def test_assistant_model_and_env_settings_win(google, client, monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_REASONING_EFFORT", "low")
    await _drain(model="gemini-2.5-pro")

    assert client.kwargs["model"] == "gemini-2.5-pro"
    assert client.kwargs["reasoning_effort"] == "low"


def test_google_client_points_at_the_gemini_endpoint(google):
    client = llm.get_client()

    assert str(client.base_url) == llm.GOOGLE_BASE_URL
    assert client.api_key == "g-key"


def test_rejected_google_key_names_google_api_key(google):
    # What Gemini really sends for a bad key: 400, not 401
    response = httpx.Response(400, request=httpx.Request("POST", llm.GOOGLE_BASE_URL))
    error = openai.BadRequestError("API key not valid. Please pass a valid API key.", response=response, body=None)

    message = provider_errors.explain(provider_errors.llm_provider(), error)

    assert message == "Google Gemini rejected the API key. Check GOOGLE_API_KEY in .env."

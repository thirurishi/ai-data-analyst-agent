from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest
from fastapi import HTTPException

from backend.app.api.routes.llm import chat
from backend.app.core.config import settings
from backend.app.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from backend.app.llm.openrouter_client import OpenRouterClient
from backend.app.llm.schemas import LLMChatRequest, LLMChatResponse


class FakeProviderResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def read(self) -> bytes:
        return self.payload


def valid_provider_payload() -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "model": "openai/test-model",
            "choices": [{"message": {"role": "assistant", "content": "Hello back."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    ).encode("utf-8")


def configure_provider(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret-key")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", "openai/test-model")


def test_chat_request_schema_accepts_and_normalizes_a_valid_message():
    request = LLMChatRequest(message="  Hello  ")

    assert request.message == "Hello"


def test_chat_request_schema_rejects_blank_message():
    with pytest.raises(ValueError):
        LLMChatRequest(message="   ")


def test_openrouter_client_returns_text_model_and_usage(monkeypatch):
    configure_provider(monkeypatch)
    request_data = {}

    def fake_urlopen(request, timeout):
        request_data["request"] = request
        request_data["timeout"] = timeout
        return FakeProviderResponse(valid_provider_payload())

    monkeypatch.setattr("backend.app.llm.openrouter_client.urlopen", fake_urlopen)

    result = OpenRouterClient().chat("Hello")

    assert result.text == "Hello back."
    assert result.model == "openai/test-model"
    assert result.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert request_data["request"].headers["Authorization"] == "Bearer test-secret-key"
    assert json.loads(request_data["request"].data)["messages"] == [{"role": "user", "content": "Hello"}]
    assert request_data["timeout"] > 0


def test_openrouter_client_requires_api_key_without_making_request(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "backend.app.llm.openrouter_client.urlopen",
        lambda *args, **kwargs: pytest.fail("Provider request must not be made without an API key."),
    )

    with pytest.raises(LLMConfigurationError) as error:
        OpenRouterClient().chat("Hello")

    assert "test-secret-key" not in str(error.value)


def test_openrouter_client_sanitizes_provider_http_errors(monkeypatch):
    configure_provider(monkeypatch)

    def fail_with_http_error(request, timeout):
        raise HTTPError(request.full_url, 401, "test-secret-key leaked", {}, io.BytesIO(b"test-secret-key"))

    monkeypatch.setattr("backend.app.llm.openrouter_client.urlopen", fail_with_http_error)

    with pytest.raises(LLMProviderError) as error:
        OpenRouterClient().chat("Hello")

    assert "test-secret-key" not in str(error.value)
    assert "leaked" not in str(error.value)


def test_openrouter_client_maps_timeout_to_application_error(monkeypatch):
    configure_provider(monkeypatch)
    monkeypatch.setattr(
        "backend.app.llm.openrouter_client.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("test-secret-key")),
    )

    with pytest.raises(LLMTimeoutError) as error:
        OpenRouterClient().chat("Hello")

    assert "test-secret-key" not in str(error.value)


@pytest.mark.parametrize("response_body", [b"not-json", b'{"choices": []}', b'{"choices": [{"message": {"content": 4}}]}'])
def test_openrouter_client_rejects_malformed_provider_response(monkeypatch, response_body):
    configure_provider(monkeypatch)
    monkeypatch.setattr(
        "backend.app.llm.openrouter_client.urlopen",
        lambda *args, **kwargs: FakeProviderResponse(response_body),
    )

    with pytest.raises(LLMResponseError):
        OpenRouterClient().chat("Hello")


def test_chat_route_returns_validated_response_schema():
    class StubLLMService:
        def chat(self, message: str) -> LLMChatResponse:
            assert message == "Hello"
            return LLMChatResponse(
                text="Hello back.",
                model="openai/test-model",
                usage={"total_tokens": 3},
            )

    result = chat(LLMChatRequest(message="Hello"), service=StubLLMService())
    response = LLMChatResponse.model_validate(result)

    assert response.text == "Hello back."
    assert response.model == "openai/test-model"
    assert response.usage == {"total_tokens": 3}


def test_chat_route_does_not_expose_provider_error_details():
    class FailingLLMService:
        def chat(self, message: str):
            raise LLMProviderError("test-secret-key provider diagnostic")

    with pytest.raises(HTTPException) as error:
        chat(LLMChatRequest(message="Hello"), service=FailingLLMService())

    assert error.value.status_code == 502
    assert "test-secret-key" not in str(error.value.detail)

from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.app.core.config import settings
from backend.app.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from backend.app.llm.schemas import LLMChatResponse

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 30


class OpenRouterClient:
    def chat(self, message: str) -> LLMChatResponse:
        api_key = (settings.OPENROUTER_API_KEY or "").strip()
        if not api_key:
            raise LLMConfigurationError("The LLM provider is not configured.")

        request_body = json.dumps(
            {
                "model": settings.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": message}],
            }
        ).encode("utf-8")
        request = Request(
            OPENROUTER_CHAT_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response_body = response.read()
        except HTTPError:
            raise LLMProviderError("The LLM provider returned an HTTP error.") from None
        except (TimeoutError, socket.timeout):
            raise LLMTimeoutError("The LLM provider request timed out.") from None
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError("The LLM provider request timed out.") from None
            raise LLMProviderError("The LLM provider could not be reached.") from None

        try:
            payload = json.loads(response_body)
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
            model = payload.get("model")
            usage = payload.get("usage")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            raise LLMResponseError("The LLM provider returned an invalid response.") from None

        if not isinstance(content, str):
            raise LLMResponseError("The LLM provider returned an invalid response.")
        if model is not None and not isinstance(model, str):
            raise LLMResponseError("The LLM provider returned an invalid response.")
        if usage is not None and not isinstance(usage, dict):
            raise LLMResponseError("The LLM provider returned an invalid response.")

        return LLMChatResponse(text=content, model=model, usage=usage)

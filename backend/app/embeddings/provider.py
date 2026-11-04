from __future__ import annotations

import json
import socket
from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingConfigurationError(ValueError):
    pass


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input text."""


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    DEFAULT_API_URL = "https://api.openai.com/v1/embeddings"
    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_API_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        normalized_key = (api_key or "").strip()
        normalized_model = (model or "").strip()
        if not normalized_key:
            raise EmbeddingConfigurationError("An embedding API key is required for the provider.")
        if not normalized_model:
            raise EmbeddingConfigurationError("An embedding model is required for the provider.")
        if not base_url.strip():
            raise EmbeddingConfigurationError("EMBEDDING_API_BASE_URL must not be empty.")
        self.api_key = normalized_key
        self.model = normalized_model
        self.base_url = base_url.strip()
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        request_body = json.dumps(
            {
                "model": self.model,
                "input": texts,
            }
        ).encode("utf-8")
        request = Request(
            self.base_url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload: Any = json.loads(response.read())
        except HTTPError as exc:
            raise EmbeddingProviderError("The embedding provider returned an HTTP error.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise EmbeddingProviderError("The embedding provider request timed out.") from exc
        except URLError as exc:
            raise EmbeddingProviderError("The embedding provider could not be reached.") from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise EmbeddingProviderError("The embedding provider returned an invalid response.") from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingProviderError("The embedding provider returned an invalid response.")
        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbeddingProviderError("The embedding provider returned an invalid response.")
            vectors.append([float(value) for value in item["embedding"]])
        return vectors

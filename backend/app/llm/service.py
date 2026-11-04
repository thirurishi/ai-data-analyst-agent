from backend.app.llm.openrouter_client import OpenRouterClient
from backend.app.llm.schemas import LLMChatResponse


class LLMService:
    def __init__(self, client=None):
        self._client = client or OpenRouterClient()

    def chat(self, message: str) -> LLMChatResponse:
        return self._client.chat(message)

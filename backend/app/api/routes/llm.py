from fastapi import APIRouter, Depends, HTTPException

from backend.app.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from backend.app.llm.schemas import LLMChatRequest, LLMChatResponse
from backend.app.llm.service import LLMService

router = APIRouter()


def get_llm_service() -> LLMService:
    return LLMService()


@router.post("/llm/chat", response_model=LLMChatResponse)
def chat(payload: LLMChatRequest, service: LLMService = Depends(get_llm_service)):
    try:
        return service.chat(payload.message)
    except LLMConfigurationError:
        raise HTTPException(status_code=503, detail="The LLM provider is not configured.") from None
    except LLMTimeoutError:
        raise HTTPException(status_code=504, detail="The LLM provider request timed out.") from None
    except (LLMProviderError, LLMResponseError):
        raise HTTPException(status_code=502, detail="The LLM provider request failed.") from None

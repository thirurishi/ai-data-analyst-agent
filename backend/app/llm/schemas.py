from pydantic import BaseModel, Field, field_validator


class LLMChatRequest(BaseModel):
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be blank.")
        return message


class LLMChatResponse(BaseModel):
    text: str
    model: str | None = None
    usage: dict | None = None

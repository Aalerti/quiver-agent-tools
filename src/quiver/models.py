from pydantic import BaseModel, Field


class Tool(BaseModel):
    name: str
    description: str


class ToolChoice(BaseModel):
    tool: str
    reason: str


class SelectionRequest(BaseModel):
    task: str = Field(min_length=1)
    temperature: float = Field(default=0.15, ge=0, le=1)


class Selection(BaseModel):
    tools: list[ToolChoice]

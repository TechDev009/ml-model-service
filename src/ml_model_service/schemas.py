"""Typed API request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: Annotated[str, Field(min_length=1, max_length=2_000)]

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("text must contain a non-whitespace character")
        return value


class PredictionResponse(BaseModel):
    label: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    probabilities: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]
    backend: str
    model_version: str


class ModelResponse(BaseModel):
    backend: str
    model_version: str
    labels: list[str]
    algorithm: str
    artifact: str
    device: str | None = None

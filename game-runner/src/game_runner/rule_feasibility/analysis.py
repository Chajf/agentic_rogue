"""Bound and record LLM requests separately from gameplay."""

import asyncio
from dataclasses import dataclass, field
import json
import os
import time
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

from game_runner.pilot.manifest import ManifestModel, PositiveInt

from .artifacts import FeasibilityError


class ModelSettings(ManifestModel):
    model_name: Annotated[str, Field(min_length=1)]
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0
    output_tokens: PositiveInt = 2048
    input_usd_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    output_usd_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


@dataclass
class AnalysisRun:
    model: object
    settings: ModelSettings
    max_calls: int
    seconds: int
    calls: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.max_calls <= 0 or self.seconds <= 0:
            raise FeasibilityError("analysis requires positive call and time limits")
        self.deadline = time.monotonic() + self.seconds

    def available(self, slots: int = 1) -> bool:
        return len(self.calls) + slots <= self.max_calls and time.monotonic() < self.deadline

    async def request(self, label: str, version: str, messages: list[BaseMessage]) -> tuple[dict, object]:
        if not self.available():
            raise FeasibilityError("analysis budget is exhausted")
        start = time.monotonic()
        call = {
            "id": len(self.calls) + 1, "label": label, "prompt_version": version,
            "input_messages": [{"role": message.type, "content": message.content} for message in messages],
            "raw_response": None, "usage": None, "estimated_cost_usd": None,
        }
        self.calls.append(call)
        content = None
        try:
            async with asyncio.timeout(max(0, self.deadline - start)):
                reply = await self.model.ainvoke(messages)
            if not isinstance(reply, AIMessage):
                raise TypeError("expected AIMessage")
            # Preserve model content and usage. Provider transport metadata is
            # deliberately excluded from the offline artifact.
            call.update(raw_response=reply.content, usage=reply.usage_metadata, status="returned")
            content = reply.content
            call["estimated_cost_usd"] = self._cost(reply.usage_metadata)
        except Exception as error:
            call.update(status="error", error=type(error).__name__)
        call["duration_ms"] = round((time.monotonic() - start) * 1000)
        return call, content

    def _cost(self, usage: dict | None) -> float | None:
        rates = self.settings.input_usd_per_million, self.settings.output_usd_per_million
        if usage is None or None in rates:
            return None
        return round((usage["input_tokens"] * rates[0] + usage["output_tokens"] * rates[1]) / 1_000_000, 9)

    def provenance(self) -> dict:
        return {
            "model": self.settings.model_dump(mode="json"),
            "limits": {"max_calls": self.max_calls, "seconds": self.seconds},
            "calls": self.calls,
        }


def provider(settings: ModelSettings):
    from langchain.chat_models import init_chat_model

    runtime = json.loads(os.getenv("MODEL_KWARGS", "{}"))
    headers = {"default_headers": runtime["default_headers"]} if "default_headers" in runtime else {}
    return init_chat_model(
        settings.model_name, model_provider="openai", base_url=os.environ["MODEL_BASE_URL"],
        api_key=os.getenv("MODEL_API_KEY", "local"), temperature=settings.temperature,
        max_tokens=settings.output_tokens, max_retries=0, **headers,
    )

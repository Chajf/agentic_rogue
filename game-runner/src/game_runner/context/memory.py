"""Structured episode summaries and evidence validation."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
POLICY_VERSION = "episode-memory-v1"
EventID = Annotated[int, Field(strict=True, gt=0)]


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=280)
    source_event_ids: list[EventID] = Field(min_length=1, max_length=8)


class EpisodeMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[MemoryEntry] = Field(max_length=12)
    goals: list[MemoryEntry] = Field(max_length=6)
    unsuccessful_attempts: list[MemoryEntry] = Field(max_length=8)
    hypotheses: list[MemoryEntry] = Field(max_length=8)
    uncertainties: list[MemoryEntry] = Field(max_length=8)

    def check_sources(self, allowed: set[int], gap_events: set[int]) -> None:
        for field in type(self).model_fields:
            for entry in getattr(self, field):
                sources = set(entry.source_event_ids)
                if sources - allowed:
                    raise ValueError(f"{field} cites unavailable event IDs {sorted(sources - allowed)}")
                if field != "uncertainties" and sources & gap_events:
                    raise ValueError(f"{field} cites an uncertain action outcome")

    def source_events(self) -> set[int]:
        return {
            event_id for field in type(self).model_fields for entry in getattr(self, field)
            for event_id in entry.source_event_ids
        }


@dataclass(frozen=True)
class MemoryRevision:
    id: int
    through_event_id: int
    content: EpisodeMemory
    gap_event_ids: frozenset[int]

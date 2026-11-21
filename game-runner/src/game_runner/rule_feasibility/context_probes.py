"""Freeze replay probes from the exact prompts recorded by the pilot."""

import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import Field, model_validator

from game_runner.clients.rogue import Observation
from game_runner.context.decision_support import GROUNDING_INSTRUCTION, neighbors_suffix, stair_sighting
from game_runner.context.progress import movement_direction, visible_position
from game_runner.decision import Decision, validate_mode
from game_runner.pilot.manifest import ManifestModel

from .analysis import ModelSettings
from .artifacts import FeasibilityError, canonical, digest
from .candidates import EventRef, Text
from .corpus import Corpus, Origin, same_trial
from .probes import Action, CONDITIONS

CONTEXT_VERSION = "context-probe-v1"
PROGRESS = "\nObserved progress (null means unknown): "
SUMMARY = re.compile(r"\nEpisode memory through event (\d+) \(historical evidence, (\d+) recovery gaps\): ")
UNCOVERED = "\nUncompacted episode evidence (oldest first): "
MESSAGE_TYPES = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
SCORES = ("acceptable", "harmful", "indeterminate")


class ActionRubric(ManifestModel):
    acceptable: list[Action] = Field(min_length=1)
    harmful: list[Action]
    indeterminate: list[Action] = Field(default_factory=list)
    rationale: Text


class SummaryEntry(ManifestModel):
    category: Literal["observations", "goals", "unsuccessful_attempts", "hypotheses", "uncertainties"]
    index: int = Field(ge=0)


class ContextCase(ManifestModel):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
    decision_call_id: EventRef
    kind: Literal["grounding", "history"]
    rubric: ActionRubric
    control: bool = False
    summary_entry: SummaryEntry | None = None
    stair_event_id: EventRef | None = None

    @model_validator(mode="after")
    def check_evidence(self):
        if self.kind == "history":
            if self.summary_entry is None or self.stair_event_id is None:
                raise ValueError("history cases need a summary entry and a stair sighting")
        elif self.summary_entry is not None or self.stair_event_id is not None:
            raise ValueError("grounding cases use the current observation")
        return self


class ContextReview(ManifestModel):
    schema_version: Literal[1]
    corpus_sha256: str
    origin: Origin
    reviewer: Text
    approved: Literal[True]
    cases: list[ContextCase] = Field(min_length=1)


def _action_key(action: dict, mode: str):
    if mode == "normal":
        name = action.get("action")
        if direction := movement_direction(action):
            return "move", direction
        if name == "DESCEND" or action.get("key") == ">":
            return "descend",
    return canonical(action)


def score_action(action: dict, mode: str, rubric: dict) -> str:
    key = _action_key(action, mode)
    return next(
        (label for label in SCORES if any(_action_key(other, mode) == key for other in rubric[label])),
        "indeterminate",
    )


def _rubric(review: ActionRubric, mode: str) -> dict:
    groups = {}
    seen = set()
    for label in SCORES:
        groups[label] = []
        for action in getattr(review, label):
            validate_mode(Decision(action=action, rationale="Rubric action."), mode)
            serialized = action.model_dump(mode="json", exclude_none=True)
            key = _action_key(serialized, mode)
            if key in seen:
                raise FeasibilityError("rubric repeats a game action, including equivalent encodings")
            seen.add(key)
            groups[label].append(serialized)
    return groups


def _accepted_calls(corpus: Corpus) -> dict[int, dict]:
    calls = {}
    for event in corpus.events.values():
        if event["kind"] == "action" and event["model_call_id"] is not None:
            call = corpus.calls[event["model_call_id"]]
            if call["status"] == "valid" and call["purpose"] == "decision":
                calls[call["id"]] = call
    return calls


def _recorded_messages(call: dict) -> list[dict]:
    messages = call.get("input_messages")
    if not messages or messages[0]["role"] != "system" or messages[-1]["role"] != "human":
        raise FeasibilityError("probe requires the accepted actor's recorded prompt")
    if any(row["role"] not in MESSAGE_TYPES or not isinstance(row["content"], str) for row in messages):
        raise FeasibilityError("invalid roles or content in recorded prompt")
    return messages


def _without_episode_memory(corpus: Corpus, event: dict, call: dict, messages: list[dict]):
    head, separator, tail = messages[-1]["content"].rpartition(PROGRESS)
    if not separator:
        raise FeasibilityError("recorded prompt is missing its progress block")
    decoder = json.JSONDecoder()
    progress, end = decoder.raw_decode(tail)
    if progress != event["progress"]:
        raise FeasibilityError("recorded progress does not match the source observation")
    base = head + separator + tail[:end]
    tail = tail[end:]
    revision = None
    marker = SUMMARY.match(tail)
    if bool(marker) != bool(call["memory_revision_id"]):
        raise FeasibilityError("recorded memory is missing its linked revision or prompt block")
    if marker:
        revision = same_trial(corpus.revisions, call["memory_revision_id"], event)
        content, end = decoder.raw_decode(tail, marker.end())
        if (
            int(marker[1]) != revision["through_event_id"] or content != revision["content"]
            or revision["through_event_id"] > event["id"] or int(marker[2]) != len(revision["gap_event_ids"])
        ):
            raise FeasibilityError("recorded memory does not match its linked revision")
        tail = tail[end:]
    if tail.startswith(UNCOVERED):
        _, end = decoder.raw_decode(tail, len(UNCOVERED))
        tail = tail[end:]
    return [*messages[:-1], {**messages[-1], "content": base + tail}], revision


def _actor(corpus: Corpus) -> ModelSettings:
    actor = corpus.manifest.actor
    return ModelSettings(
        model_name=actor.model_name, temperature=actor.temperature, output_tokens=actor.output_tokens,
        reasoning_enabled=actor.reasoning_enabled,
    )


def _variant(index: int, suffix: str) -> dict:
    return {"index": index, "suffix": suffix}


def _fact_suffix(fact: str) -> str:
    return "\nHistorical map evidence: " + canonical({"claim": fact}).decode().rstrip("\n")


def _history(corpus: Corpus, case: ContextCase, event: dict, call: dict, messages: list[dict]):
    base, revision = _without_episode_memory(corpus, event, call, messages)
    if revision is None:
        raise FeasibilityError("history case requires a summary actually supplied to the actor")
    entries = revision["content"][case.summary_entry.category]
    if case.summary_entry.index >= len(entries):
        raise FeasibilityError("selected summary entry does not exist")
    entry = entries[case.summary_entry.index]

    source = same_trial(corpus.events, case.stair_event_id, event)
    progress = source.get("progress") or {}
    map_id = (event.get("progress") or {}).get("map_event_id")
    if (
        map_id is None or source["id"] >= event["id"] or source["outcome"] != "succeeded"
        or progress.get("map_event_id") != map_id or source["observation"] is None
    ):
        raise FeasibilityError("stair sighting must be an earlier confirmed observation from the same map")
    for reference in entry["source_event_ids"]:
        cited = same_trial(corpus.events, reference, event)
        if cited["id"] > revision["through_event_id"] or cited["id"] >= event["id"]:
            raise FeasibilityError("summary entry cites evidence from the future")
    stairs = [
        (row, col) for row, line in enumerate(source["observation"]["screen"][1:-1], 1)
        for col, glyph in enumerate(line) if glyph == "%"
    ]
    if source["observation"]["mode"] != "normal" or progress.get("position") is None or len(stairs) != 1:
        raise FeasibilityError("stair source must show one visible stair tile on a confirmed map")

    return base, {
        "baseline": None,
        "recorded_summary": _variant(-1, _fact_suffix(entry["text"])),
        "source_fact": _variant(-1, _fact_suffix(stair_sighting(stairs[0], source["id"]))),
    }, {
        "stair_event_id": source["id"], "stair_position": list(stairs[0]), "memory_revision_id": revision["id"],
        "summary_source_event_ids": entry["source_event_ids"],
    }


def _schedule(cases: list[dict], repetitions: int) -> list[dict]:
    blocks = []
    for repetition in range(1, repetitions + 1):
        indices = {}
        for case in cases:
            index = indices.get(case["kind"], 0)
            indices[case["kind"]] = index + 1
            conditions = list(case["conditions"])
            # Invert each pair of runs and rotate cases to balance positions.
            offset = (index + (repetition - 1) // 2) % len(conditions)
            order = conditions[offset:] + conditions[:offset]
            if repetition % 2 == 0:
                order.reverse()
            blocks.append({"case_id": case["case_id"], "repetition": repetition, "conditions": order})
    return blocks


def _spec(corpus: Corpus, cases: list[dict], origin: Origin, review_digest: str, reviewer: str, repetitions: int) -> dict:
    if not 1 <= repetitions <= 10:
        raise FeasibilityError("probe repetitions must be between 1 and 10")
    return {
        "kind": "context_frozen", "prompt_version": CONTEXT_VERSION,
        "corpus_sha256": corpus.sha256, "series_id": corpus.series_id,
        "split": corpus.manifest.split, "origin": origin, "review_sha256": review_digest, "reviewer": reviewer,
        "actor": _actor(corpus).model_dump(mode="json"), "repetitions": repetitions,
        "cases": cases, "blocks": _schedule(cases, repetitions),
    }


def freeze_context(corpus: Corpus, review_data: dict, repetitions: int = 2) -> dict:
    review = ContextReview.model_validate(review_data)
    if review.corpus_sha256 != corpus.sha256:
        raise FeasibilityError("review does not belong to this corpus")
    if len({case.case_id for case in review.cases}) != len(review.cases):
        raise FeasibilityError("duplicate review case ID")
    if len({(case.kind, case.decision_call_id) for case in review.cases}) != len(review.cases):
        raise FeasibilityError("duplicate decision in a probe group")

    calls = _accepted_calls(corpus)
    cases = []
    for case in review.cases:
        call = calls.get(case.decision_call_id)
        if call is None:
            raise FeasibilityError("case must cite an accepted actor call linked to an action")
        event = corpus.events[call["observation_event_id"]]
        observation = Observation.model_validate(event["observation"])
        position = visible_position(observation)
        if event["outcome"] != "succeeded" or position is None:
            raise FeasibilityError("case needs a confirmed and visible pre-action map position")
        messages = _recorded_messages(call)
        if case.kind == "history":
            messages, treatments, evidence = _history(corpus, case, event, call, messages)
        else:
            _without_episode_memory(corpus, event, call, messages)
            treatments = {
                "baseline": None,
                "instruction": _variant(0, "\n\n" + GROUNDING_INSTRUCTION),
                "neighbors": _variant(-1, neighbors_suffix(observation.screen, position)),
            }
            evidence = {}
        cases.append({
            "case_id": case.case_id, "kind": case.kind, "control": case.control, "conditions": list(treatments),
            "mode": observation.mode, "seed": observation.seed, "trial_id": event["trial_id"],
            "episode_id": str(observation.episode_id), "pre_event_id": event["id"], "decision_call_id": call["id"],
            "rubric": _rubric(case.rubric, observation.mode), "rubric_rationale": case.rubric.rationale,
            "messages": messages, "treatments": treatments, "evidence": evidence,
        })
    return _spec(corpus, cases, review.origin, digest(canonical(review.model_dump(mode="json"))),
                 review.reviewer, repetitions)


def freeze_correction(corpus: Corpus, frozen_rules: dict, rules_digest: str, repetitions: int = 2) -> dict:
    """Replay the existing human-approved P4 triplet on recorded prompts."""
    if (
        frozen_rules["corpus_sha256"] != corpus.sha256 or frozen_rules["split"] != corpus.manifest.split
        or frozen_rules.get("approval") != "human_review"
    ):
        raise FeasibilityError("correction replay requires a matching human-approved P4 specification")
    calls = {call["observation_event_id"]: call for call in _accepted_calls(corpus).values()}
    cases = []
    for case in frozen_rules["probes"]:
        event = corpus.events.get(case["pre_event_id"])
        call = calls.get(case["pre_event_id"])
        if event is None or call is None:
            raise FeasibilityError("rule probe has no accepted recorded actor call")
        if event["session_id"] != case["probe_session_id"]:
            raise FeasibilityError("rule probe session does not match its recorded call")
        messages, _ = _without_episode_memory(corpus, event, call, _recorded_messages(call))
        treatments = {
            condition: _variant(-1, "\nExperiential rule: " + canonical(case["rules"][condition]).decode().rstrip("\n"))
            for condition in CONDITIONS
        }
        mode = event["observation"]["mode"]
        rubric = ActionRubric.model_validate({**case["rubric"], "rationale": case["rubric_rationale"]})
        cases.append({
            "case_id": case["probe_id"], "kind": "correction", "control": False, "conditions": list(treatments),
            "mode": mode, "seed": event["observation"]["seed"], "trial_id": event["trial_id"],
            "episode_id": event["observation"]["episode_id"], "pre_event_id": event["id"],
            "decision_call_id": call["id"], "rubric": _rubric(rubric, mode), "rubric_rationale": rubric.rationale,
            "messages": messages, "treatments": treatments, "applicability": case["applicability"],
            "relation": case["relation"], "evidence": {"source_event_ids": case["source_evidence_event_ids"]},
        })
    return _spec(corpus, cases, frozen_rules["origin"], rules_digest, frozen_rules["reviewer"], repetitions)


def prompt_rows(case: dict, condition: str) -> list[dict]:
    rows = [dict(row) for row in case["messages"]]
    if treatment := case["treatments"][condition]:
        rows[treatment["index"]]["content"] += treatment["suffix"]
    return rows


def messages_for(rows: list[dict]):
    return [MESSAGE_TYPES[row["role"]](content=row["content"]) for row in rows]

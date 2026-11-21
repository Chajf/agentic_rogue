"""Measure observed progress from a verified pilot export."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median

from game_runner.context.decision_support import MOVE_OFFSETS, NEIGHBORS_PREFIX
from game_runner.context.progress import COMBAT, Position, movement_direction
from game_runner.rule_feasibility.artifacts import FeasibilityError
from game_runner.rule_feasibility.corpus import Corpus

CHECKPOINTS = (25, 50, 100, 200)
WALLS = {"-", "|"}
TERRAIN_WITHOUT_STAIRS = WALLS | {".", "#", "+"}


@dataclass(frozen=True)
class MapView:
    map_id: int
    position: Position
    screen: list[str]

    def glyph(self, position: Position) -> str | None:
        row, col = position
        if 0 < row < len(self.screen) - 1 and 0 <= col < len(self.screen[row]):
            return self.screen[row][col]
        return None

    def target(self, direction: str) -> str | None:
        row, col = self.position
        dy, dx = MOVE_OFFSETS[direction]
        return self.glyph((row + dy, col + dx))

    def location(self) -> tuple[int, int, int]:
        return (self.map_id, *self.position)


def _map_view(event: dict | None) -> MapView | None:
    if event is None or event["outcome"] != "succeeded" or not (observation := event["observation"]):
        return None
    progress = event.get("progress") or {}
    position = progress.get("position")
    map_id = progress.get("map_event_id")
    if observation["mode"] != "normal" or observation["terminated"] or position is None or map_id is None:
        return None
    row, col = position
    return MapView(map_id, (row, col), observation["screen"])


@dataclass
class SeenStairs:
    """Track stairs only within the last confirmed map instance."""

    map_id: int | None = None
    sources: dict[Position, int] = field(default_factory=dict)

    def observe(self, event: dict, view: MapView | None) -> None:
        if event["outcome"] != "succeeded":
            self.map_id = None
            self.sources.clear()
            return
        if view is None:
            return
        if view.map_id != self.map_id:
            self.map_id = view.map_id
            self.sources.clear()
        # Visible static terrain supersedes a prior sighting. An overlay such
        # as the player glyph or an unobserved blank leaves it intact.
        self.sources = {
            position: source for position, source in self.sources.items()
            if view.glyph(position) not in TERRAIN_WITHOUT_STAIRS
        }
        for row, line in enumerate(view.screen[1:-1], 1):
            for col, glyph in enumerate(line):
                if glyph == "%":
                    self.sources[(row, col)] = event["id"]

    def nearby(self, view: MapView | None) -> tuple[int | None, list[int]]:
        if view is None or self.map_id != view.map_id:
            return None, []
        if (source := self.sources.get(view.position)) is not None:
            return source, []
        row, col = view.position
        adjacent = {
            source for (y, x), source in self.sources.items()
            if max(abs(y - row), abs(x - col)) == 1
        }
        return None, sorted(adjacent)


def _decision_row(
    event: dict, before: dict | None, call: dict | None, previous: dict | None,
    input_map: MapView | None, output_map: MapView | None, stairs: SeenStairs,
) -> dict:
    action = event["action_request"]
    if call is not None and call["parsed_action"] != action:
        raise FeasibilityError(f"action {event['id']} differs from its accepted decision")
    input_is_map = before is not None and before["observation"]["mode"] == "normal"
    direction = movement_direction(action) if action and input_is_map else None
    target = input_map.target(direction) if direction and input_map else None
    comparable = (
        direction is not None and input_map is not None and output_map is not None
        and input_map.map_id == output_map.map_id and event["api_step"] == before["api_step"] + 1
    )
    displaced = input_map.position != output_map.position if comparable else None
    output_messages = event["observation"]["messages"] if event["observation"] else []
    combat = bool(
        comparable and target and "A" <= target <= "Z"
        and any(COMBAT.search(message) for message in output_messages)
    )
    quiet_wall = (
        displaced is False and target in WALLS
        and (event.get("progress") or {}).get("no_movement_streak", 0) > 0
    )
    follows_previous = previous is not None and before is not None and previous["output_event_id"] == before["id"]
    repeat_source = None
    if follows_previous and input_map and previous["quiet_wall_failure"]:
        if input_map.target(previous["movement_direction"]) in WALLS:
            repeat_source = previous["output_event_id"]
    returned = bool(
        follows_previous and displaced and previous["displaced"]
        and previous["input_location"] == output_map.location()
    )
    stair_source, adjacent_sources = stairs.nearby(input_map)
    descend = bool(action and (action.get("action") == "DESCEND" or (input_is_map and action.get("key") == ">")))
    return {
        "trial_id": event["trial_id"],
        "call_id": call["id"] if call else None,
        "input_event_id": before["id"] if before else None,
        "output_event_id": event["id"],
        "api_step": event["api_step"],
        "outcome": event["outcome"],
        "action": action,
        "rationale": call["decision_rationale"] if call else None,
        "input_location": input_map.location() if input_map else None,
        "output_location": output_map.location() if output_map else None,
        "movement_direction": direction,
        "target_glyph": target,
        "displaced": displaced,
        "combat_feedback": combat,
        "quiet_wall_failure": quiet_wall,
        "repeat_wall_opportunity_event_id": repeat_source,
        "repeated_wall_move": repeat_source is not None and direction == previous["movement_direction"],
        "returned_to_previous_position": returned,
        "on_known_stairs_source_event_id": stair_source,
        "adjacent_stair_source_event_ids": adjacent_sources,
        "descent_attempt": descend,
    }


def _trial_rows(trial: dict, events: list[dict], corpus: Corpus) -> tuple[list[dict], dict | None]:
    rows = []
    stairs = SeenStairs()
    reset = None
    last_event = None
    for event in events:
        view = _map_view(event)
        if event["kind"] == "reset":
            reset = event
        elif event["kind"] == "action":
            call = corpus.calls.get(event["model_call_id"])
            before = corpus.events[call["observation_event_id"]] if call else None
            input_map = _map_view(before)
            if before is None or last_event is None or before["id"] != last_event["id"]:
                stairs = SeenStairs()
                if before is not None:
                    stairs.observe(before, input_map)
            previous = rows[-1] if rows else None
            row = _decision_row(event, before, call, previous, input_map, view, stairs)
            rows.append(row)
        stairs.observe(event, view)
        if event["kind"] == "action":
            row["output_on_known_stairs_source_event_id"], _ = stairs.nearby(view)
        last_event = event
    if len(rows) != trial.get("action_dispatches", len(rows)):
        raise FeasibilityError(f"dispatch count disagrees with recorded actions for trial {trial['trial_id']}")
    return rows, reset


def _coverage(rows: list[dict], reset: dict | None, last_step: int | None) -> dict:
    visited = set()
    first_visits = []
    reset_view = _map_view(reset)
    if reset_view:
        visited.add(reset_view.location())
        first_visits.append(reset["api_step"])

    last_new_dispatch = 0 if reset_view else None
    run_start = None
    run_length = 0
    longest = {"actions": 0, "start_event_id": None, "end_event_id": None}
    previous = None
    for dispatch, row in enumerate(rows, 1):
        location = row["output_location"]
        row["new_position"] = location is not None and location not in visited
        if row["new_position"]:
            visited.add(location)
            first_visits.append(row["api_step"])
            last_new_dispatch = dispatch
        if row["displaced"] is None or row["new_position"]:
            run_length = 0
        else:
            if previous and previous["output_event_id"] != row["input_event_id"]:
                run_length = 0
            if not run_length:
                run_start = row["output_event_id"]
            run_length += 1
            if run_length > longest["actions"]:
                longest = {
                    "actions": run_length, "start_event_id": run_start, "end_event_id": row["output_event_id"],
                }
        previous = row

    checkpoints = []
    for limit in CHECKPOINTS:
        observed_through = min(last_step, limit) if last_step is not None else None
        checkpoints.append({
            "api_step": limit,
            "observed_through_api_step": observed_through,
            "positions": sum(step <= limit for step in first_visits) if observed_through is not None else None,
        })
    return {
        "checkpoints": checkpoints,
        "positions": len(visited),
        "confirmed_position_outputs": sum(row["output_location"] is not None for row in rows),
        "last_new_position_api_step": first_visits[-1] if first_visits else None,
        "actions_since_last_new_position": len(rows) - last_new_dispatch if last_new_dispatch is not None else None,
        "longest_no_new_position_move_run": longest,
        "immediate_returns": sum(row["returned_to_previous_position"] for row in rows),
    }


def _trial_metrics(trial: dict, events: list[dict], rows: list[dict], reset: dict | None, calls: list[dict]) -> dict:
    last_step = max((e["api_step"] for e in events if e["observation"] is not None), default=None)
    coverage = _coverage(rows, reset, last_step)
    usage = trial.get("model_usage", {})
    metrics = {
        **{key: trial.get(key) for key in (
            "trial_id", "ordinal", "requested_seed", "observed_reset_seed", "context_policy", "session_id",
            "episode_id", "outcome", "reason", "game_status", "max_observed_depth", "level_changes",
        )},
        **{key: trial[key] for key in ("context_variant", "prompt_version") if key in trial},
        "action_dispatches": len(rows),
        "last_observed_api_step": last_step,
        "dispatch_outcomes": dict(Counter(row["outcome"] for row in rows)),
        "coverage": coverage,
        "movement": {
            "comparable_moves": sum(row["displaced"] is not None for row in rows),
            "moves_with_displacement": sum(row["displaced"] is True for row in rows),
            "moves_without_displacement": sum(row["displaced"] is False for row in rows),
            "same_position_combat": sum(row["displaced"] is False and row["combat_feedback"] for row in rows),
            "repeat_wall_opportunities": sum(row["repeat_wall_opportunity_event_id"] is not None for row in rows),
            "repeated_wall_moves": sum(row["repeated_wall_move"] for row in rows),
        },
        "stairs": {
            "outputs_on_known_stairs": sum(row["output_on_known_stairs_source_event_id"] is not None for row in rows),
            "on_stairs_opportunities": sum(row["on_known_stairs_source_event_id"] is not None for row in rows),
            "adjacent_stairs_opportunities": sum(bool(row["adjacent_stair_source_event_ids"]) for row in rows),
            "descent_attempts": sum(row["descent_attempt"] for row in rows),
            "descents_attempted_on_stairs": sum(
                row["descent_attempt"] and row["on_known_stairs_source_event_id"] is not None for row in rows
            ),
            "descents_attempted_adjacent": sum(
                row["descent_attempt"] and bool(row["adjacent_stair_source_event_ids"]) for row in rows
            ),
        },
        "model_usage": {
            purpose: {
                **usage.get(purpose, {}),
                "status_counts": dict(Counter(call["status"] for call in calls if call["purpose"] == purpose)),
            }
            for purpose in ("decision", "episode_summary")
        },
    }
    if "context_variant" in trial:
        decision_prompts = [
            call["input_messages"][-1]["content"] for call in calls
            if call["purpose"] == "decision" and call.get("input_messages")
        ]
        metrics["neighbors_prompt_exposure"] = {
            "recorded_decision_prompts": len(decision_prompts),
            "with_neighbors": sum(NEIGHBORS_PREFIX in prompt for prompt in decision_prompts),
        }
    return metrics



def _matched_pairs(trials: list[dict]) -> list[dict]:
    by_seed = defaultdict(list)
    for trial in trials:
        by_seed[trial["requested_seed"]].append(trial)
    pairs = []
    for seed, group in by_seed.items():
        by_policy = {trial["context_policy"]: trial for trial in group}
        if len(group) != 2 or set(by_policy) != {"history_only", "working_memory"}:
            continue
        baseline, memory = by_policy["history_only"], by_policy["working_memory"]
        comparable = all(trial["last_observed_api_step"] is not None for trial in group)
        pairs.append({
            "seed": seed,
            "history_only_trial_id": baseline["trial_id"],
            "working_memory_trial_id": memory["trial_id"],
            "positions_delta": (
                memory["coverage"]["positions"] - baseline["coverage"]["positions"] if comparable else None
            ),
            "longest_no_new_position_move_run_delta": (
                memory["coverage"]["longest_no_new_position_move_run"]["actions"]
                - baseline["coverage"]["longest_no_new_position_move_run"]["actions"]
                if comparable else None
            ),
        })
    return pairs


def _finished_for_comparison(trial: dict, horizon: int) -> bool:
    step = trial["last_observed_api_step"]
    if step is None or trial["coverage"]["positions"] == 0:
        return False
    if any(outcome != "succeeded" for outcome in trial["dispatch_outcomes"]):
        return False
    if trial["outcome"] == "ended":
        return trial["game_status"] in {"dead", "won", "quit"}
    return trial["outcome"] == "limited" and trial["reason"] == "trial_action_limit" and step == horizon


def _variant_delta(reference: dict, other: dict, horizon: int) -> dict:
    complete = _finished_for_comparison(reference, horizon) and _finished_for_comparison(other, horizon)
    row = {
        "seed": reference["requested_seed"], "baseline_trial_id": reference["trial_id"],
        "comparison_trial_id": other["trial_id"], "complete": complete,
        "positions_delta": None, "longest_no_new_position_move_run_delta": None,
        "baseline_died": reference["game_status"] == "dead", "comparison_died": other["game_status"] == "dead",
    }
    if complete:
        first, second = reference["coverage"], other["coverage"]
        row.update(
            positions_delta=second["positions"] - first["positions"],
            longest_no_new_position_move_run_delta=(
                second["longest_no_new_position_move_run"]["actions"]
                - first["longest_no_new_position_move_run"]["actions"]
            ),
        )
    return row


def _neighbors_comparison(trials: list[dict], horizon: int) -> dict:
    by_seed = defaultdict(list)
    for trial in trials:
        by_seed[trial["requested_seed"]].append(trial)
    primary, repeats = [], []
    for group in by_seed.values():
        baseline, *replicates = [trial for trial in group if trial.get("context_variant", "baseline") == "baseline"]
        neighbors, = [trial for trial in group if trial.get("context_variant") == "neighbors"]
        primary.append(_variant_delta(baseline, neighbors, horizon))
        repeats.extend(_variant_delta(baseline, trial, horizon) for trial in replicates)
    complete = [pair for pair in primary if pair["complete"]]
    return {
        "action_horizon": horizon, "primary_pairs": primary, "baseline_repeats": repeats,
        "complete_primary_pairs": len(complete),
        "pairs_with_improved_coverage": sum(pair["positions_delta"] > 0 for pair in complete),
        "median_primary_streak_delta": (
            median(pair["longest_no_new_position_move_run_delta"] for pair in complete) if complete else None
        ),
        "baseline_deaths": sum(pair["baseline_died"] for pair in primary),
        "neighbors_deaths": sum(pair["comparison_died"] for pair in primary),
    }


def analyze(corpus: Corpus) -> tuple[dict, list[dict]]:
    """Return trial metrics and one evidence-linked row per recorded dispatch."""
    events_by_trial = defaultdict(list)
    calls_by_trial = defaultdict(list)
    for event in sorted(corpus.events.values(), key=lambda event: event["id"]):
        events_by_trial[event["trial_id"]].append(event)
    for call in corpus.calls.values():
        calls_by_trial[call["trial_id"]].append(call)

    trials = []
    ledger = []
    for trial_id, trial in corpus.trials.items():
        events = events_by_trial[trial_id]
        rows, reset = _trial_rows(trial, events, corpus)
        trials.append(_trial_metrics(trial, events, rows, reset, calls_by_trial[trial_id]))
        ledger.extend(rows)
    metrics = {
        "schema_version": 1,
        "series_id": corpus.series_id,
        "split": corpus.manifest.split,
        "bundle_sha256": corpus.sha256,
        "trial_counts": dict(Counter(trial["outcome"] for trial in trials)),
        "action_dispatches": len(ledger),
        "trials": trials,
        "matched_seed_pairs": _matched_pairs(trials),
    }
    if any(trial.get("context_variant") == "neighbors" for trial in trials):
        metrics["neighbors_comparison"] = _neighbors_comparison(trials, corpus.manifest.trial_limits.actions)
    return metrics, ledger

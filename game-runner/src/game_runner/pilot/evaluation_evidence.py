"""Link measured trials to their setup and verify the v2 evidence."""

from collections import defaultdict

from game_runner.clients.rogue import Observation
from game_runner.context.progress import visible_position
from game_runner.context.transitions import DecisionTransition
from game_runner.context.working_state import UnlinkedObservation, replay_working_state
from game_runner.pilot.diagnostics import action_payload, fingerprints
from game_runner.rule_feasibility.artifacts import FeasibilityError


def trial_evidence(trial: dict, planned: dict, events: list[dict]) -> dict:
    scenario = planned.get("diagnostic")
    if scenario is None:
        anchor = next((
            event for event in events if event["kind"] == "reset" and event["outcome"] == "succeeded"
            and event["observation"] and event["api_step"] == 0
            and event["observation"]["seed"] == trial["requested_seed"]
        ), None)
        verification = None
    else:
        recorded = next((event for event in events if (event.get("progress") or {}).get("diagnostic_anchor")), None)
        verification = (recorded["progress"]["diagnostic_anchor"] if recorded else None)
        anchor = recorded if verification and verification.get("verified") is True else None
    return {
        "condition": planned.get("condition"),
        "scenario_id": scenario["scenario_id"] if scenario else None,
        "setup_dispatches": sum(event.get("scripted_input_event_id") is not None for event in events),
        "anchor_event_id": anchor["id"] if anchor else None,
        "anchor_api_step": anchor["api_step"] if anchor else None,
        "anchor_verification": verification,
    }


def measured_events(events: list[dict], anchor_id: int | None) -> list[dict]:
    if anchor_id is None:
        return []
    return [
        event for event in events
        if event["kind"] == "action" and event.get("scripted_input_event_id") is None and event["id"] > anchor_id
    ]


def validate_evaluation(corpus, metadata: dict) -> None:
    events_by_trial = defaultdict(list)
    calls_by_trial = defaultdict(list)
    for event in sorted(corpus.events.values(), key=lambda row: row["id"]):
        events_by_trial[event["trial_id"]].append(event)
    for call in corpus.calls.values():
        calls_by_trial[call["trial_id"]].append(call)
    binary = (metadata.get("provenance") or {}).get("rogue_binary_sha256")
    for plan in corpus.manifest.trials:
        trial = corpus.trials[plan.trial_id]
        events, calls = events_by_trial[plan.trial_id], calls_by_trial[plan.trial_id]
        _validate_trial(trial, plan, events, calls, binary)


def _validate_trial(trial, plan, events, calls, binary):
    by_id = {event["id"]: event for event in events}
    call_by_id = {call["id"]: call for call in calls}
    evidence = trial_evidence(trial, plan.model_dump(mode="json"), events)
    if any(trial.get(key) != value for key, value in evidence.items()):
        raise FeasibilityError("evaluation result disagrees with its condition or recorded start")

    resets = [event for event in events if event["kind"] == "reset"]
    if len(resets) > 1 or (events and (not resets or resets[0] != events[0])):
        raise FeasibilityError("evaluation trial requires a single initial reset")
    reset = resets[0] if resets else None
    previous = reset if reset and reset["observation"] and reset["outcome"] == "succeeded" else None
    transitions = []
    setup = [event for event in events if event.get("scripted_input_event_id") is not None]
    spec = plan.diagnostic
    if (spec is None and setup) or (spec and len(setup) > len(spec.prefix)):
        raise FeasibilityError("scripted dispatches disagree with the diagnostic prefix")
    frames = [Observation.model_validate(previous["observation"])] if previous else []
    actions = []
    for index, event in enumerate(setup):
        if (
            previous is None or event["scripted_input_event_id"] != previous["id"]
            or event["model_call_id"] is not None or event["kind"] != "action"
            or event["action_request"] != action_payload(spec.prefix[index])
        ):
            raise FeasibilityError("diagnostic setup is outside its linked observation boundary")
        if event["outcome"] == "succeeded" and event["observation"]:
            transition = _observed_transition(previous, event)
            transitions.append(transition)
            frames.append(transition.output_observation)
            actions.append(transition.action)
            previous = event
        else:
            previous = None

    anchors = [(event, event["progress"]["diagnostic_anchor"]) for event in events
               if (event.get("progress") or {}).get("diagnostic_anchor") is not None]
    if anchors:
        if len(anchors) != 1 or spec is None or anchors[0][0] != previous:
            raise FeasibilityError("diagnostic verification is outside its setup boundary")
        trace_hash, anchor_hash = fingerprints(frames, actions)
        verified = (
            len(actions) == len(spec.prefix) and binary == spec.binary_sha256
            and trace_hash == spec.trace_sha256 and anchor_hash == spec.anchor_sha256
            and visible_position(frames[-1]) is not None
        )
        if anchors[0][1] != {
            "scenario_id": spec.scenario_id, "verified": verified,
            "trace_sha256": trace_hash, "anchor_sha256": anchor_hash,
        }:
            raise FeasibilityError("diagnostic fingerprints do not match the recorded evidence")

    anchor_id = evidence["anchor_event_id"]
    measured = measured_events(events, anchor_id)
    if len(setup) + len(measured) != sum(event["kind"] == "action" for event in events):
        raise FeasibilityError("measured action precedes its verified start")
    if trial["action_dispatches"] != len(measured):
        raise FeasibilityError("evaluation dispatch count disagrees with the recorded measured actions")
    for call in calls:
        event = by_id.get(call["observation_event_id"])
        if anchor_id is None or event is None or event["id"] < anchor_id or event["observation"] is None:
            raise FeasibilityError("model call precedes the verified start or lacks its input observation")

    previous = by_id.get(anchor_id)
    for event in measured:
        call = call_by_id.get(event["model_call_id"])
        if (
            previous is None or call is None or call["observation_event_id"] != previous["id"]
            or call["parsed_action"] != event["action_request"]
        ):
            raise FeasibilityError("measured action differs from its linked actor decision")
        if event["outcome"] in {"succeeded", "uncertain"} and event["observation"]:
            transitions.append(_observed_transition(previous, event, call))
            previous = event
        else:
            previous = None

    if plan.structured_state == "map-v1" and frames and reset["observation"]["seed"] == plan.seed:
        try:
            _, reconstructed = replay_working_state(reset["id"], frames[0], transitions)
        except UnlinkedObservation as exc:
            raise FeasibilityError("working-state evidence has an unlinked transition") from exc
        saved = [(by_id[delta["event_id"]].get("progress") or {}).get("working_state_delta") for delta in reconstructed]
        if saved != reconstructed:
            raise FeasibilityError("working-state deltas disagree with the observed trace")


def _observed_transition(before, after, call=None) -> DecisionTransition:
    first = Observation.model_validate(before["observation"])
    last = Observation.model_validate(after["observation"])
    if last.api_step != first.api_step + 1:
        raise FeasibilityError("evaluation observations must advance one API step")
    return DecisionTransition(
        before["id"], first, before["outcome"], call["id"] if call else None,
        after["action_request"], call["decision_rationale"] if call else "",
        after["id"], last, after["outcome"], after.get("progress"), before.get("progress"),
    )

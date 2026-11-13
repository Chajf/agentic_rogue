from dataclasses import replace

from test_agent_graph import observed

from game_runner.context.progress import Progress, visible_position

from uuid import uuid4


def test_only_visible_map_position_is_used():
    base = observed(uuid4(), screen=["", "|@.|", "-+-", "Level: 1"])
    assert visible_position(base) == (1, 1)
    assert visible_position(base.model_copy(update={"mode": "confirm"})) is None
    assert visible_position(base.model_copy(update={"screen": ["", "a) Food", "Level: 1"]})) is None
    assert visible_position(base.model_copy(update={"screen": ["", "|@@|", "Level: 1"]})) is None


def test_streak_counts_movement_but_breaks_on_wait_combat_and_recovery():
    base = observed(uuid4(), screen=["", "|@.|", "---", "Level: 1"]).model_copy(update={"messages": []})
    progress = Progress.initial(1, base)
    movement = {"type": "semantic", "action": "MOVE_UP"}
    progress = progress.advance(2, base, movement, base, recovered=False)
    assert progress.no_movement_streak == 1
    progress = progress.advance(3, base, {"type": "key", "key": "k"}, base, recovered=False)
    assert progress.no_movement_streak == 2
    for action, after, recovered in [
        ({"type": "semantic", "action": "REST"}, base, False),
        (movement, base.model_copy(update={"messages": ["The bat misses."]}), False),
        (movement, base.model_copy(update={"mode": "confirm"}), False),
        (movement, base, True),
    ]:
        updated = progress.advance(4, base, action, after, recovered=recovered)
        assert updated.no_movement_streak == 0
        if recovered:
            assert updated.position_change is None
            assert updated.map_event_id == 4
            assert updated.messages is None


def test_inventory_slot_provenance_and_freshness_and_new_map():
    base = observed(uuid4(), screen=["", "|@.|", "---", "Level: 1"])
    listing = base.model_copy(update={"messages": ["   a) Some food"]})
    progress = Progress.initial(1, base).advance(
        2, base, {"type": "semantic", "action": "INVENTORY"}, listing, recovered=False,
    )
    assert progress.inventory.slots == (("a", "Some food"),)
    assert progress.inventory.event_id == 2
    assert not progress.inventory.stale
    assert progress.position is None
    stepped = base.model_copy(update={"state": base.state.model_copy(update={"dungeon_level": 2, "hp": 9})})
    advanced = progress.advance(3, listing, {"type": "semantic", "action": "DESCEND"}, stepped, recovered=False)
    assert advanced.inventory == replace(progress.inventory, stale=True)
    assert advanced.map_event_id == 3
    assert advanced.position_change is None
    assert advanced.level == (1, 2)
    assert advanced.hp == (12, 9)
    assert advanced.hunger is None

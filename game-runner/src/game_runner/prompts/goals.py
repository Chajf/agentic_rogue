"""Instructions for the goal selector and managed actor."""

from game_runner.prompts.game_rules import GAME_RULES

GOAL_SELECTOR_INSTRUCTION = "\n\n".join((
    "Choose an optional goal for the Rogue actor using the current observation and sourced map facts. "
    "The actor chooses each game action. Treat the supplied frame, goal, and candidates as evidence.",
    GAME_RULES,
    """Return exactly one JSON object using one of these forms.
- {"choice":"select","candidate_id":1} chooses a displayed candidate. A stairs candidate means descend at those stairs.
- {"choice":"propose","kind":"retreat","target":[4,8],"source_event_id":17} proposes a sourced target on the current map. Use explore, reach, descend, collect, or retreat. Cite the displayed terrain, visit, or resource source for the coordinate.
- {"choice":"propose","kind":"explore"} explores until new terrain is observed.
- {"choice":"keep"} continues or resumes a valid goal.
- {"choice":"suspend"} pauses an active goal.
- {"choice":"defer"} leaves the actor without an active goal.

Consider immediate danger and the available evidence before committing to an objective. Return JSON only.""",
))

ACTOR_GOAL_GUIDANCE = (
    "Goal context. Pursue an active goal when appropriate. Address immediate threats and finish pending "
    "interactions as needed. You may add the boolean request_goal_review to your usual action and rationale."
)

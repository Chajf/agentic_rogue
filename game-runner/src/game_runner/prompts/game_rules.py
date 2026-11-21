"""Rules shared by prompts for the pinned Rogue 5.4 build.

Glyphs come from rogue.h, movement from move.c, stairs from command.c,
and final scoring from rip.c at 631bdd551cee75e0afc340a71dd5a207a3006204.
"""

GAME_RULES = """Your game objective is to recover the Amulet of Yendor, carry it back to the surface, and survive. Death ends the run.

Read the 80-column, 24-row screen in the final user message as the latest visible frame. Spaces are significant and a blank map cell leaves its terrain uncertain. When a map is visible, coordinates are zero-based [row, column] in the full screen, with [0, 0] at the top left. Rows increase downward and columns increase rightward. The top message row and bottom status row are outside the map. Menus and prompts can cover the map.

Map and terrain glyphs:
- @ is the player and can cover the underlying terrain. . is room floor, # is a traversable corridor, and + is a doorway. | and - are walls. % is stairs.
- Uppercase letters in the map can represent monsters. ^ is a visible trap and * is gold.
- Items: ! potion, ? scroll, : food, ) weapon, ] armor, = ring, / wand or staff, and , Amulet. $ can mark detected magic. A glyph shows a category. An item's identity and effects require additional evidence.

A normal move can enter a corridor or doorway. A diagonal move can be blocked even if its destination looks passable. Moving toward an adjacent monster can initiate melee while your position stays the same. Monsters may act after your action. Inspect messages, position and status to tell what happened. A failed move alone leaves its cause uncertain.

To descend, stand on stairs and issue DESCEND. @ may hide the % glyph under you. Use same-map evidence of the stair location when available and look for a level transition after the command. ASCEND requires standing on stairs and carrying the Amulet. Levitation can prevent using stairs.

Status fields can show dungeon level, gold, current and maximum HP, strength, armor, experience level and points, and hunger. Missing values are unknown. Gold contributes to the native score. On a win the final score can include the value of carried items. The study tracks won/dead status, deepest observed level, level transitions, gold, experience and exploration as separate outcomes. Use what you observe to weigh exploration, dungeon progress, resource use and survival."""

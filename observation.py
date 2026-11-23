"""Encode protocol v3 duel states for the first 1v1 PPO policy."""
import math

import numpy as np

from duel import Controls, WEAPONS

DIRECTIONS = ("Left", "Right", "Up", "Down")
EVENTS = ("HitDealt", "HitReceived", "BlockMade", "AttackBlocked",
          "KickAttempted", "KickLanded", "KickReceived",
          "BashAttempted", "BashLanded", "BashReceived")
# Two 24-value fighter slots, visibility, relative position, event counts, and damage.
OBSERVATION_SIZE = 2 * 24 + 1 + 3 + len(EVENTS) + 2


def decode_action(action, fighter):
    forward, right, yaw, pitch, weapon, jump, crouch = map(int, action)
    x, y, z = fighter["lookDirection"]
    heading = math.degrees(math.atan2(y, x))
    elevation = math.degrees(math.asin(max(-1, min(1, z))))
    axis = (0, -1, 1)
    move = WEAPONS[weapon]
    return Controls(
        right=axis[right], forward=axis[forward],
        yaw=(heading + 7.5 * axis[yaw] + 180) % 360 - 180 if yaw else None,
        pitch=max(-60, min(60, elevation + 5 * axis[pitch])) if pitch else None,
        weapon=move if move not in ("Kick", "Bash") else "None",
        jump=bool(jump), kick=move == "Kick", bash=move == "Bash", crouch=bool(crouch),
    )


def fighter_features(fighter):
    actions = (fighter["lowerBodyAction"], fighter["upperBodyAction"])
    attack = fighter["attackDirection"].removeprefix("Attack")
    guard = fighter["guard"].removeprefix("Defend")
    return [
        *fighter["lookDirection"],
        *(component / 8 for component in fighter["velocity"]),
        fighter["attackStage"] in ("AttackReady", "AttackQuickReady"),
        fighter["attackStage"] == "AttackRelease",
        *(attack == direction for direction in DIRECTIONS),
        *(guard == direction for direction in DIRECTIONS),
        "Kick" in actions, "WeaponBash" in actions,
        bool(fighter["mainHand"]), bool(fighter["offHand"]),
        fighter["isMounted"], fighter["isCrouching"],
        fighter["eyeHeight"] / 3, fighter["scale"] / 2,
    ]


def combat_damage(events):
    dealt = received = 0.0
    for event in events:
        damage = max(0.0, event["damage"] or 0.0)
        if event["kind"] in ("HitDealt", "KickLanded", "BashLanded"):
            dealt += damage
        elif event["kind"] in ("HitReceived", "KickReceived", "BashReceived"):
            received += damage
    return dealt, received


def encode_observation(state, side):
    current = state.observations[side]
    opponent_id = state.observations[1 - side]["self"]["index"]
    opponent = next((agent for agent in current["visibleAgents"]
                     if agent["index"] == opponent_id), None)
    events = current["combatFeedback"]
    dealt, received = combat_damage(events)
    values = [
        *fighter_features(current["self"]),
        *(fighter_features(opponent) if opponent else [0] * 24),
        opponent is not None,
        *(offset / 20 for offset in (opponent["relativePosition"] if opponent else (0, 0, 0))),
        *(sum(event["kind"] == kind for event in events) / 4 for kind in EVENTS),
        dealt / 100, received / 100,
    ]
    return np.asarray(values, dtype=np.float32).clip(-1, 1)

"""Client and smoke test for the Bannerlord duel protocol."""
import argparse
from dataclasses import dataclass
from enum import IntEnum
import json
import math
import random
import socket
from typing import NamedTuple


class Action(IntEnum):
    NEUTRAL = 0
    ATTACK_LEFT = 1
    GUARD_LEFT = 2
    GUARD_RIGHT = 3
    ATTACK_RIGHT = 4
    ATTACK_UP = 5
    ATTACK_DOWN = 6
    GUARD_UP = 7
    GUARD_DOWN = 8
    KICK = 9


class Controls(NamedTuple):
    forward: int = 0
    strafe: int = 0
    yaw: int = 0
    pitch: int = 0
    weapon: int = Action.NEUTRAL


ACTION_SIZES = (3, 3, 3, 3, len(Action))
PROTOCOL_VERSION = 2
OBSERVATION_SIZE = 31
SIGNED_OBSERVATIONS = (3, 4, 8, 9, 10)
MAX_LINE_BYTES = 16_384


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class DuelState:
    episode: int
    step: int
    observations: tuple[tuple[float, ...], tuple[float, ...]]
    hits: tuple[int, int]
    blocks: tuple[int, int]
    outcome: str


class DuelClient:
    def __init__(self, port: int, timeout: float = 5.0):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout)
        self.reader = self.socket.makefile("rb")
        self.state: DuelState | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        try:
            self.reader.close()
        finally:
            self.socket.close()

    def reset(self, seed: int | None = None) -> DuelState:
        previous = self.state
        self.state = None
        state = self._exchange("reset", seed=seed)
        if state.step != 0 or state.outcome != "running":
            raise ProtocolError("Reset did not return a fresh duel.")
        if previous is not None and state.episode <= previous.episode:
            raise ProtocolError("Reset did not advance the episode ID.")
        self.state = state
        return state

    def step(self, action_a: Controls, action_b: Controls) -> DuelState:
        previous = self.state
        if previous is None or previous.outcome != "running":
            raise ProtocolError("Reset the duel before submitting an action.")
        actions = [list(map(int, action_a)), list(map(int, action_b))]
        if any(len(action) != len(ACTION_SIZES) or
               any(not 0 <= value < size for value, size in zip(action, ACTION_SIZES))
               for action in actions):
            raise ValueError("Invalid duel controls.")
        self.state = None
        state = self._exchange(
            "step", episode=previous.episode, step=previous.step, actions=actions
        )
        if state.episode != previous.episode or state.step != previous.step + 1:
            raise ProtocolError("The server returned a stale or out-of-order state.")
        self.state = state
        return state

    def _exchange(self, command: str, **fields) -> DuelState:
        request = json.dumps({"version": PROTOCOL_VERSION, "command": command, **fields})
        self.socket.sendall(request.encode("utf-8") + b"\n")
        line = self.reader.readline(MAX_LINE_BYTES + 1)
        if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
            raise ProtocolError("The server closed the connection or sent an invalid frame.")
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or data.get("version") != PROTOCOL_VERSION:
                raise ValueError(f"Expected duel protocol v{PROTOCOL_VERSION}. Update the Python repo and server together.")
            if "error" in data:
                raise ProtocolError(f"Server rejected {command}: {data['error']}")
            episode, step = data["episode"], data["step"]
            if any(type(value) is not int or value < 0 for value in (episode, step)):
                raise ValueError("Episode and step must be nonnegative integers.")
            observations = data["observations"]
            if len(observations) != 2 or any(len(row) != OBSERVATION_SIZE for row in observations):
                raise ValueError("Unexpected observation shape.")
            for row in observations:
                if any(type(value) not in (int, float) or not math.isfinite(value)
                       or not (-1 if index in SIGNED_OBSERVATIONS else 0) <= value <= 1
                       for index, value in enumerate(row)):
                    raise ValueError("Observation outside its normalized range.")
            hits, blocks = data["hits"], data["blocks"]
            for counts in (hits, blocks):
                if len(counts) != 2 or any(type(count) is not int or count < 0 for count in counts):
                    raise ValueError("Invalid event counts.")
            outcome = data["outcome"]
            if outcome not in ("running", "a_win", "b_win", "draw", "timeout"):
                raise ValueError("Unknown duel outcome.")
            return DuelState(
                episode, step, tuple(map(tuple, observations)), tuple(hits), tuple(blocks), outcome
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError(f"Invalid {command} response: {exc}") from exc


def check_swing(client: DuelClient, guard: Action) -> tuple[int, int, float]:
    state = client.reset()
    health = state.observations[1][0]
    hits = blocks = 0
    # At 50 ms per step, settle for 1.5 s, set the guard for 0.25 s,
    # hold the attack for 0.9 s, then release and observe contact.
    for duration, attacker, defender in (
        (30, Action.NEUTRAL, Action.NEUTRAL),
        (5, Action.NEUTRAL, guard),
        (18, Action.ATTACK_LEFT, guard),
        (25, Action.NEUTRAL, guard),
    ):
        for _ in range(duration):
            state = client.step(Controls(weapon=attacker), Controls(weapon=defender))
            hits += state.hits[0]
            blocks += state.blocks[0]
            if state.outcome != "running":
                return hits, blocks, health - state.observations[1][0]
    return hits, blocks, health - state.observations[1][0]


def check_controls(client: DuelClient) -> None:
    neutral = Controls()
    initial = client.reset()
    if initial.observations[0][5] != 1:
        raise ProtocolError("Fighter A cannot see B at the spawn point. Use clear, flat ground.")

    state = initial
    for _ in range(18):
        state = client.step(Controls(yaw=1), neutral)
    hidden = state.observations[0]
    live_enemy = (1, 9, 10, 12, 14, *range(19, 23), *range(27, 31))
    if hidden[5] != 0 or hidden[6] != 1 or hidden[7] <= 0 or any(hidden[i] for i in live_enemy):
        raise ProtocolError("Turning away did not hide live opponent data and preserve last sighting.")
    for _ in range(18):
        state = client.step(Controls(yaw=2), neutral)
    if state.observations[0][5] != 1:
        raise ProtocolError("Turning back did not restore visibility.")
    print("Yaw and visibility: PASS")

    initial = client.reset()
    for _ in range(9):
        state = client.step(Controls(pitch=2), neutral)
    raised = state.observations[0]
    if raised[8] - initial.observations[0][8] < 0.15:
        raise ProtocolError("Pitch did not raise the fighter's view.")
    if raised[5] != 0 or raised[6] != 1 or any(raised[i] for i in live_enemy):
        raise ProtocolError("Looking up did not hide live opponent data.")
    for _ in range(9):
        state = client.step(Controls(pitch=1), neutral)
    if state.observations[0][5] != 1:
        raise ProtocolError("Looking level did not restore visibility.")
    print("Pitch and visibility: PASS")

    initial = client.reset()
    for _ in range(9):
        state = client.step(Controls(forward=1), neutral)
    if state.observations[0][2] <= initial.observations[0][2] + 0.025:
        raise ProtocolError("Walking backward did not increase separation.")
    backward_distance = state.observations[0][2]
    for _ in range(9):
        state = client.step(Controls(forward=2), neutral)
    if state.observations[0][2] >= backward_distance - 0.025:
        raise ProtocolError("Walking forward did not close distance.")
    for _ in range(7):
        state = client.step(Controls(strafe=2), neutral)
    if abs(state.observations[0][3]) < 0.10:
        raise ProtocolError("Strafing did not change the opponent's bearing.")
    print("Footwork: PASS")

    client.reset()
    for _ in range(10):
        client.step(neutral, neutral)
    state = client.step(Controls(weapon=Action.KICK), neutral)
    kicked = state.observations[0][13] == 1
    for _ in range(12):
        state = client.step(neutral, neutral)
        kicked |= state.observations[0][13] == 1
    if not kicked:
        raise ProtocolError("The kick input did not trigger a visible kick animation.")
    print("Kick: PASS")

    guards = (Action.GUARD_LEFT, Action.GUARD_RIGHT, Action.GUARD_UP, Action.GUARD_DOWN)
    for direction, attack in enumerate(
        (Action.ATTACK_LEFT, Action.ATTACK_RIGHT, Action.ATTACK_UP, Action.ATTACK_DOWN)
    ):
        client.reset()
        attack_seen = guard_seen = False
        for _ in range(30):
            client.step(neutral, neutral)
        for _ in range(18):
            state = client.step(Controls(weapon=attack), Controls(weapon=guards[direction]))
            attack_seen |= state.observations[0][15 + direction] == 1
            guard_seen |= state.observations[1][23 + direction] == 1
        if not (attack_seen and guard_seen):
            raise ProtocolError(f"{attack.name} or {guards[direction].name} did not enter its expected animation state.")
        print(f"{attack.name} and {guards[direction].name}: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Bannerlord duel bridge.")
    parser.add_argument("--port", type=int, default=49317)
    parser.add_argument("--random-bouts", type=int, default=0)
    parser.add_argument("--full-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.random_bouts < 0:
        parser.error("--random-bouts must be nonnegative.")
    try:
        with DuelClient(args.port) as client:
            results = {}
            for guard in (Action.NEUTRAL, Action.GUARD_LEFT, Action.GUARD_RIGHT):
                hits, blocks, damage = check_swing(client, guard)
                results[guard] = hits, blocks, damage
                print(f"{guard.name}: hits={hits} blocks={blocks} health_lost={damage:.3f}")
            hits, _, damage = results[Action.NEUTRAL]
            guarded = any(blocks > 0 and abs(loss) < 1e-6
                          for guard, (_, blocks, loss) in results.items() if guard != Action.NEUTRAL)
            if hits == 0 or damage <= 0 or not guarded:
                raise ProtocolError("Expected an unguarded hit and a directional block.")
            if args.full_controls:
                check_controls(client)
            rng = random.Random(args.seed)
            for index in range(args.random_bouts):
                state = client.reset(seed=rng.randrange(2**31))
                while state.outcome == "running":
                    if state.step >= 400:
                        raise ProtocolError("The server did not end the bout within 400 steps.")
                    a, b = (Controls(*(rng.randrange(size) for size in ACTION_SIZES)) for _ in range(2))
                    state = client.step(a, b)
                print(f"Random bout {index + 1}: {state.outcome} after {state.step} steps")
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Duel check failed on localhost:{args.port}: {exc}\n")
    print("PASS")


if __name__ == "__main__":
    main()
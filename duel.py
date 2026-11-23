"""Client and smoke test for the Bannerlord duel protocol."""
import argparse
from dataclasses import dataclass
from enum import IntEnum
import json
import math
import random
import socket


class Action(IntEnum):
    NEUTRAL = 0
    ATTACK_LEFT = 1
    GUARD_LEFT = 2
    GUARD_RIGHT = 3


OBSERVATION_SIZE = 9
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

    def step(self, action_a: Action, action_b: Action) -> DuelState:
        previous = self.state
        if previous is None or previous.outcome != "running":
            raise ProtocolError("Reset the duel before submitting an action.")
        actions = [int(Action(action_a)), int(Action(action_b))]
        self.state = None
        state = self._exchange(
            "step", episode=previous.episode, step=previous.step, actions=actions
        )
        if state.episode != previous.episode or state.step != previous.step + 1:
            raise ProtocolError("The server returned a stale or out-of-order state.")
        self.state = state
        return state

    def _exchange(self, command: str, **fields) -> DuelState:
        request = json.dumps({"version": 1, "command": command, **fields})
        self.socket.sendall(request.encode("utf-8") + b"\n")
        line = self.reader.readline(MAX_LINE_BYTES + 1)
        if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
            raise ProtocolError("The server closed the connection or sent an invalid frame.")
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("Unsupported protocol version.")
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
                       or abs(value) > 1 for value in row) or min((*row[:3], *row[5:])) < 0:
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
            state = client.step(attacker, defender)
            hits += state.hits[0]
            blocks += state.blocks[0]
            if state.outcome != "running":
                return hits, blocks, health - state.observations[1][0]
    return hits, blocks, health - state.observations[1][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Bannerlord duel bridge.")
    parser.add_argument("--port", type=int, default=49317)
    parser.add_argument("--random-bouts", type=int, default=0)
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
            rng = random.Random(args.seed)
            for index in range(args.random_bouts):
                state = client.reset(seed=rng.randrange(2**31))
                while state.outcome == "running":
                    if state.step >= 400:
                        raise ProtocolError("The server did not end the bout within 400 steps.")
                    state = client.step(rng.choice(tuple(Action)), rng.choice(tuple(Action)))
                print(f"Random bout {index + 1}: {state.outcome} after {state.step} steps")
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Duel check failed on localhost:{args.port}: {exc}\n")
    print("PASS")


if __name__ == "__main__":
    main()
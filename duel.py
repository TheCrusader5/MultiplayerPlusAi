"""Client and smoke test for the Bannerlord protocol v3 duel bridge."""
import argparse
from dataclasses import asdict, dataclass
import json
import socket

PROTOCOL_VERSION = 3
WEAPONS = ("None", "AttackLeft", "AttackRight", "AttackUp", "AttackDown",
           "BlockLeft", "BlockRight", "BlockUp", "BlockDown", "Kick", "Bash")
ACTION_SIZES = (3, 3, 3, 3, len(WEAPONS), 2, 2)


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Controls:
    right: float = 0.0
    forward: float = 0.0
    yaw: float | None = None
    pitch: float | None = None
    weapon: str = "None"
    jump: bool = False
    kick: bool = False
    bash: bool = False
    crouch: bool = False


@dataclass(frozen=True)
class DuelState:
    episode: int
    step: int
    outcome: str
    observations: tuple[dict, dict]
    native_side: int | None = None
    opponent_class_id: str | None = None


class DuelClient:
    def __init__(self, port: int, timeout: float = 10):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = self.socket.makefile("rb")
        self.state = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.reader.close()
        finally:
            self.socket.close()

    def reset(self, native_side: int | None = None) -> DuelState:
        state = self._request("reset", nativeSide=native_side)
        if state.native_side != native_side:
            raise ProtocolError("The server did not start the requested opponent mode.")
        if state.step != 0 or state.outcome != "running" or (
            self.state is not None and state.episode <= self.state.episode
        ):
            raise ProtocolError("Reset did not start a fresh bout.")
        self.state = state
        return state

    def step(self, first: Controls, second: Controls) -> DuelState:
        before = self.state
        if before is None or before.outcome != "running":
            raise ProtocolError("Reset the duel before stepping.")
        self.state = None
        state = self._request("step", episode=before.episode, step=before.step,
                              actions=[asdict(first), asdict(second)])
        if state.episode != before.episode or state.step != before.step + 1:
            raise ProtocolError("The server returned an out-of-order step.")
        self.state = state
        return state

    def _request(self, command, **payload):
        message = {"version": PROTOCOL_VERSION, "command": command, **payload}
        self.socket.sendall(json.dumps(message, allow_nan=False).encode() + b"\n")
        line = self.reader.readline(1_048_577)
        if not line.endswith(b"\n") or len(line) > 1_048_576:
            raise ProtocolError("The server closed the bridge or sent an invalid frame.")
        try:
            reply = json.loads(line)
            if "error" in reply:
                raise ProtocolError(reply["error"])
            if reply["version"] != PROTOCOL_VERSION:
                raise ProtocolError("Expected protocol v3. Check the deployed server DLL.")
            a, b = reply["observations"]
            outcome = reply["outcome"]
            if outcome not in ("running", "a_win", "b_win", "draw", "timeout"):
                raise ValueError("unknown bout outcome")
            return DuelState(reply["episode"], reply["step"], outcome, (a, b),
                             reply.get("nativeSide"), reply.get("opponentClassId"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"Invalid {command} response: {exc}") from exc


def check_terminal_hit(client: DuelClient) -> None:
    state = client.reset()
    for _ in range(8):
        for length, weapon in ((12, "None"), (18, "AttackLeft"), (25, "None")):
            for _ in range(length):
                state = client.step(Controls(weapon=weapon), Controls())
                if state.outcome == "running":
                    continue
                a, b = state.observations
                def has_hit(fighter, opponent, kind):
                    return any(event["kind"] == kind and (event["damage"] or 0) > 0
                               and event["otherAgentIndex"] == opponent["self"]["index"]
                               for event in fighter["combatFeedback"])
                if state.outcome != "a_win" or not (
                    has_hit(a, b, "HitDealt") and has_hit(b, a, "HitReceived")
                ):
                    raise ProtocolError(f"Missing terminal feedback or unexpected outcome: {state.outcome}.")
                print(f"PASS: fighter A won at step {state.step}; both observations contain the killing blow.")
                return
    raise ProtocolError("No knockout after eight scripted swings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=49317)
    args = parser.parse_args()
    try:
        with DuelClient(args.port) as client:
            check_terminal_hit(client)
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Duel check failed: {exc}\n")


if __name__ == "__main__":
    main()

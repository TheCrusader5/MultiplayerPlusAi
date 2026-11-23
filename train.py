"""Train a policy against frozen copies of itself in the Bannerlord duel."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from duel import ACTION_SIZES, OBSERVATION_SIZE, SIGNED_OBSERVATIONS, Controls, DuelClient, ProtocolError


class BannerlordDuel(gym.Env):
    def __init__(self, client: DuelClient):
        super().__init__()
        self.client = client
        self.action_space = gym.spaces.MultiDiscrete(ACTION_SIZES)
        low = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        low[list(SIGNED_OBSERVATIONS)] = -1
        self.observation_space = gym.spaces.Box(low=low, high=np.ones(OBSERVATION_SIZE, np.float32))
        self.opponents = deque(maxlen=8)
        self.opponent = None
        self.learner_side = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.learner_side = 1 - self.learner_side
        self.opponent = self.opponents[int(self.np_random.integers(len(self.opponents)))]
        state = self.client.reset(seed=int(self.np_random.integers(2**31)))
        return self._observation(state, self.learner_side), {}

    def step(self, action):
        before = self.client.state
        other_side = 1 - self.learner_side
        other_action, _ = self.opponent.predict(self._observation(before, other_side))
        opponent_controls = Controls(*map(int, other_action))
        actions = [opponent_controls, opponent_controls]
        actions[self.learner_side] = Controls(*map(int, action))
        after = self.client.step(*actions)
        damage_taken = before.observations[self.learner_side][0] - after.observations[self.learner_side][0]
        damage_dealt = before.observations[other_side][0] - after.observations[other_side][0]
        reward = 0.2 * (damage_dealt - damage_taken)
        if after.outcome in ("a_win", "b_win"):
            winner = 0 if after.outcome == "a_win" else 1
            reward += 1 if winner == self.learner_side else -1
        terminated = after.outcome in ("a_win", "b_win", "draw")
        truncated = after.outcome == "timeout"
        info = {"outcome": after.outcome, "learner_side": self.learner_side}
        return self._observation(after, self.learner_side), reward, terminated, truncated, info

    @staticmethod
    def _observation(state, side: int) -> np.ndarray:
        return np.asarray(state.observations[side], dtype=np.float32)


class ParallelDuelEnv(DummyVecEnv):
    """Send each duel's inputs concurrently while keeping SB3's batch order."""

    def __init__(self, duels, executor):
        super().__init__([lambda duel=duel: duel for duel in duels])
        self.executor = executor

    def reset(self):
        results = self.executor.map(self._reset_duel, self.envs, self._seeds, self._options)
        observations, infos = zip(*results)
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def step_wait(self):
        results = self.executor.map(self._step_duel, self.envs, self.actions)
        observations, rewards, dones, infos, resets = zip(*results)
        for index, reset_info in enumerate(resets):
            if reset_info is not None:
                self.reset_infos[index] = reset_info
        return np.stack(observations), np.asarray(rewards, dtype=np.float32), np.asarray(dones), list(infos)

    @staticmethod
    def _reset_duel(duel, seed, options):
        return duel.reset(seed=seed, options=options)

    @staticmethod
    def _step_duel(duel, action):
        observation, reward, terminated, truncated, info = duel.step(action)
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        reset_info = None
        if done:
            info["terminal_observation"] = observation
            observation, reset_info = duel.reset()
        return observation, reward, done, info, reset_info


def snapshot_policy(policy):
    # Rebuild from weights so the opponent carries no live tensors from PPO's last update.
    with BytesIO() as checkpoint:
        policy.save(checkpoint)
        checkpoint.seek(0)
        opponent = type(policy).load(checkpoint, device=policy.device)
    opponent.requires_grad_(False)
    opponent.set_training_mode(False)
    return opponent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--port", type=int, default=49317, help="First training port (default: 49317).")
    connection.add_argument("--ports", type=int, nargs="+", help="Explicit ports of parallel duel bridges.")
    parser.add_argument("--arenas", type=int, help="Number of consecutive arena ports starting at --port.")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=512, help="Decisions per duel per update.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/duel"))
    args = parser.parse_args()
    if args.updates < 1 or args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error("--updates must be positive and --rollout-steps must be a positive multiple of 64.")
    if args.arenas is not None:
        if args.ports is not None or not 1 <= args.arenas <= 64:
            parser.error("--arenas expects a count from 1 to 64 and cannot be combined with --ports.")
        if not 1 <= args.port <= 65536 - args.arenas:
            parser.error("--port and --arenas specify a range outside ports 1 to 65535.")
        ports = list(range(args.port, args.port + args.arenas))
    else:
        ports = args.ports or [args.port]
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        parser.error("Use distinct training ports from 1 to 65535.")
    rollout_size = args.rollout_steps * len(ports)
    try:
        with ExitStack() as resources:
            duels = []
            for port in ports:
                try:
                    client = resources.enter_context(DuelClient(port))
                except OSError as exc:
                    raise OSError(f"Could not connect to training bridge on localhost:{port}: {exc}") from exc
                duels.append(BannerlordDuel(client))

            env = duels[0]
            if len(duels) > 1:
                executor = resources.enter_context(ThreadPoolExecutor(max_workers=len(duels)))
                env = VecMonitor(ParallelDuelEnv(duels, executor))

            args.output.mkdir(parents=True, exist_ok=False)
            print(f"Training on {len(duels)} duel(s) with {rollout_size} decisions per update.", flush=True)
            model = PPO("MlpPolicy", env, n_steps=args.rollout_steps, batch_size=64,
                        seed=args.seed, device="cpu", verbose=1)
            for update in range(1, args.updates + 1):
                # Each duel gets its own copy because policy prediction mutates its distribution.
                for duel in duels:
                    duel.opponents.append(snapshot_policy(model.policy))
                model.learn(total_timesteps=rollout_size, reset_num_timesteps=False)
                model.save(args.output / f"checkpoint-{update:04d}")
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
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

from duel import ACTION_SIZES, Controls, DuelClient, ProtocolError
from observation import OBSERVATION_SIZE, combat_damage, decode_action, encode_observation


class BannerlordDuel(gym.Env):
    def __init__(self, client: DuelClient, native_fraction: float = 0.0):
        super().__init__()
        self.client = client
        self.native_fraction = native_fraction
        self.native_opponent = False
        self.action_space = gym.spaces.MultiDiscrete(ACTION_SIZES)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        self.opponents = deque(maxlen=8)
        self.opponent = None
        self.learner_side = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.learner_side = 1 - self.learner_side
        self.native_opponent = self.np_random.random() < self.native_fraction
        self.opponent = None if self.native_opponent else self.opponents[int(self.np_random.integers(len(self.opponents)))]
        self.bout_had_damage = False
        state = self.client.reset(native_side=1 - self.learner_side if self.native_opponent else None)
        return self._observation(state, self.learner_side), {}

    def step(self, action):
        before = self.client.state
        other_side = 1 - self.learner_side
        opponent_controls = Controls()
        if self.opponent is not None:
            other_action, _ = self.opponent.predict(self._observation(before, other_side))
            opponent_controls = decode_action(other_action, before.observations[other_side]["self"])
        actions = [opponent_controls, opponent_controls]
        actions[self.learner_side] = decode_action(action, before.observations[self.learner_side]["self"])
        after = self.client.step(*actions)
        damage_dealt, damage_taken = combat_damage(after.observations[self.learner_side]["combatFeedback"])
        self.bout_had_damage |= damage_dealt > 0 or damage_taken > 0
        reward = (0.4 * damage_dealt - 0.2 * damage_taken) / 100
        if after.outcome in ("a_win", "b_win"):
            winner = 0 if after.outcome == "a_win" else 1
            reward += 1 if winner == self.learner_side else -1
        terminated = after.outcome in ("a_win", "b_win", "draw")
        truncated = after.outcome == "timeout"
        if truncated and not self.bout_had_damage:
            reward -= 0.1
        info = {"outcome": after.outcome, "learner_side": self.learner_side, "native_opponent": self.native_opponent}
        return self._observation(after, self.learner_side), reward, terminated, truncated, info

    @staticmethod
    def _observation(state, side: int) -> np.ndarray:
        return encode_observation(state, side)


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
    parser.add_argument("--port", type=int, default=49317)
    parser.add_argument("--arenas", type=int, default=1)
    parser.add_argument("--native-fraction", type=float, default=0.0,
                        help="Probability of a Bannerlord AI opponent for each new bout (0 to 1).")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/duel"))
    args = parser.parse_args()
    if args.updates < 1 or args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error("--updates must be positive and --rollout-steps must be a positive multiple of 64.")
    if not 1 <= args.arenas <= 16 or not 1 <= args.port <= 65536 - args.arenas:
        parser.error("Choose 1 to 16 arenas and a valid consecutive port range.")
    if not 0 <= args.native_fraction <= 1:
        parser.error("--native-fraction must be between 0 and 1.")
    try:
        with ExitStack() as resources:
            duels = []
            for port in range(args.port, args.port + args.arenas):
                try:
                    client = resources.enter_context(DuelClient(port))
                except OSError as exc:
                    raise OSError(f"Could not connect to the duel bridge on port {port}: {exc}") from exc
                duels.append(BannerlordDuel(client, args.native_fraction))
            env = duels[0]
            if len(duels) > 1:
                executor = resources.enter_context(ThreadPoolExecutor(max_workers=len(duels)))
                env = VecMonitor(ParallelDuelEnv(duels, executor))

            args.output.mkdir(parents=True, exist_ok=False)
            model = PPO("MlpPolicy", env, n_steps=args.rollout_steps, batch_size=64,
                        seed=args.seed, device="cpu", verbose=1)
            decisions_per_update = args.rollout_steps * len(duels)
            for update in range(1, args.updates + 1):
                if args.native_fraction < 1:
                    # Keep a separate frozen copy per arena for concurrent predictions.
                    for duel in duels:
                        duel.opponents.append(snapshot_policy(model.policy))
                model.learn(total_timesteps=decisions_per_update, reset_num_timesteps=False)
                model.save(args.output / f"checkpoint-{update:04d}")
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
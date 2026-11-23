"""Train a policy against frozen copies of itself in the Bannerlord duel."""
import argparse
from collections import deque
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from duel import ACTION_SIZES, DuelClient, ProtocolError
from observation import OBSERVATION_SIZE, combat_damage, decode_action, encode_observation


class BannerlordDuel(gym.Env):
    def __init__(self, client: DuelClient):
        super().__init__()
        self.client = client
        self.action_space = gym.spaces.MultiDiscrete(ACTION_SIZES)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        self.opponents = deque(maxlen=8)
        self.opponent = None
        self.learner_side = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.learner_side = 1 - self.learner_side
        self.opponent = self.opponents[int(self.np_random.integers(len(self.opponents)))]
        self.bout_had_damage = False
        state = self.client.reset()
        return self._observation(state, self.learner_side), {}

    def step(self, action):
        before = self.client.state
        other_side = 1 - self.learner_side
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
        info = {"outcome": after.outcome, "learner_side": self.learner_side}
        return self._observation(after, self.learner_side), reward, terminated, truncated, info

    @staticmethod
    def _observation(state, side: int) -> np.ndarray:
        return encode_observation(state, side)


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
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/duel"))
    args = parser.parse_args()
    if args.updates < 1 or args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error("--updates must be positive and --rollout-steps must be a positive multiple of 64.")
    try:
        with DuelClient(args.port) as client:
            args.output.mkdir(parents=True, exist_ok=False)
            env = BannerlordDuel(client)
            model = PPO("MlpPolicy", env, n_steps=args.rollout_steps, batch_size=64,
                        seed=args.seed, device="cpu", verbose=1)
            env.opponents.append(snapshot_policy(model.policy))
            for update in range(1, args.updates + 1):
                model.learn(total_timesteps=args.rollout_steps, reset_num_timesteps=False)
                model.save(args.output / f"checkpoint-{update:04d}")
                env.opponents.append(snapshot_policy(model.policy))
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
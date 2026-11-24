"""Train a policy against native Bannerlord duel AI."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from duel import ACTION_SIZES, Controls, DuelClient, ProtocolError
from observation import OBSERVATION_SIZE, combat_damage, decode_action, encode_observation


DAMAGE_POINTS_PER_HP = 1.0
SURVIVAL_POINTS_PER_STEP = 0.001


class BannerlordDuel(gym.Env):
    def __init__(self, client: DuelClient, port: int):
        super().__init__()
        self.client = client
        self.port = port
        self.action_space = gym.spaces.MultiDiscrete(ACTION_SIZES)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        self.learner_side = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.learner_side = 1 - self.learner_side
        self.damage_points = 0.0
        self.survival_points = 0.0
        state = self.client.reset(native_side=1 - self.learner_side)
        if not state.opponent_class_id:
            raise ProtocolError("Deploy the server build with randomized duel bot characters.")
        return self._observation(state, self.learner_side), {}

    def step(self, action):
        before = self.client.state
        other_side = 1 - self.learner_side
        actions = [Controls(), Controls()]
        actions[self.learner_side] = decode_action(action, before.observations[self.learner_side]["self"])
        after = self.client.step(*actions)
        opponent_index = after.observations[other_side]["self"]["index"]
        events = after.observations[self.learner_side]["combatFeedback"]
        damage_dealt, _ = combat_damage(event for event in events if event["otherAgentIndex"] == opponent_index)
        damage_points = DAMAGE_POINTS_PER_HP * damage_dealt
        learner_lost = after.outcome in ("draw", "b_win" if self.learner_side == 0 else "a_win")
        survival_points = 0.0 if learner_lost else SURVIVAL_POINTS_PER_STEP
        self.damage_points += damage_points
        self.survival_points += survival_points
        reward = damage_points + survival_points

        terminated = after.outcome in ("a_win", "b_win", "draw")
        truncated = after.outcome == "timeout"
        info = {"outcome": after.outcome, "learner_side": self.learner_side, "native_opponent": True,
                "opponent_class_id": before.opponent_class_id}
        if terminated or truncated:
            total = self.damage_points + self.survival_points
            print(f"[Duel {self.port}, round {after.episode}] {before.opponent_class_id} "
                  f"{after.outcome} | {self.damage_points:.3f} damage + {self.survival_points:.3f} survival "
                  f"= {total:.3f} points", flush=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=49317)
    parser.add_argument("--arenas", type=int, default=1)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, help="Steps per arena per update (default 512 for a new run).")
    parser.add_argument("--resume", type=Path, help="Continue from a saved PPO checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/duel"))
    args = parser.parse_args()
    if args.updates < 1 or (args.rollout_steps is not None and
                            (args.rollout_steps < 64 or args.rollout_steps % 64)):
        parser.error("--updates must be positive and --rollout-steps must be a positive multiple of 64.")
    if not 1 <= args.arenas <= 16 or not 1 <= args.port <= 65536 - args.arenas:
        parser.error("Choose 1 to 16 arenas and a valid consecutive port range.")
    try:
        with ExitStack() as resources:
            duels = []
            for port in range(args.port, args.port + args.arenas):
                try:
                    client = resources.enter_context(DuelClient(port))
                except OSError as exc:
                    raise OSError(f"Could not connect to the duel bridge on port {port}: {exc}") from exc
                duels.append(BannerlordDuel(client, port))
            env = duels[0]
            if len(duels) > 1:
                executor = resources.enter_context(ThreadPoolExecutor(max_workers=len(duels)))
                env = VecMonitor(ParallelDuelEnv(duels, executor))

            if args.resume is None:
                model = PPO("MlpPolicy", env, n_steps=args.rollout_steps or 512, batch_size=64,
                            seed=args.seed, device="cpu", verbose=1)
            else:
                overrides = {} if args.rollout_steps is None else {"n_steps": args.rollout_steps}
                model = PPO.load(args.resume, env=env, device="cpu", **overrides)
                print(f"Resumed {args.resume} at {model.num_timesteps:,} timesteps.", flush=True)
            args.output.mkdir(parents=True, exist_ok=False)
            decisions_per_update = model.n_steps * len(duels)
            for update in range(1, args.updates + 1):
                model.learn(total_timesteps=decisions_per_update, reset_num_timesteps=False)
                model.save(args.output / f"checkpoint-{update:04d}")
    except (OSError, ProtocolError) as exc:
        parser.exit(1, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
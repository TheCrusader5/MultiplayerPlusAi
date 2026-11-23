# MultiplayerPlusAi

Python tools for training Bannerlord duel bots against frozen copies of their own policy. MultiplayerPlusEngine runs the duel and applies both agents' inputs.

## Run locally

Deploy the matching server build. Join a private Duel server as an admin, spawn on foot, face clear ground, and enter `$combattest train`. Run the Python commands on the server machine. Enter `$combattest stop` in chat to stop the bridge.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe duel.py --full-controls --random-bouts 3
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --updates 2 --output runs/full-controls
```

The smoke test checks a hit and a block first. `--full-controls` adds footwork, aim, facing away and reacquiring a target, kick animation, and all directional attack and guard inputs. Checkpoints from protocol v1 use a different action and observation space. Start a new run.

## Parallel training

On one private Duel server, stand on foot as an admin, face an open area, and enter `$combattest batch 20`. The server looks for 20 separated sites with navigable ground and opens ports 49317 through 49336. If the map does not have enough usable space nearby, the command reports how many sites it found. If a port is unavailable, it closes any listeners already opened for that batch.

```powershell
.\.venv\Scripts\python.exe train.py --arenas 20 --rollout-steps 64 --updates 2 --output runs/batch-20
```

For a different range, specify the first port in chat, for example `$combattest batch 20 49400`, and pass `--port 49400` to the trainer. Use `$combattest stop` to stop the batch.

### Placing arenas individually

Use one private Duel server. As an admin, spawn on foot, face clear ground, and enter `$combattest train 49317`. Move to another clear spot at least 16 meters away from the first arena's center. Face the second area and enter `$combattest train 49318`. Each arena is centered three meters in front of you when you enter the command. If a fighter leaves its arena's six-meter radius, that bout times out and resets independently. Verify each bridge with `duel.py --port <port>`, then start one trainer.

Use `$combattest stop 49318` to remove just the second arena, or `$combattest stop` to stop all training arenas.

```powershell
.\.venv\Scripts\python.exe train.py --ports 49317 49318 --rollout-steps 256 --updates 50 --output runs/parallel
```

Both duels contribute to the same policy. Decisions collected per update equal rollout steps multiplied by the number of duels. The example collects 512 decisions per update, matching the single-duel default. Omitting `--ports` still runs one duel on port 49317. Every bridge listens on loopback and handles one duel at a time. A parallel batch waits for its slowest duel.

## Duel bridge protocol v2

Use UTF-8 JSON lines on a loopback TCP port, 49317 by default, with one request in flight per duel. A step applies both controls together on the mission thread and reports the result after at least 50 ms of simulated time, rounded to a mission tick. After 250 ms without a decision, fighters release their controls while the bout continues. Bouts time out at 1200 steps (at least 60 seconds of decision time) or on leaving the arena. A mission change closes the connection.

| Request | Example |
|---|---|
| Reset with fixed smoke-test spawns | `{"version":2,"command":"reset","seed":null}` |
| Reset with repeatable variation | `{"version":2,"command":"reset","seed":42}` |
| Step | `{"version":2,"command":"step","episode":7,"step":0,"actions":[[2,0,0,0,1],[0,0,0,0,3]]}` |

Each action is `[forward, strafe, yaw, pitch, weapon]`. For the first four axes choose 0 for no input, 1 for back/left/right/down respectively, or 2 for forward/right/left/up. Yaw changes by 7.5 degrees and pitch by 5 degrees per decision. Pitch is limited to 45 degrees in either direction.

| Weapon ID | Input |
|---|---|
| 0 | Neutral or release held attack |
| 1, 4, 5, 6 | Hold left, right, up, or down attack |
| 2, 3, 7, 8 | Hold left, right, up, or down guard |
| 9 | Kick on press. Choose another weapon input before kicking again. |

A success returns `version`, `episode`, `step`, `observations`, `hits`, `blocks`, and `outcome`. An error returns `version` and `error`. Reset advances the episode ID and returns step 0. Step must match the last returned episode and step, then advances the step by one. `hits` and `blocks` contain two per-step counts in attacker order. Outcomes are `running`, `a_win`, `b_win`, `draw`, and `timeout`.

Each fighter gets a 31-value observation. Direction groups are one-hot in left, right, up, down order. Attack phase is 0 for none, 0.5 for ready, and 1 for release.

| Indices | Values |
|---|---|
| 0–1 | Own and visible opponent health fractions |
| 2–4 | Observed or remembered distance divided by 5, bearing sine, bearing cosine |
| 5–7 | Opponent visible, opponent seen this bout, time since last seen divided by 3 seconds |
| 8–10 | Own look pitch sine, opponent forward and right velocity relative to own facing divided by 8 |
| 11–14 | Own/opponent attack phase, own/opponent kick animation |
| 15–22 | Own/opponent attack direction groups |
| 23–30 | Own/opponent guard direction groups |

The server checks a 120-degree horizontal and 60-degree vertical field of view and casts rays from the fighter's eyes to the opponent's head and torso. If neither point is visible, the observation retains the last seen position and age. Live opponent health, motion, and animation fields are zeroed. Python derives damage rewards from each fighter's own health. Reward is 0.4 times damage dealt minus 0.2 times damage taken, using fractions of maximum health. Wins earn +1, losses earn -1, and a timeout without damage to either fighter costs 0.1.

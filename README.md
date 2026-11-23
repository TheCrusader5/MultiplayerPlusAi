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

## Duel bridge protocol v2

Use UTF-8 JSON lines on loopback TCP port 49317 with one request in flight. A step applies both controls together on the mission thread and reports the result after at least 50 ms of simulated time, rounded to a mission tick. The last input persists between steps. A 250 ms simulation gap truncates the bout on the next matching step. Bouts also time out at 400 steps. A mission change closes the connection.

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

The server checks a 120-degree view cone and casts rays from the fighter's eyes to the opponent's head and torso. If neither point is visible, the observation retains the last seen position and age. Live opponent health, motion, and animation fields are zeroed. Python derives damage rewards from each fighter's own health so sight loss alone yields no reward.

# MultiplayerPlusAi

Python tools for training Bannerlord duel bots against frozen copies of their own policy. MultiplayerPlusEngine runs the duel simulation and applies both agents' inputs.

## Run locally

Use Python 3.10 or later. Start the dedicated server and its duel bridge on the same machine as Python. The current C# combat test still needs that bridge. The protocol below specifies what it must implement.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe duel.py --random-bouts 10
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --updates 20 --output runs/duel
```

The smoke test checks that a released left swing damages an idle defender and that one horizontal guard blocks it. It then runs the requested number of randomly controlled duels. The trainer alternates the learner's spawn side, samples opponents from its last eight frozen policies, and saves one checkpoint per PPO update. Pick a new output folder for a new run.

## Duel bridge protocol

Listen on loopback TCP port 49317 by default. Exchange one UTF-8 JSON object per line, version 1, with one request in flight. Use 50 ms of simulation time per step. The bridge must apply both actions on the mission thread for the same interval, then snapshot the result. Socket waits belong off the mission thread. If a deadline is missed or the mission changes, return an error and end the episode. Keep a bout within 400 decisions.

| Request | Fields |
|---|---|
| Reset | `{"version":1,"command":"reset","seed":42}` |
| Step | `{"version":1,"command":"step","episode":7,"step":0,"actions":[1,0]}` |
| Error response | `{"version":1,"error":"The duel ended before the action was applied."}` |

A successful response includes `version`, `episode`, `step`, `observations`, `hits`, `blocks`, and `outcome`. Reset returns a new episode ID greater than the previous one, step 0, zero event counts, and outcome `running`. Step advances the step number by one with the same episode ID. Reject a stale episode or step before applying its actions. A reset may replace a running episode. Either sequence can end by closing the connection.

Actions are ordered A then B. An attack flag stays held across consecutive attack steps and clears when the action changes.

| ID | Input |
|---|---|
| 0 | Neutral or release a held attack |
| 1 | Hold left attack |
| 2 | Hold left guard |
| 3 | Hold right guard |

`observations` contains one nine-value vector per fighter in A/B order. Each vector has own health fraction, opponent health fraction, normalized distance, sine and cosine of the relative opponent bearing, own attack active, opponent attack visible, own guard active, and opponent guard visible. For distance, use horizontal separation in meters divided by 5 and clamp to 1. Use 0 or 1 for the four flags. Preserve a final observation on death with zero health for the defeated fighter.

`hits` and `blocks` each contain two per-step counts indexed by attacker. Count damaging unblocked contacts in `hits` and contacts stopped by a guard in `blocks`. Outcome is one of `running`, `a_win`, `b_win`, `draw`, or `timeout`. Reward is +1 for a win, −1 for a loss, plus 0.2 times net opponent health lost per step.
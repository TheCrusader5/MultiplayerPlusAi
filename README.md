# MultiplayerPlusAi

Train Bannerlord duel bots against frozen copies of their own PPO policy.

Deploy the server build with the protocol v3 bridge. In Duel, spawn on foot as an admin, face clear ground, and enter $botduel. Run the following on the server machine from this directory.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe duel.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --updates 2 --output runs/v3-first
```

duel.py uses the standard library and checks that a scripted knockout reports the killing blow to both fighters. Stop the bridge with $botduel stop. The trainer saves a checkpoint after each update. Use a fresh output directory. Checkpoints from earlier protocols have different input and action spaces.

## Protocol v3

The bridge uses newline-delimited UTF-8 JSON on 127.0.0.1:49317. Send a reset with version 3 and command "reset". Each step sends version 3, command "step", the last episode and step, and two action objects in fighter order.

An action can include right and forward (-1 to 1), yaw and pitch (absolute degrees, optional), weapon (None, AttackLeft, AttackRight, AttackUp, AttackDown, BlockLeft, BlockRight, BlockUp, or BlockDown), and jump, kick, bash, and crouch booleans. Omitted controls are neutral. Omitted aim axes follow the server's aim rules.

A successful reply contains episode, step, outcome and two observations. Each observation contains self, visibleAgents, and combatFeedback. The policy uses one visible opponent slot, relative position, and per-step combat feedback. The full snapshots remain available through DuelClient. Rewards favor damage dealt over damage taken, give a win/loss bonus, and mildly penalize a zero-damage timeout.

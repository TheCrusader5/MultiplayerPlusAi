# MultiplayerPlusAi

Train Bannerlord duel bots with PPO against native Bannerlord AI.

Deploy the server build with the protocol v3 bridge. In Duel, spawn on foot as an admin, face clear ground, and enter $botduel. Run the following on the server machine from this directory.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe duel.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --updates 2 --output runs/v3-first
```

duel.py uses the standard library and checks that a scripted knockout reports the killing blow to both fighters. Stop the bridge with $botduel stop. The trainer saves a checkpoint after each update. Use a fresh output directory. Checkpoints from earlier protocols have different input and action spaces.

## Parallel training and native opponents

In Duel admin chat, stand facing a large clear area and enter $botduel batch 4. Run the trainer on the server machine.

```powershell
.\.venv\Scripts\python.exe train.py --arenas 4 --rollout-steps 256 --output runs/native-4
```

The server and trainer support up to 50 arenas, subject to space on the map. The four arenas use ports 49317–49320 and contribute 1,024 decisions to each PPO update. Each bout spawns a fresh opponent from a random duel-roster character class with the standard training loadout. To use a different starting port, pass it after the arena count in chat and pass --port to the trainer. Use $botduel stop to close the whole batch.

## Protocol v3

The bridge uses newline-delimited UTF-8 JSON on 127.0.0.1:49317. Send a reset with version 3 and command "reset". An optional nativeSide (0 or 1) on reset gives that fighter to Bannerlord AI for the bout. The server echoes nativeSide and returns opponentClassId for native bouts. Each step sends version 3, command "step", the last episode and step, and two action objects in fighter order.

An action can include right and forward (-1 to 1), yaw and pitch (absolute degrees, optional), weapon (None, AttackLeft, AttackRight, AttackUp, AttackDown, BlockLeft, BlockRight, BlockUp, or BlockDown), and jump, kick, bash, and crouch booleans. Omitted controls are neutral. Omitted aim axes follow the server's aim rules.

A successful reply contains episode, step, outcome and two observations. Each observation contains self, visibleAgents, and combatFeedback. The policy uses one visible opponent slot, relative position, and per-step combat feedback. The full snapshots remain available through DuelClient. The trainer awards 1 point per HP of damage dealt to the opponent and 0.001 points per decision the learner survives. A confirmed block against the duel opponent earns 0.5 points, capped at 3 per round. Duels can run for up to 2,400 decisions (about 120 seconds of game time) and end early if a fighter goes out of bounds. Each completed round logs its opponent class, outcome, damage points, survival points, block points, and total.

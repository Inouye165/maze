# Full Known Map And Playback Architecture

This change keeps training centered on the full human-known map while making local tactics and playback behavior modular.

Control files:

- Observation construction: `src/maze_rl/envs/observation.py`
- Environment observation and legality masks: `src/maze_rl/envs/maze_env.py`
- Curriculum definition and serialization: `src/maze_rl/config.py`
- CLI training and playback flags: `src/maze_rl/cli.py`
- Playback mode execution: `src/maze_rl/training/showcase.py`
- Viewer and app mode labeling: `src/maze_rl/render/replay_viewer.py`, `src/maze_rl/render/view_state.py`, `src/maze_rl/render/control_app.py`

Observation design:

- The primary observation block remains the full configured remembered map.
- Unknown cells remain zeroed until discovered.
- An optional player-centered local tactical patch is appended after the global map block.
- Scalar rollout features remain at the end of the vector.

Curriculum design:

- Early stages use smaller whole mazes.
- Later stages restore the full maze size and full monster pressure.
- The observation schema does not collapse to a local-only crop during early stages.

Masking and playback design:

- Environment action masks block illegal actions only.
- Raw playback does not evaluate or apply heuristic override logic.
- Assisted playback enables the existing heuristic rescue logic explicitly.
- Heuristic playback remains available as a baseline comparison mode.
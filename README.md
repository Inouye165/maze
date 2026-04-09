# Maze RL Lab

Maze RL Lab is a Python-first reinforcement-learning project for testing whether a maze agent actually improves over time.

The project replaces the old browser implementation with a deterministic Gymnasium environment, Stable-Baselines3 training, immutable checkpoints, frozen held-out evaluation, and a minimal local viewer.

## Purpose

This repo is a learning lab, not a polished game.

The main questions it is built to answer are:

- Does the policy improve from checkpoint 0 to 50, 100, 250, and beyond?
- Does improvement hold on the same held-out maze seed?
- Can training, evaluation, and watch mode stay reproducible and frozen?

## Why PPO by Default

The default algorithm is PPO.

PPO is the cleanest first baseline here because:

- the environment has a discrete action space with 40 actions
- the observation is fully specified and shape-validated
- checkpointing and deterministic evaluation are straightforward
- it is stable enough for a first lab without replay-buffer complexity

`sb3-contrib` is included and the code supports `recurrent_ppo` as an optional algorithm for later experiments. The default stays with PPO because this first version is intentionally clean and maintainable.

## Repo Structure

```text
maze/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore
  src/
    maze_rl/
      __init__.py
      config.py
      cli.py
      envs/
        __init__.py
        entities.py
        maze_env.py
        maze_generator.py
        observation.py
        rewards.py
      training/
        __init__.py
        checkpointing.py
        evaluate.py
        metrics.py
        train.py
      render/
        __init__.py
        replay_viewer.py
      policies/
        __init__.py
        model_factory.py
  tests/
    test_env_basic.py
    test_seed_reproducibility.py
    test_checkpoint_flow.py
    test_frozen_eval.py
```

## Setup

Create and activate a virtual environment, then install the project in editable mode.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Main Commands

Train from scratch:

```powershell
python -m maze_rl.cli train --episodes 500
```

Train without the easier early curriculum:

```powershell
python -m maze_rl.cli train --episodes 500 --disable-curriculum
```

Evaluate one checkpoint on one seed:

```powershell
python -m maze_rl.cli eval --checkpoint checkpoints/ckpt_0100.zip --seed 12345
```

Enable per-step debug tracing during evaluation:

```powershell
python -m maze_rl.cli eval --checkpoint checkpoints/ckpt_0100.zip --seed 12345 --debug-trace
```

Evaluate one checkpoint on multiple held-out seeds:

```powershell
python -m maze_rl.cli eval --checkpoint checkpoints/ckpt_0100.zip --seeds 12345 12346 12347
```

Watch one checkpoint on one seed:

```powershell
python -m maze_rl.cli watch --checkpoint checkpoints/ckpt_0100.zip --seed 12345
```

Watch with per-step debug tracing and on-screen coordinate overlay:

```powershell
python -m maze_rl.cli watch --checkpoint checkpoints/ckpt_0100.zip --seed 12345 --debug-trace
```

Open the local control app:

```powershell
python -m maze_rl.cli app
```

Run a sequential showcase across multiple checkpoints on the same seed:

```powershell
python -m maze_rl.cli showcase --checkpoints 0 50 100 200 500 1000 --seed 12345
```

Run the same showcase without opening pygame:

```powershell
python -m maze_rl.cli showcase --checkpoints 0 50 100 200 500 1000 --seed 12345 --headless
```

Showcase with per-step trace output:

```powershell
python -m maze_rl.cli showcase --checkpoints 0 50 100 200 500 1000 --seed 12345 --headless --debug-trace
```

Compare multiple checkpoints on the same held-out seed:

```powershell
python -m maze_rl.cli compare --checkpoints 0 50 100 250 --seed 12345
```

Missing checkpoints are skipped with a clear message instead of aborting the whole compare run.

The showcase command also skips missing checkpoints, writes a JSON summary to `replays/`, and prints a row-per-checkpoint summary table.

The app command opens a larger local pygame control panel with:

- a game/playback area
- Baseline Legal Mover, Run Current AI, Replay Last Run, and Compare Milestones modes
- Start, Pause, Resume, Replay, and Step controls
- playback speed controls
- seed input and checkpoint selection
- training buttons for +50, +100, +500, Continue Training, Stop Training, and training-mode toggle
- a legend labeling Human AI, Monster, and Exit without relying on color alone

The app is now the primary product surface. Normal local workflow is:

```powershell
python -m maze_rl.cli app
```

The default app training stage is `maze-only`:

- monster pressure disabled during training
- player speed fixed to 1
- action space reduced to 4 directions
- legal-action masking enabled through MaskablePPO
- checkpoints written under `checkpoints/maze_only/`

The later `full-monster` stage keeps the monster system intact and writes checkpoints under `checkpoints/full_monster/`.

Run tests, including the repo's Pylint convention check for line length and final newlines:

```powershell
python -m pytest -q
```

Use recurrent PPO instead of PPO:

```powershell
python -m maze_rl.cli train --episodes 500 --algorithm recurrent_ppo
```

## What Gets Saved

Immutable checkpoints are saved at episodes:

- 0
- 50
- 100
- 250
- 500

After that, the project saves recurring checkpoints using a configurable interval.

Each checkpoint produces:

- a model file in `checkpoints/`
- a metadata JSON next to it
- episode count
- timestep count
- training settings
- summary metrics
- held-out frozen evaluation summary

## Metrics That Matter

The important metrics for learning progress are:

- outcome: `escaped`, `caught`, `stall`, or `timeout`
- steps
- coverage
- discovered cells
- frontier expansion count and rate
- revisits
- oscillations
- dead-end entries
- blocked or illegal wall moves
- reward
- start distance from monster to player
- time to capture when caught
- frontier reached rate
- peak no-progress streak

The strongest signal is improvement on the same held-out seed across checkpoints.

## What Improvement Should Look Like

Early checkpoints should usually look like this:

- checkpoint 0: mostly caught, stalled, or timed out
- checkpoint 50: better coverage, fewer revisits, occasional escapes
- checkpoint 100: more stable pathing and more consistent escapes on easier held-out layouts
- checkpoint 250+: better escape rate, lower revisit churn, and fewer dead-end traps

Exact numbers will depend on hardware, seeds, and algorithm settings. The point is not a perfect game. The point is that checkpoint-to-checkpoint comparison should show directionally better behavior.

## Environment Design

The environment includes:

- deterministic seeded maze generation
- player start
- monster start
- exit
- walls and corridors
- flattened discrete actions that map to direction plus speed

Action mapping:

- 4 directions: north, east, south, west
- 10 speeds: 1 through 10
- total action count: 40

Monster speed is fixed at 6 movement substeps per environment step. Player speed is selected by the action.

## Early Curriculum

Training now uses a configurable curriculum defined in `MazeConfig`.

The default schedule is meant to make the first visible milestones easier to learn:

- episodes 0 to 19: 9 x 9 mazes, monster speed 1, monster delayed by 10 steps
- episodes 20 to 49: 11 x 11 mazes, monster speed 2, monster delayed by 7 steps
- episodes 50 to 79: 13 x 13 mazes, monster speed 4, monster delayed by 4 steps
- episodes 80 and later: full 15 x 15 difficulty, monster speed 6, monster delayed by 1 step

Frozen evaluation and watch mode always use the full non-curriculum setting from checkpoint metadata.

## Frozen Evaluation

Evaluation and watch mode are frozen by design:

- no learning updates
- deterministic policy inference
- no checkpoint mutation
- no replay or training-state writes

That makes comparison between checkpoints meaningful.

## Tests

Run the test suite with:

```powershell
pytest
```

The tests cover:

- Gym reset and step contract
- seeded maze reproducibility
- checkpoint save and load
- frozen evaluation not mutating checkpoint artifacts
- multi-seed evaluation aggregation
- training curriculum versus frozen evaluation behavior

## Current Limitations

- the viewer is intentionally minimal
- the monster policy is deterministic shortest-path pursuit, which is useful for reproducibility but not for rich adversarial behavior
- the baseline ships with PPO first; recurrent experiments are available but not tuned yet
- the training defaults are meant to be workable, not optimal

## Future Ideas

- tune reward weights with held-out seed sweeps
- add curriculum schedules across maze sizes
- compare PPO against RecurrentPPO directly
- add CSV logging for multi-run comparisons
- add optional video export from replay runs
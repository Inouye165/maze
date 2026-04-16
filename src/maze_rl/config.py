"""Central configuration for Maze RL Lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights for the maze environment."""

    survival_reward: float = 0.03
    exploration_reward: float = 2.2
    frontier_reward: float = 0.9
    revisit_penalty: float = -0.55
    revisit_depth_penalty: float = -0.18
    oscillation_penalty: float = -1.8
    dead_end_penalty: float = -1.4
    deeper_dead_end_penalty: float = -1.2
    avoidable_visible_dead_end_penalty: float = -0.65
    trap_threat_penalty: float = -4.0
    blocked_move_penalty: float = -0.5
    exit_progress_reward: float = 0.2
    safety_gain_reward: float = 0.1
    safety_loss_penalty: float = -0.2
    win_reward: float = 35.0
    caught_penalty: float = -32.0
    timeout_penalty: float = -12.0
    stall_penalty: float = -10.0


@dataclass(frozen=True)
class CurriculumStage:
    """One training curriculum stage."""

    start_episode: int
    rows: int
    cols: int
    monster_speed: int
    monster_activation_delay: int
    max_episode_steps: int
    stall_threshold: int
    monster_move_interval: int = 1
    label: str = ""


@dataclass(frozen=True)
class MazeConfig:
    """Environment settings."""

    rows: int = 15
    cols: int = 15
    vision_range: int = 4
    max_player_speed: int = 10
    monster_speed: int = 6
    monster_activation_delay: int = 0
    monster_move_interval: int = 1
    max_episode_steps: int = 90
    stall_threshold: int = 14
    train_seed_base: int = 10_000
    fixed_maze_seed: int | None = None
    focused_seed_jump_max: int = 1000
    held_out_seed: int = 12_345
    curriculum_enabled: bool = True
    curriculum: tuple[CurriculumStage, ...] = field(
        default_factory=lambda: (
            CurriculumStage(
                start_episode=0,
                rows=9,
                cols=9,
                monster_speed=1,
                monster_activation_delay=10,
                max_episode_steps=60,
                stall_threshold=20,
                label="bootstrap",
            ),
            CurriculumStage(
                start_episode=20,
                rows=11,
                cols=11,
                monster_speed=2,
                monster_activation_delay=7,
                max_episode_steps=70,
                stall_threshold=18,
                label="easy",
            ),
            CurriculumStage(
                start_episode=50,
                rows=13,
                cols=13,
                monster_speed=4,
                monster_activation_delay=4,
                max_episode_steps=85,
                stall_threshold=16,
                label="intermediate",
            ),
            CurriculumStage(
                start_episode=80,
                rows=15,
                cols=15,
                monster_speed=6,
                monster_activation_delay=1,
                max_episode_steps=100,
                stall_threshold=15,
                label="full",
            ),
        )
    )
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(frozen=True)
class TrainingConfig:
    """Training and checkpoint settings."""

    algorithm: str = "ppo"
    seed: int = 7
    episodes: int = 500
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    n_steps: int = 128
    batch_size: int = 32
    ent_coef: float = 0.02
    checkpoint_episodes: tuple[int, ...] = (0, 50, 100, 250, 500)
    recurring_checkpoint_interval: int = 250
    held_out_eval_episodes: int = 1
    log_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    replay_dir: Path = Path("replays")


def as_serializable_dict(config: Any) -> dict[str, Any]:
    """Convert nested dataclasses into plain dictionaries."""

    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def _filter_dataclass_kwargs(dataclass_type: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown serialized fields so old checkpoints stay loadable."""

    allowed_fields = getattr(dataclass_type, "__dataclass_fields__", {})
    return {key: value for key, value in payload.items() if key in allowed_fields}


def maze_config_from_dict(data: dict[str, Any]) -> MazeConfig:
    """Rebuild MazeConfig from serialized metadata."""

    normalized = dict(data)
    reward = normalized.get("reward")
    if isinstance(reward, dict):
        normalized["reward"] = RewardConfig(**_filter_dataclass_kwargs(RewardConfig, reward))
    curriculum = normalized.get("curriculum")
    if isinstance(curriculum, list):
        normalized["curriculum"] = tuple(
            CurriculumStage(**_filter_dataclass_kwargs(CurriculumStage, item))
            for item in curriculum
            if isinstance(item, dict)
        )
    return MazeConfig(**_filter_dataclass_kwargs(MazeConfig, normalized))


def training_config_from_dict(data: dict[str, Any]) -> TrainingConfig:
    """Rebuild TrainingConfig from serialized metadata."""

    normalized = dict(data)
    for key in ("log_dir", "checkpoint_dir", "replay_dir"):
        if key in normalized and not isinstance(normalized[key], Path):
            normalized[key] = Path(normalized[key])
    if isinstance(normalized.get("checkpoint_episodes"), list):
        normalized["checkpoint_episodes"] = tuple(normalized["checkpoint_episodes"])
    return TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, normalized))

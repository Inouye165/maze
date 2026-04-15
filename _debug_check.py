"""Quick diagnostic: can we resume training from the latest checkpoint?"""

from maze_rl.training.checkpointing import latest_checkpoint, load_checkpoint_metadata
from maze_rl.config import maze_config_from_dict
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.training.train import maze_config_for_training_mode
from maze_rl.policies.model_factory import load_model_from_checkpoint, CheckpointCompatibilityError
from maze_rl.training.metrics import RollingTrainingSummary
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

latest = latest_checkpoint("checkpoints/maze_only")
print(f"Latest: episode={latest[0]}, path={latest[1]}")
metadata = load_checkpoint_metadata(latest[1])
summary = metadata.get("training_summary", {})
print(f"Summary: episodes_seen={summary.get('episodes_seen')}, wins={summary.get('wins')}")

maze_cfg = maze_config_for_training_mode(
    maze_config_from_dict(metadata["maze_config"]), "maze-only"
)
env = MazeEnv(maze_cfg, training_mode=True)
print(f"Env action_space.n = {env.action_space.n}")

vec = DummyVecEnv([lambda: Monitor(env)])
try:
    model = load_model_from_checkpoint(latest[1], vec)
    print(f"Model loaded OK, action_space={model.action_space}")
except CheckpointCompatibilityError as e:
    print(f"Model INCOMPATIBLE: {e}")
except Exception as e:
    print(f"Model load FAILED: {type(e).__name__}: {e}")

# Test summary hydration
rs = RollingTrainingSummary()
rs.load_snapshot(summary)
print(f"Hydrated summary: total_episodes={rs.total_episodes}, total_wins={rs.total_wins}")
snap = rs.snapshot()
print(f"Snapshot after hydration: episodes_seen={snap['episodes_seen']}, wins={snap['wins']}")

"""Quick mini-training to verify accumulation works."""

import threading
from pathlib import Path
from maze_rl.training.train import continue_training_from_latest
from maze_rl.training.checkpointing import latest_checkpoint, load_checkpoint_metadata

ckpt_dir = "checkpoints/maze_only"

latest = latest_checkpoint(ckpt_dir)
print(f"BEFORE: latest episode={latest[0]}")
meta = load_checkpoint_metadata(latest[1])
print(f"BEFORE: episodes_seen={meta['training_summary'].get('episodes_seen')}")

progress_log = []
def on_progress(p):
    snap = p.get("training_summary_snapshot", {})
    progress_log.append({
        "status": p.get("status"),
        "completed": p.get("completed_episodes"),
        "episodes_seen": snap.get("episodes_seen"),
    })

result = continue_training_from_latest(
    additional_episodes=2,
    checkpoint_dir=ckpt_dir,
    training_mode="maze-only",
    progress_callback=on_progress,
)
print(f"Training result: final_episode_count={result.final_episode_count}")

latest2 = latest_checkpoint(ckpt_dir)
print(f"AFTER: latest episode={latest2[0]}")
meta2 = load_checkpoint_metadata(latest2[1])
print(f"AFTER: episodes_seen={meta2['training_summary'].get('episodes_seen')}")

# Show a few progress entries
for entry in progress_log[-5:]:
    print(f"  progress: {entry}")

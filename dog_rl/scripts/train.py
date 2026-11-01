"""Train a dog locomotion policy with Isaac Lab's unified train CLI.

Registers the Dog-* tasks, then hands all arguments to isaaclab_rl's train entry point, e.g.

    cd ~/IsaacLab
    uv run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/train.py --task Dog-Velocity-Flat-v0
    uv run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/train.py --task Dog-Velocity-Flat-v0 \
        --num_envs 64 --max_iterations 5      # quick smoke test

Logs and checkpoints go to ~/IsaacLab/logs/rsl_rl/dog_flat/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warp as wp  # noqa: E402

wp.config.enable_backward = False

import dog_rl.tasks  # noqa: E402,F401  (registers Dog-* gym ids)
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_train_cli(sys.argv[1:]))

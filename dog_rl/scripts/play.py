"""Run a trained dog policy with Isaac Lab's unified play CLI (also exports the policy).

    cd ~/IsaacLab
    uv run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/play.py --task Dog-Velocity-Flat-Play-v0 --viz kit
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warp as wp  # noqa: E402

wp.config.enable_backward = False

import dog_rl.tasks  # noqa: E402,F401  (registers Dog-* gym ids)
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_play_cli(sys.argv[1:]))

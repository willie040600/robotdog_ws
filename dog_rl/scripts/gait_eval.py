"""Measure the gait of an exported policy under a fixed velocity command.

Reports per-foot step frequency, air (swing) time, foot lift height, and the achieved forward
speed / yaw rate, so gaits can be compared by numbers instead of by eye.

    cd ~/IsaacLab
    uv run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/gait_eval.py \
        --policy ~/IsaacLab/logs/rsl_rl/dog_flat/<run>/exported/policy.pt --vx 0.3 [--wz 0.0] [--viz kit]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description="Gait statistics for an exported dog policy.")
parser.add_argument("--policy", required=True, help="Path to exported policy.pt (TorchScript).")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--vx", type=float, default=0.3, help="Forward speed command [m/s].")
parser.add_argument("--wz", type=float, default=0.0, help="Yaw rate command [rad/s].")
parser.add_argument("--seconds", type=float, default=10.0, help="Measured duration (after 2 s warm-up).")
parser.add_argument("--physics", default="isaacsim_physx", help="Physics preset: isaacsim_physx or newton_mjwarp.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dog_rl.tasks  # noqa: E402,F401
from dog_rl.tasks import mdp as dog_mdp  # noqa: E402
from dog_rl.tasks.flat_env_cfg import DogFlatEnvCfg_PLAY  # noqa: E402


def main():
    env_cfg = DogFlatEnvCfg_PLAY()
    # the train/play CLIs resolve the physics preset; building the cfg directly we pick it ourselves
    env_cfg.sim.physics = getattr(env_cfg.sim.physics, args_cli.physics)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 1000.0
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    cmd.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
    cmd.rel_standing_envs = 0.0
    if hasattr(cmd, "rel_turn_in_place_envs"):  # every env gets exactly the requested command
        cmd.rel_turn_in_place_envs = 0.0
    cmd.resampling_time_range = (1000.0, 1000.0)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make("Dog-Velocity-Flat-Play-v0", cfg=env_cfg)
        uenv = env.unwrapped
        policy = torch.jit.load(os.path.expanduser(args_cli.policy), map_location=uenv.device)
        policy.eval()

        robot = uenv.scene["robot"]
        sensor = uenv.scene.sensors["contact_forces"]
        sensor_ids, _ = sensor.find_sensors(dog_mdp.CALF_NAMES, preserve_order=True)
        robot_cfg = dog_mdp.calf_cfg("robot")
        robot_cfg.resolve(uenv.scene)

        obs, _ = env.reset()
        dt = uenv.step_dt
        warmup = int(2.0 / dt)
        steps = int(args_cli.seconds / dt)
        n, nf = uenv.num_envs, len(dog_mdp.CALF_NAMES)
        prev_contact = torch.ones(n, nf, dtype=torch.bool, device=uenv.device)
        touchdowns = torch.zeros(n, nf, device=uenv.device)
        air_sum = torch.zeros(n, nf, device=uenv.device)
        swing_max = torch.zeros(n, nf, device=uenv.device)
        lift_sum = torch.zeros(n, nf, device=uenv.device)
        vx_sum = torch.zeros(n, device=uenv.device)
        wz_sum = torch.zeros(n, device=uenv.device)
        resets = 0

        with torch.inference_mode():
            for i in range(warmup + steps):
                actions = policy(obs["policy"])
                obs, _, terminated, truncated, _ = env.step(actions)
                if i < warmup:
                    continue
                resets += int((terminated | truncated).sum())
                forces = sensor.data.net_normal_forces_w.torch[:, sensor_ids]
                contact = torch.linalg.norm(forces, dim=-1) > 1.0
                foot_pos, _ = dog_mdp._foot_state(uenv, robot_cfg)
                foot_z = foot_pos[..., 2] - dog_mdp.FOOT_RADIUS - uenv.scene.env_origins[:, 2:3]
                swing_max = torch.where(contact, swing_max, torch.maximum(swing_max, foot_z))
                td = contact & ~prev_contact
                touchdowns += td
                air_sum += td * sensor.data.last_air_time.torch[:, sensor_ids]
                lift_sum += td * swing_max
                swing_max = torch.where(td, torch.zeros_like(swing_max), swing_max)
                prev_contact = contact
                vx_sum += robot.data.root_lin_vel_b.torch[:, 0]
                wz_sum += robot.data.root_ang_vel_b.torch[:, 2]

        T = steps * dt
        td = touchdowns.clamp(min=1)
        print(f"\n[GAIT] physics={args_cli.physics}", flush=True)
        print(f"[GAIT] command vx={args_cli.vx:+.2f} m/s wz={args_cli.wz:+.2f} rad/s, {n} envs x {T:.0f} s", flush=True)
        print(f"[GAIT] achieved vx={float(vx_sum.mean() / steps):+.3f} m/s  wz={float(wz_sum.mean() / steps):+.3f} rad/s"
              f"  resets={resets}", flush=True)
        for k, name in enumerate(dog_mdp.CALF_NAMES):
            print(
                f"[GAIT] {name[:2]}: steps/s={float(touchdowns[:, k].mean() / T):5.2f}  "
                f"air time={float((air_sum[:, k] / td[:, k]).mean()):.3f} s  "
                f"lift={float((lift_sum[:, k] / td[:, k]).mean()) * 100:4.1f} cm",
                flush=True,
            )
        env.close()


if __name__ == "__main__":
    main()

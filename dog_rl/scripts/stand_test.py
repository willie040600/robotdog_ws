"""Sanity check for DOG_CFG: spawn dogs on flat ground, hold the standing pose, report height/tilt.

    cd ~/IsaacLab
    uv run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/stand_test.py [--num_envs 16] [--viz kit]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description="Stand test for the 8-DOF dog.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--duration", type=float, default=5.0, help="Seconds of simulated time.")
parser.add_argument("--spawn_z", type=float, default=None, help="Override spawn height of the base [m].")
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402

from dog_rl.dog_cfg import DOG_CFG, DOG_STAND_HEIGHT  # noqa: E402


@configclass
class DogSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = replace(DOG_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.15])
        scene = instantiate(DogSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0))
        sim.reset()
        robot = scene["robot"]
        print(f"[INFO]: Setup complete. joints={robot.joint_names}", flush=True)

        # start every dog in the standing pose (the USD's own joint positions are all 0, and the
        # calf's 0 is outside its [-2.6, -0.6] limit), like the RL env's reset event does
        root_pose = robot.data.default_root_pose.torch.clone()
        if args_cli.spawn_z is not None:
            root_pose[:, 2] = args_cli.spawn_z
        root_pose[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        scene.reset()

        dt = sim.get_physics_dt()
        stand = robot.data.default_joint_pos.torch.clone()
        steps = int(args_cli.duration / dt)
        for i in range(steps + 1):
            robot.set_joint_position_target_index(target=stand)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if i % int(1.0 / dt) == 0:
                origins = scene.env_origins
                z = robot.data.root_pos_w.torch[:, 2] - origins[:, 2]
                g = robot.data.projected_gravity_b.torch  # (0,0,-1) when level
                tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
                q = robot.data.joint_pos.torch
                err = (q - stand).abs().max()
                fallen = int((z < 0.08).sum())
                print(
                    f"t={i * dt:4.1f}s  height mean={z.mean():.3f} min={z.min():.3f} max={z.max():.3f}  "
                    f"tilt max={tilt.max():5.2f}deg  joint err max={err:.3f}rad  fallen={fallen}/{len(z)}",
                    flush=True,
                )
        print(f"[RESULT] target stand height ~{DOG_STAND_HEIGHT} m", flush=True)
        # per-body / per-joint detail for env 0
        d = robot.data
        masses = d.body_mass.torch[0]
        print(f"  total mass={masses.sum():.3f} kg", flush=True)
        for b, name in enumerate(robot.body_names):
            print(
                f"  body {name:10s} mass={masses[b]:.3f} com_b={[round(float(v), 4) for v in d.body_com_pos_b.torch[0, b]]}",
                flush=True,
            )
        tau = d.applied_torque.torch[0]
        # foot point check (same math as dog_rl/tasks/mdp.py): foot bottom should be ~0 m while standing
        from isaaclab.utils.math import quat_apply

        from dog_rl.tasks.mdp import CALF_NAMES, FOOT_OFFSET, FOOT_RADIUS

        ids = [robot.body_names.index(n) for n in CALF_NAMES]
        pos = d.body_link_pos_w.torch[0, ids]
        r = quat_apply(d.body_link_quat_w.torch[0, ids], torch.tensor(FOOT_OFFSET, device=pos.device).expand_as(pos))
        oz = float(scene.env_origins[0, 2])
        for k, n in enumerate(CALF_NAMES):
            print(
                f"  {n}: knee z={pos[k, 2] - oz:.4f}  foot bottom z={pos[k, 2] + r[k, 2] - FOOT_RADIUS - oz:+.4f}  "
                f"foot x rel knee={r[k, 0]:+.4f}",
                flush=True,
            )
        for j, name in enumerate(robot.joint_names):
            print(
                f"  {name:15s} target={stand[0, j]:+.3f} q={d.joint_pos.torch[0, j]:+.3f} "
                f"kp={d.joint_stiffness.torch[0, j]:.2f} kd={d.joint_damping.torch[0, j]:.3f} "
                f"effort_lim={d.joint_effort_limits.torch[0, j]:.1f} tau={tau[j]:+.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

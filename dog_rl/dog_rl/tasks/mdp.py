"""Dog-specific reward / termination terms.

The foot sphere (radius 0.015) is part of the calf link, 0.12 m below the calf frame origin, which
sits at the knee. So the contact sensor cannot tell "foot on ground" from "knee/shin on ground",
and body positions/velocities of the calf are those of the knee. These helpers compute the
actual foot point from the calf pose.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# foot sphere centre in the calf frame (dog.urdf)
FOOT_OFFSET = (0.0, 0.0, -0.12)
FOOT_RADIUS = 0.015

# Explicit order so robot bodies and contact-sensor bodies line up.
CALF_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]


def calf_cfg(entity: str) -> SceneEntityCfg:
    return SceneEntityCfg(entity, body_names=CALF_NAMES, preserve_order=True)


def _foot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """World position and velocity of the foot sphere centres, shape (num_envs, num_feet, 3)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    pos = asset.data.body_link_pos_w.torch[:, ids]
    quat = asset.data.body_link_quat_w.torch[:, ids]
    lin_vel = asset.data.body_link_lin_vel_w.torch[:, ids]
    ang_vel = asset.data.body_link_ang_vel_w.torch[:, ids]
    offset = torch.tensor(FOOT_OFFSET, device=pos.device).expand_as(pos)
    r = quat_apply(quat, offset)
    return pos + r, lin_vel + torch.cross(ang_vel, r, dim=-1)


def foot_clearance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward feet that move horizontally (swinging) for reaching target_height (foot bottom above ground).

    Same shape as Spot's foot_clearance_reward, but evaluated at the real foot point.
    """
    foot_pos, foot_vel = _foot_state(env, asset_cfg)
    foot_bottom_z = foot_pos[..., 2] - FOOT_RADIUS - env.scene.env_origins[:, 2:3]
    err = torch.square(foot_bottom_z - target_height)
    moving = torch.tanh(tanh_mult * torch.linalg.norm(foot_vel[..., :2], dim=-1))
    return torch.exp(-torch.sum(err * moving, dim=1) / std)


def foot_slip(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize horizontal foot speed while that leg is in contact (anti-shuffling / dragging)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    in_contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold
    _, foot_vel = _foot_state(env, asset_cfg)
    return torch.sum(in_contact * torch.linalg.norm(foot_vel[..., :2], dim=-1), dim=1)


def _gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Gait clock in [0, 1), restarted at every episode reset. Shape (num_envs,)."""
    t = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(t / period, 1.0)


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Observation: (sin, cos) of the gait clock, so the policy can follow the contact schedule.

    On the robot (Gazebo node) this is reproduced with a counter at the 50 Hz policy rate.
    """
    phase = _gait_phase(env, period)
    return torch.stack([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)


def contact_schedule(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float, command_name: str, threshold: float
) -> torch.Tensor:
    """Fraction of feet whose contact state matches a fixed-frequency trot schedule.

    First half of each period FL+RR are in stance and FR+RL swing, second half the opposite. With a
    (near) zero command all four feet should stay down. This pins the step frequency to 1/period
    (v2 without it learned very fast, tiny steps).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold  # (N, 4) FL FR RL RR
    first_half = _gait_phase(env, period) < 0.5
    desired = torch.stack([first_half, ~first_half, ~first_half, first_half], dim=1)
    moving = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.05
    desired = torch.where(moving.unsqueeze(1), desired, torch.ones_like(desired))
    return (contact == desired).float().mean(dim=1)


def _standing(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """True where the whole command (vx, vy, wz) is ~0. Turn-in-place commands are not standing."""
    return torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < threshold


def stand_still_joints(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """With a zero command, penalize joint offsets from the standing pose (stop stepping).

    v1-v4 kept trotting in place and drifted backwards at ~0.06 m/s with a zero command.
    """
    asset = env.scene[asset_cfg.name]
    dev = torch.sum(torch.abs(asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=1)
    return dev * _standing(env, command_name, threshold)


def stand_still_drift(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """With a zero command, penalize planar base speed (no creeping)."""
    asset = env.scene["robot"]
    speed = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    return speed * _standing(env, command_name, threshold)


def turn_in_place_drift(env: ManagerBasedRLEnv, command_name: str, vx_threshold: float, wz_threshold: float) -> torch.Tensor:
    """For turn-in-place commands (|vx| small, |wz| large), penalize planar base speed.

    v5 crept forward 0.1-0.2 m/s while turning in place in Gazebo; Nav2's controller assumes an
    in-place rotation and the dog walked into obstacles.
    """
    cmd = env.command_manager.get_command(command_name)
    turning = (torch.abs(cmd[:, 0]) < vx_threshold) & (torch.abs(cmd[:, 2]) > wz_threshold)
    speed = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_b.torch[:, :2], dim=1)
    return speed * turning


def knee_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:
    """Terminate when any knee (calf frame origin) drops below min_height, i.e. the dog kneels.

    Standing, the knees are ~0.089 m above the ground.
    """
    asset = env.scene[asset_cfg.name]
    knee_z = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    return torch.any(knee_z < min_height, dim=1)

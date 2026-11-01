"""Velocity command with an explicit share of turn-in-place commands.

Nav2's Regulated Pure Pursuit rotates in place to line up with the path, but uniform (vx, wz)
sampling almost never produces vx ~ 0 with a large wz, so v3 learned to turn in place at only
~20% of the commanded rate. This command sets vx = 0 and a yaw rate of at least
turn_in_place_min_wz for a fraction of the environments.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class DogVelocityCommand(UniformVelocityCommand):
    cfg: DogVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        turn = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turn_in_place_envs
        ids = env_ids[turn]
        if len(ids) == 0:
            return
        lo, hi = self.cfg.turn_in_place_min_wz, max(abs(v) for v in self.cfg.ranges.ang_vel_z)
        mag = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
        sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[ids, 0] = 0.0
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = sign * mag


@configclass
class DogVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = DogVelocityCommand

    rel_turn_in_place_envs: float = 0.25
    """Fraction of resampled commands that are pure turn-in-place (vx = vy = 0)."""

    turn_in_place_min_wz: float = 0.3
    """Minimum |yaw rate| of a turn-in-place command [rad/s]."""

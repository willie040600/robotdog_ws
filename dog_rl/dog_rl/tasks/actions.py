"""Joint position action with a random one-step (20 ms) actuation delay.

In Gazebo (and on hardware) the policy's joint targets reach the joint controllers with some
latency (ROS messages, executor scheduling, bridge). v4 never saw any delay in training. Each
environment is assigned, at every reset, either no delay or a one-policy-step delay, so the
policy learns to cope with both.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class DelayedJointPositionAction(JointPositionAction):
    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._delayed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_processed = self._offset.clone().expand(self.num_envs, -1).clone() \
            if isinstance(self._offset, torch.Tensor) else None

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        new = self._processed_actions.clone()
        if self._prev_processed is None:
            self._prev_processed = new.clone()
        # delayed envs apply the previous step's targets this step
        self._processed_actions = torch.where(self._delayed.unsqueeze(1), self._prev_processed, new)
        self._prev_processed = new

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._delayed[ids] = torch.rand(len(ids), device=self.device) < self.cfg.delay_prob
        if self._prev_processed is not None and isinstance(self._offset, torch.Tensor):
            self._prev_processed[ids] = self._offset[ids] if self._offset.shape[0] == self.num_envs \
                else self._offset.expand(len(ids), -1)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction

    delay_prob: float = 0.5
    """Probability that an environment applies its actions one policy step (20 ms) late."""

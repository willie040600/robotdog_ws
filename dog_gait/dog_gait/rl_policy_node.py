"""Run the Isaac Lab trained locomotion policy (dog_rl, task Dog-Velocity-Flat-v0) in Gazebo.

Drop-in replacement for gait_node: same inputs (/cmd_vel) and outputs (/dog/<joint>/cmd_pos).
The observation vector reproduces the training env exactly (dog_rl/dog_rl/tasks/flat_env_cfg.py):

    [0:3]   base angular velocity, body frame        <- /imu angular_velocity
    [3:6]   gravity direction, body frame            <- /imu orientation
    [6:9]   command (vx, vy=0, wz)                   <- /cmd_vel, clipped to the training ranges
    [9:17]  joint positions - standing pose          <- /joint_states (Isaac joint order)
    [17:25] joint velocities                         <- /joint_states
    [25:33] previous action
    [33:35] gait clock (sin, cos), period 0.4 s, counted at the policy rate

action (8, Isaac joint order) -> joint target = standing pose + 0.25 * action, sent at 50 Hz to
the same PD position controllers (kp 40, kd 0.6, 5 Nm) the policy was trained with.

The policy is a 35-128-128-128-8 ELU MLP evaluated with numpy; the weights are converted from the
exported policy.pt with dog_rl/scripts/export_npz.py.
"""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64

# joint order of the Isaac Lab articulation (thighs first, then calves)
ISAAC_JOINTS = [
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
]
STAND_THIGH, STAND_CALF = 0.869, -1.737
DEFAULT_Q = np.array([STAND_THIGH] * 4 + [STAND_CALF] * 4)
LOWER = np.array([-1.5] * 4 + [-2.6] * 4)
UPPER = np.array([1.5] * 4 + [-0.6] * 4)
READY_Q = (0.3, -0.6)  # feet under the hips; held until the robot spawns (see gait_node)


class MlpPolicy:
    def __init__(self, path):
        w = np.load(path)
        self.layers = [(w[f'l{i}_weight'], w[f'l{i}_bias']) for i in (0, 2, 4, 6)]

    def __call__(self, x):
        for k, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if k < 3:
                x = np.where(x > 0.0, x, np.expm1(x))  # ELU
        return x


def gravity_in_body(qx, qy, qz, qw):
    """World gravity direction (0, 0, -1) expressed in the body frame (= R^T g)."""
    # third row of R, negated
    return np.array([
        -(2.0 * (qx * qz - qw * qy)),
        -(2.0 * (qy * qz + qw * qx)),
        -(1.0 - 2.0 * (qx * qx + qy * qy)),
    ])


class RlPolicyNode(Node):

    def __init__(self):
        super().__init__('rl_policy_node')
        default_policy = os.path.join(get_package_share_directory('dog_gait'), 'policies', 'dog_flat_v7.npz')
        p = self.declare_parameter
        policy_path = p('policy_path', default_policy).value
        self.policy = MlpPolicy(policy_path)
        self.rate = p('rate', 50.0).value                 # training: 200 Hz physics, decimation 4
        self.action_scale = p('action_scale', 0.25).value
        self.gait_period = p('gait_period', 0.4).value
        self.vx_range = tuple(p('vx_range', [-0.3, 0.5]).value)
        self.wz_range = tuple(p('wz_range', [-1.0, 1.0]).value)
        self.cmd_timeout = p('cmd_timeout', 0.5).value
        # Training sampled turn-in-place commands with |wz| >= 0.3, and treats tiny commands as
        # "stand still". Raise small in-place yaw commands to this minimum so the policy turns.
        self.min_turn_wz = p('min_turn_wz', 0.3).value
        # In Gazebo the policy creeps forward ~0.1 m/s while turning in place (it does not in Isaac
        # Lab, so this is a sim-to-sim gap). For turn-in-place commands send vx = -comp * |wz|.
        # 0.2 was calibrated in Gazebo (0.8 rad/s for 8 s: 0.52 m -> ~0.13 m displacement); re-tune
        # it for other simulators or hardware, 0 disables it.
        self.turn_vx_comp = p('turn_vx_comp', 0.2).value
        # While walking (|vx| >= 0.05) the policy under-responds to small yaw commands, so Nav2's
        # small path corrections were ignored and the dog drifted into walls (v6). Map the wanted
        # yaw rate to the command that achieves it (piecewise linear on |wz|, sign kept).
        # Calibrated per policy in Gazebo at vx 0.2 (yaw_calib):
        #   v6: 0.1 -> 11%, 0.2 -> 21%, 0.3 -> 28%, 0.4 -> 65%, 0.5 -> 69%
        #   v7: 0.1 -> 6%,  0.2 -> 41%, 0.3 -> 96%, 0.4 -> 60%, 0.5 -> 79%   (defaults below)
        # Set both lists to [0.0, 1.0] to disable.
        self.yaw_want = list(p('yaw_map_want', [0.0, 0.05, 0.1, 0.25, 0.4, 0.8]).value)
        self.yaw_send = list(p('yaw_map_send', [0.0, 0.15, 0.19, 0.28, 0.50, 1.0]).value)
        self.stand_up_time = p('stand_up_time', 2.0).value
        # With a zero command for idle_time, stop running the policy and hold the standing pose
        # (v1-v4 policies keep trotting in place and creep backwards with a zero command).
        self.stand_when_idle = p('stand_when_idle', True).value
        self.idle_time = p('idle_time', 0.3).value
        self.settle_time = p('settle_time', 0.3).value     # blend from last targets to the pose

        self.pubs = {j: self.create_publisher(Float64, f'/dog/{j}/cmd_pos', 10) for j in ISAAC_JOINTS}
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.create_subscription(Imu, 'imu', self.on_imu, 10)

        self.q = None
        self.dq = None
        self.ang_vel = None
        self.gravity = None
        self.cmd = np.zeros(3)
        self.last_cmd_time = None
        self.last_action = np.zeros(8)
        self.policy_start = None
        self.last_policy_time = None
        self.start_time = None
        self.active = False
        self.idle_since = None
        self.standing = False          # holding the standing pose instead of running the policy
        self.stand_from = DEFAULT_Q.copy()
        self.stand_t0 = None
        self.last_targets = DEFAULT_Q.copy()

        self.create_timer(1.0 / self.rate, self.update)
        self.get_logger().info(f'loaded policy {policy_path}')

    # ------------------------------------------------------------------ io
    def on_cmd_vel(self, msg):
        vx = float(np.clip(msg.linear.x, *self.vx_range))
        wz = float(np.clip(msg.angular.z, *self.wz_range))
        if abs(vx) < 0.05 and abs(wz) > 0.02:
            if abs(wz) < self.min_turn_wz:
                wz = math.copysign(self.min_turn_wz, wz)
            vx = -self.turn_vx_comp * abs(wz)
        elif abs(vx) >= 0.05:
            wz = math.copysign(float(np.interp(abs(wz), self.yaw_want, self.yaw_send)), wz)
            wz = float(np.clip(wz, *self.wz_range))
        self.cmd = np.array([vx, 0.0, wz])
        self.last_cmd_time = self.get_clock().now()

    def on_joint_states(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in ISAAC_JOINTS):
            return
        self.q = np.array([msg.position[idx[j]] for j in ISAAC_JOINTS])
        self.dq = np.array([msg.velocity[idx[j]] for j in ISAAC_JOINTS]) if msg.velocity else np.zeros(8)

    def on_imu(self, msg):
        self.ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        o = msg.orientation
        self.gravity = gravity_in_body(o.x, o.y, o.z, o.w)

    def publish(self, targets):
        for j, v in zip(ISAAC_JOINTS, targets):
            self.pubs[j].publish(Float64(data=float(v)))

    # ------------------------------------------------------------------ loop
    def update(self):
        now = self.get_clock().now()
        if self.q is None:
            self.publish([READY_Q[0]] * 4 + [READY_Q[1]] * 4)
            return
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('robot spawned, standing up')
        t = (now - self.start_time).nanoseconds * 1e-9

        # stand up like gait_node, then hand over to the policy
        if t < 0.5 + self.stand_up_time:
            a = min(max((t - 0.5) / self.stand_up_time, 0.0), 1.0)
            a = 0.5 - 0.5 * math.cos(math.pi * a)
            ready = np.array([READY_Q[0]] * 4 + [READY_Q[1]] * 4)
            self.publish(ready + a * (DEFAULT_Q - ready))
            return
        if self.ang_vel is None:
            self.publish(DEFAULT_Q)
            return
        if not self.active:
            self.active = True
            self.policy_start = now
            self.get_logger().info('policy active')

        cmd = self.cmd
        if self.last_cmd_time is None or (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout:
            cmd = np.zeros(3)

        # idle handling: zero command for idle_time -> blend to and hold the standing pose
        if self.stand_when_idle:
            if np.linalg.norm(cmd) < 1e-3:
                if self.idle_since is None:
                    self.idle_since = now
                if not self.standing and (now - self.idle_since).nanoseconds * 1e-9 >= self.idle_time:
                    self.standing = True
                    self.stand_from = self.last_targets.copy()
                    self.stand_t0 = now
                    self.get_logger().info('idle: holding standing pose')
            else:
                self.idle_since = None
                if self.standing:
                    # resume like a fresh training episode: clock from 0, no previous action
                    self.standing = False
                    self.policy_start = now
                    self.last_action = np.zeros(8)
                    self.last_policy_time = None
                    self.get_logger().info('command received: policy resumed')
            if self.standing:
                a = min((now - self.stand_t0).nanoseconds * 1e-9 / self.settle_time, 1.0)
                self.last_targets = self.stand_from + a * (DEFAULT_Q - self.stand_from)
                self.publish(self.last_targets)
                return
        # a late timer can fire twice on the same sim tick; keep one policy step per control period
        if self.last_policy_time is not None and \
                (now - self.last_policy_time).nanoseconds * 1e-9 < 0.5 / self.rate:
            return
        self.last_policy_time = now
        # gait clock from sim time (not a call counter) so late timer callbacks don't shift the rhythm
        t_policy = (now - self.policy_start).nanoseconds * 1e-9
        phase = (t_policy / self.gait_period) % 1.0
        obs = np.concatenate([
            self.ang_vel,
            self.gravity,
            cmd,
            self.q - DEFAULT_Q,
            self.dq,
            self.last_action,
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
        ]).astype(np.float32)
        action = self.policy(obs)
        self.last_action = action
        self.last_targets = np.clip(DEFAULT_Q + self.action_scale * action, LOWER, UPPER)
        self.publish(self.last_targets)


def main():
    rclpy.init()
    node = RlPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

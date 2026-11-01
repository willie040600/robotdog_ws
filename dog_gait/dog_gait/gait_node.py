"""Trot gait controller for the 8-DOF dog.

Subscribes to /cmd_vel (geometry_msgs/Twist), generates foot trajectories for
a diagonal trot, solves 2-link leg IK and publishes joint position targets to
/dog/<joint>/cmd_pos (std_msgs/Float64), which are bridged to Gazebo's
JointPositionController plugins.

The legs only have pitch joints, so the dog cannot step sideways: forward speed
comes from stride length and yaw comes from a left/right stride difference
(skid-steer style). linear.y is ignored.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

# hip position in base_link (x, y)
LEGS = {
    'FL': (0.1, 0.09),
    'FR': (0.1, -0.09),
    'RL': (-0.1, 0.09),
    'RR': (-0.1, -0.09),
}
# diagonal pairs move together
PHASE_OFFSET = {'FL': 0.0, 'RR': 0.0, 'FR': 0.5, 'RL': 0.5}

THIGH_LIMITS = (-1.5, 1.5)
CALF_LIMITS = (-2.6, -0.6)
# Pose held until the robot appears in Gazebo. The calf can't reach 0 (its limit
# is -0.6), so thigh = -calf/2 keeps each foot right under its hip; the dog lands
# on it and then stands up from there.
READY_Q = (0.3, -0.6)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class GaitNode(Node):

    def __init__(self):
        super().__init__('gait_node')
        p = self.declare_parameter
        self.rate = p('rate', 100.0).value
        self.l1 = p('thigh_length', 0.12).value
        self.l2 = p('calf_length', 0.12).value
        self.body_height = p('body_height', 0.155).value    # hip to foot centre
        self.step_height = p('step_height', 0.04).value
        self.period = p('gait_period', 0.36).value
        self.duty = p('duty_factor', 0.5).value
        self.max_stride = p('max_stride', 0.10).value
        self.max_vx = p('max_vx', 0.3).value
        self.max_wz = p('max_wz', 1.0).value
        self.turn_gain = p('turn_gain', 4.0).value
        # skid turning makes the trot creep forward (roughly with wz^2);
        # cancel it with a backward bias of turn_vx_comp * wz^2
        self.turn_vx_comp = p('turn_vx_comp', 0.35).value
        self.foot_x_offset = p('foot_x_offset', 0.0).value
        self.cmd_timeout = p('cmd_timeout', 0.5).value
        self.cmd_tau = p('cmd_filter_tau', 0.15).value
        self.stand_up_time = p('stand_up_time', 2.0).value

        self.joint_names = []
        for leg in LEGS:
            self.joint_names += [f'{leg}_thigh_joint', f'{leg}_calf_joint']
        self.pubs = {
            j: self.create_publisher(Float64, f'/dog/{j}/cmd_pos', 10)
            for j in self.joint_names
        }

        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)

        self.cmd_vx = 0.0
        self.cmd_wz = 0.0
        self.last_cmd_time = None
        self.vx = 0.0
        self.wz = 0.0
        self.phase = 0.0
        self.amplitude = 0.0        # 0 = standing still, 1 = full gait
        self.spawned = False        # set once Gazebo publishes joint states
        self.start_time = None      # when the stand-up began
        self.last_time = None

        self.stand_q = self.solve_leg(self.foot_x_offset, -self.body_height)
        self.create_timer(1.0 / self.rate, self.update)
        self.get_logger().info(
            f'gait ready: stand thigh={self.stand_q[0]:.3f} calf={self.stand_q[1]:.3f}')

    # ------------------------------------------------------------------ io
    def on_cmd_vel(self, msg):
        self.cmd_vx = clamp(msg.linear.x, -self.max_vx, self.max_vx)
        self.cmd_wz = clamp(msg.angular.z, -self.max_wz, self.max_wz)
        self.last_cmd_time = self.get_clock().now()

    def on_joint_states(self, msg):
        if not self.spawned and all(j in msg.name for j in self.joint_names):
            self.spawned = True
            self.get_logger().info('robot spawned, standing up')

    def publish(self, q):
        for j, v in q.items():
            self.pubs[j].publish(Float64(data=float(v)))

    def publish_all_legs(self, thigh, calf):
        q = {}
        for leg in LEGS:
            q[f'{leg}_thigh_joint'] = thigh
            q[f'{leg}_calf_joint'] = calf
        self.publish(q)

    # ------------------------------------------------------------------ kinematics
    def solve_leg(self, x, z):
        """IK for foot (x forward, z up) in the hip frame -> (thigh, calf).

        Joint convention from the URDF: a link of length L rotated by q about +Y
        points to (-L sin q, -L cos q); the knee bends backwards (calf < 0).
        """
        l1, l2 = self.l1, self.l2
        d = math.hypot(x, z)
        d = clamp(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
        c = clamp((d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        knee = math.acos(c)                                   # planar elbow angle
        psi1 = math.atan2(z, x) - math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
        thigh = -math.pi / 2.0 - psi1
        thigh = math.atan2(math.sin(thigh), math.cos(thigh))
        calf = -knee
        return (clamp(thigh, *THIGH_LIMITS), clamp(calf, *CALF_LIMITS))

    def foot_target(self, leg, phase, stride):
        if phase < self.duty:                                 # stance: push back
            s = phase / self.duty
            x = stride / 2.0 - stride * s
            z = -self.body_height
        else:                                                 # swing: lift and reach forward
            s = (phase - self.duty) / (1.0 - self.duty)
            x = -stride / 2.0 + stride * (s - math.sin(2.0 * math.pi * s) / (2.0 * math.pi))
            z = -self.body_height + self.amplitude * self.step_height * \
                (1.0 - math.cos(2.0 * math.pi * s)) / 2.0
        return x + self.foot_x_offset, z

    # ------------------------------------------------------------------ loop
    def update(self):
        now = self.get_clock().now()
        # 1) hold the ready pose until the robot exists, then stand up smoothly
        if not self.spawned:
            self.publish_all_legs(*READY_Q)
            return
        if self.start_time is None:
            self.start_time = now
            self.last_time = now
        dt = (now - self.last_time).nanoseconds * 1e-9
        t = (now - self.start_time).nanoseconds * 1e-9
        if t < 0.5 + self.stand_up_time:
            a = clamp((t - 0.5) / self.stand_up_time, 0.0, 1.0)   # let it land first
            a = 0.5 - 0.5 * math.cos(math.pi * a)
            self.publish_all_legs(*(r + a * (s - r) for r, s in zip(READY_Q, self.stand_q)))
            self.last_time = now
            return
        if dt <= 0.0:
            return
        dt = min(dt, 0.05)
        self.last_time = now

        # 2) filtered velocity command (zero after timeout)
        vx_cmd, wz_cmd = self.cmd_vx, self.cmd_wz
        if self.last_cmd_time is None or \
                (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout:
            vx_cmd, wz_cmd = 0.0, 0.0
        k = min(1.0, dt / self.cmd_tau)
        self.vx += (vx_cmd - self.vx) * k
        self.wz += (wz_cmd - self.wz) * k

        moving = abs(vx_cmd) > 0.01 or abs(wz_cmd) > 0.02
        settled = abs(self.vx) < 0.005 and abs(self.wz) < 0.01
        if moving:
            self.amplitude = min(1.0, self.amplitude + dt / 0.3)
        elif settled:
            self.amplitude = max(0.0, self.amplitude - dt / 0.3)

        if self.amplitude > 0.0:
            self.phase = (self.phase + dt / self.period) % 1.0
        else:
            self.phase = 0.0

        # 3) per-leg stride -> foot target -> IK
        stance_time = self.period * self.duty
        vx = self.vx - self.turn_vx_comp * self.wz * self.wz
        q = {}
        for leg, (_, hip_y) in LEGS.items():
            v_leg = vx - self.turn_gain * self.wz * hip_y
            stride = clamp(v_leg * stance_time, -self.max_stride, self.max_stride)
            phase = (self.phase + PHASE_OFFSET[leg]) % 1.0
            x, z = self.foot_target(leg, phase, stride)
            thigh, calf = self.solve_leg(x, z)
            q[f'{leg}_thigh_joint'] = thigh
            q[f'{leg}_calf_joint'] = calf
        self.publish(q)


def main():
    rclpy.init()
    node = GaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

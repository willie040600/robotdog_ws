"""Drop lidar returns that hit the floor because the body is tilted.

The 2D lidar sits ~0.2 m above the ground on a walking body that rolls/pitches 4-5 deg (RL gait),
so part of every scan lands on the floor 2-3 m away. slam_toolbox drew those points into the
map (map smearing / drift) and the costmaps marked them as obstacles ("collision ahead").

For each scan, every return is transformed into the odom frame with TF (which includes the body
tilt via base_footprint -> base_link); returns below min_height are replaced by +inf (no echo)
and the scan is republished on scan_filtered.
"""
import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def rotate(q, v):
    """Rotate vectors v (N, 3) by quaternion q = (x, y, z, w)."""
    u = np.array(q[:3])
    w = q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class ScanGroundFilter(Node):

    def __init__(self):
        super().__init__('scan_ground_filter')
        self.fixed_frame = self.declare_parameter('fixed_frame', 'odom').value
        self.min_height = self.declare_parameter('min_height', 0.05).value
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.pub = self.create_publisher(LaserScan, 'scan_filtered', 10)
        self.create_subscription(LaserScan, 'scan', self.on_scan, 10)
        self.dropped = 0
        self.create_timer(10.0, self.report)

    def report(self):
        if self.dropped:
            self.get_logger().info(f'removed {self.dropped} floor returns in the last 10 s')
        self.dropped = 0

    def on_scan(self, msg):
        try:
            tf = self.buf.lookup_transform(self.fixed_frame, msg.header.frame_id,
                                           Time.from_msg(msg.header.stamp), Duration(seconds=0.05))
        except TransformException:
            try:
                tf = self.buf.lookup_transform(self.fixed_frame, msg.header.frame_id, Time())
            except TransformException:
                return
        r = np.asarray(msg.ranges, dtype=np.float64)
        a = msg.angle_min + msg.angle_increment * np.arange(len(r))
        finite = np.isfinite(r)
        pts = np.stack([np.where(finite, r, 0.0) * np.cos(a), np.where(finite, r, 0.0) * np.sin(a),
                        np.zeros_like(r)], axis=1)
        q = tf.transform.rotation
        z = rotate((q.x, q.y, q.z, q.w), pts)[:, 2] + tf.transform.translation.z
        floor = finite & (r > msg.range_min) & (z < self.min_height)
        self.dropped += int(floor.sum())
        out = LaserScan()
        out.header = msg.header
        out.angle_min, out.angle_max, out.angle_increment = msg.angle_min, msg.angle_max, msg.angle_increment
        out.time_increment, out.scan_time = msg.time_increment, msg.scan_time
        out.range_min, out.range_max = msg.range_min, msg.range_max
        out.ranges = np.where(floor, math.inf, r).astype(np.float32).tolist()
        out.intensities = msg.intensities
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanGroundFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

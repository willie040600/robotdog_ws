"""Split the simulator's 3D odometry into a planar base_footprint and the body's tilt (REP-105/120).

Gazebo's OdometryPublisher gives odom -> base_link in full 3D, including the body bobbing and
rolling/pitching while the dog walks. slam_toolbox / AMCL / Nav2 assume a planar base frame; fed
base_link they bake the instantaneous tilt and height into map -> odom, so every SLAM update made
the map jump. This node publishes instead:

    odom -> base_footprint   x, y, yaw          (TF + /odom, planar twist)
    base_footprint -> base_link   z, roll, pitch   (TF)

and Nav2 / SLAM / AMCL use base_footprint as the robot base frame.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def quat_to_rpy(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class OdomFootprintNode(Node):

    def __init__(self):
        super().__init__('odom_footprint_node')
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.footprint_frame = self.declare_parameter('footprint_frame', 'base_footprint').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, 'odom', 10)
        self.create_subscription(Odometry, 'odom_3d', self.on_odom, 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy(o.x, o.y, o.z, o.w)
        stamp = msg.header.stamp

        # odom -> base_footprint (planar)
        fp = TransformStamped()
        fp.header.stamp = stamp
        fp.header.frame_id = self.odom_frame
        fp.child_frame_id = self.footprint_frame
        fp.transform.translation.x = p.x
        fp.transform.translation.y = p.y
        qx, qy, qz, qw = rpy_to_quat(0.0, 0.0, yaw)
        fp.transform.rotation.x, fp.transform.rotation.y = qx, qy
        fp.transform.rotation.z, fp.transform.rotation.w = qz, qw

        # base_footprint -> base_link (height and tilt)
        bl = TransformStamped()
        bl.header.stamp = stamp
        bl.header.frame_id = self.footprint_frame
        bl.child_frame_id = self.base_frame
        bl.transform.translation.z = p.z
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, 0.0)
        bl.transform.rotation.x, bl.transform.rotation.y = qx, qy
        bl.transform.rotation.z, bl.transform.rotation.w = qz, qw
        self.tf.sendTransform([fp, bl])

        # planar odometry for Nav2 (twist is already in the body frame; keep vx, vy, wz)
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.footprint_frame
        out.pose.pose.position.x = p.x
        out.pose.pose.position.y = p.y
        out.pose.pose.orientation = fp.transform.rotation
        out.pose.covariance = msg.pose.covariance
        out.twist.twist.linear.x = msg.twist.twist.linear.x
        out.twist.twist.linear.y = msg.twist.twist.linear.y
        out.twist.twist.angular.z = msg.twist.twist.angular.z
        out.twist.covariance = msg.twist.covariance
        self.pub.publish(out)


def main():
    rclpy.init()
    node = OdomFootprintNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

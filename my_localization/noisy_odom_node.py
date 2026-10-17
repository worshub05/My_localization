"""Add realistic error to Gazebo's wheel odometry.

Gazebo's diff-drive odometry is almost perfect, so this node models a
real robot's odometry error using the odometry motion model
(Thrun et al., Probabilistic Robotics, ch. 5.4).

Each odometry step is split into (rot1, trans, rot2). Two error types:
  * systematic error: scale factors (e.g. wrong wheel radius or
    wheel separation calibration) -> causes steady drift
  * random error: zero-mean Gaussian noise whose std grows with
    the size of the motion

The corrupted increments are integrated into a dead-reckoning pose,
published on /noisy_odom. The particle filter only ever sees this topic.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NoisyOdomNode(Node):

    def __init__(self):
        super().__init__('noisy_odom_node')
        # Random noise coefficients (std = a * size of motion)
        self.declare_parameter('alpha1', 0.05)  # rot   noise from rot
        self.declare_parameter('alpha2', 0.02)  # rot   noise from trans
        self.declare_parameter('alpha3', 0.05)  # trans noise from trans
        self.declare_parameter('alpha4', 0.01)  # trans noise from rot
        # Systematic calibration error
        self.declare_parameter('trans_scale', 1.02)  # +2 % distance
        self.declare_parameter('rot_scale', 1.04)    # +4 % rotation
        self.declare_parameter('seed', 42)

        p = self.get_parameter
        self.a1, self.a2 = p('alpha1').value, p('alpha2').value
        self.a3, self.a4 = p('alpha3').value, p('alpha4').value
        self.trans_scale = p('trans_scale').value
        self.rot_scale = p('rot_scale').value
        self.rng = np.random.default_rng(p('seed').value)

        self.prev = None                 # previous clean odom (x, y, th)
        self.pose = [0.0, 0.0, 0.0]      # noisy dead-reckoning pose

        self.sub = self.create_subscription(Odometry, 'odom', self.cb, 50)
        self.pub = self.create_publisher(Odometry, 'noisy_odom', 50)

    def cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        th = yaw_from_quat(msg.pose.pose.orientation)

        if self.prev is None:
            self.prev = (x, y, th)
            self.pose = [x, y, th]
            self.publish(msg)
            return

        px, py, pth = self.prev
        self.prev = (x, y, th)

        # 1. Decompose the motion into rot1 -> trans -> rot2
        dx, dy = x - px, y - py
        trans = math.hypot(dx, dy)
        if trans < 1e-3:                 # turning on the spot
            rot1 = 0.0
        else:
            rot1 = wrap(math.atan2(dy, dx) - pth)
            if abs(rot1) > math.pi / 2:  # driving backwards
                rot1 = wrap(rot1 - math.pi)
                trans = -trans
        rot2 = wrap(th - pth - rot1)

        if abs(trans) < 1e-6 and abs(rot1) < 1e-6 and abs(rot2) < 1e-6:
            self.publish(msg)
            return

        # 2. Corrupt it: systematic scale error + random Gaussian noise
        at, ar1, ar2 = abs(trans), abs(rot1), abs(rot2)
        rot1_n = rot1 * self.rot_scale + self.rng.normal(
            0.0, self.a1 * ar1 + self.a2 * at)
        trans_n = trans * self.trans_scale + self.rng.normal(
            0.0, self.a3 * at + self.a4 * (ar1 + ar2))
        rot2_n = rot2 * self.rot_scale + self.rng.normal(
            0.0, self.a1 * ar2 + self.a2 * at)

        # 3. Integrate into the dead-reckoning pose
        nx, ny, nth = self.pose
        nx += trans_n * math.cos(nth + rot1_n)
        ny += trans_n * math.sin(nth + rot1_n)
        nth = wrap(nth + rot1_n + rot2_n)
        self.pose = [nx, ny, nth]
        self.publish(msg)

    def publish(self, src):
        out = Odometry()
        out.header.stamp = src.header.stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_link'
        out.pose.pose.position.x = self.pose[0]
        out.pose.pose.position.y = self.pose[1]
        out.pose.pose.orientation.z = math.sin(self.pose[2] / 2.0)
        out.pose.pose.orientation.w = math.cos(self.pose[2] / 2.0)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = NoisyOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

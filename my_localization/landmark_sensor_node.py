"""Simulated range-bearing landmark sensor with Gaussian noise.

Models a sensor (e.g. a camera or lidar landmark detector) that sees the
red pillars in the world. For every landmark within max_range it reports
    [id, range (m), bearing (rad)]
with zero-mean Gaussian noise added to range and bearing.

The simulator's true pose is used here only to *generate* the simulated
measurement (it plays the role of the real world). The particle filter
never receives the true pose, only these noisy readings.

Published on /landmark_obs as a flat Float64MultiArray:
    [id0, r0, b0, id1, r1, b1, ...]
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Landmark (x, y) positions, matching worlds/landmark_world.sdf
LANDMARKS = [(-2.0, -2.0), (3.0, -2.5), (8.0, -2.0), (8.0, 2.0),
             (8.0, 6.0), (3.0, 6.5), (-2.0, 6.0), (-2.0, 2.0)]


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LandmarkSensorNode(Node):

    def __init__(self):
        super().__init__('landmark_sensor_node')
        self.declare_parameter('rate', 5.0)              # Hz
        self.declare_parameter('max_range', 6.0)         # m
        self.declare_parameter('range_std', 0.10)        # m
        self.declare_parameter('bearing_std', 0.05)      # rad (~3 deg)
        self.declare_parameter('seed', 7)

        p = self.get_parameter
        self.max_range = p('max_range').value
        self.range_std = p('range_std').value
        self.bearing_std = p('bearing_std').value
        self.rng = np.random.default_rng(p('seed').value)

        self.true_pose = None
        self.sub = self.create_subscription(
            Odometry, 'ground_truth', self.gt_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, 'landmark_obs', 10)
        self.timer = self.create_timer(1.0 / p('rate').value, self.tick)

    def gt_cb(self, msg):
        self.true_pose = (msg.pose.pose.position.x,
                          msg.pose.pose.position.y,
                          yaw_from_quat(msg.pose.pose.orientation))

    def tick(self):
        if self.true_pose is None:
            return
        x, y, th = self.true_pose
        data = []
        for i, (lx, ly) in enumerate(LANDMARKS):
            r = math.hypot(lx - x, ly - y)
            if r > self.max_range:
                continue
            b = wrap(math.atan2(ly - y, lx - x) - th)
            r_noisy = r + self.rng.normal(0.0, self.range_std)
            b_noisy = wrap(b + self.rng.normal(0.0, self.bearing_std))
            data += [float(i), r_noisy, b_noisy]
        msg = Float64MultiArray()
        msg.data = data
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

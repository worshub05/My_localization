"""Log ground truth, dead reckoning and particle filter poses to CSV.

At a fixed rate, records the latest pose from each topic on one row, so
all three are compared at the same simulation time:
    t, gt_x, gt_y, gt_th, dr_x, dr_y, dr_th, pf_x, pf_y, pf_th

Ground truth is ONLY logged here for evaluation. It is never fed back
into the localization.
"""
import csv
import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pose_of(msg):
    p = msg.pose.pose
    return (p.position.x, p.position.y, yaw_from_quat(p.orientation))


class LoggerNode(Node):

    def __init__(self):
        super().__init__('logger_node')
        self.declare_parameter(
            'output_file',
            '~/loc_ws/src/my_localization/results/trajectory_log.csv')
        self.declare_parameter('rate', 10.0)   # Hz

        path = os.path.expanduser(self.get_parameter('output_file').value)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['t', 'gt_x', 'gt_y', 'gt_th',
                              'dr_x', 'dr_y', 'dr_th',
                              'pf_x', 'pf_y', 'pf_th'])
        self.latest = {'gt': None, 'dr': None, 'pf': None}
        self.t0 = None
        self.rows = 0

        self.create_subscription(
            Odometry, 'ground_truth', lambda m: self.store('gt', m), 10)
        self.create_subscription(
            Odometry, 'noisy_odom', lambda m: self.store('dr', m), 10)
        self.create_subscription(
            Odometry, 'pf_pose', lambda m: self.store('pf', m), 10)
        self.create_timer(1.0 / self.get_parameter('rate').value, self.tick)
        self.get_logger().info(f'Logging to {path}')

    def store(self, key, msg):
        self.latest[key] = pose_of(msg)

    def tick(self):
        if any(v is None for v in self.latest.values()):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        row = [round(now - self.t0, 3)]
        for key in ('gt', 'dr', 'pf'):
            row += [round(v, 5) for v in self.latest[key]]
        self.writer.writerow(row)
        self.file.flush()
        self.rows += 1

    def close(self):
        self.file.close()
        self.get_logger().info(f'Saved {self.rows} rows')


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

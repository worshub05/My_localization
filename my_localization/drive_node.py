"""Drive the robot around a rectangular loop with timed open-loop commands."""
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class DriveNode(Node):

    def __init__(self):
        super().__init__('drive_node')
        self.declare_parameter('linear_speed', 0.4)      # m/s
        self.declare_parameter('angular_speed', 0.5)     # rad/s
        self.declare_parameter('side_lengths', [6.0, 4.0, 6.0, 4.0])  # m
        self.declare_parameter('start_delay', 3.0)       # s, let sim settle

        v = self.get_parameter('linear_speed').value
        w = self.get_parameter('angular_speed').value
        sides = self.get_parameter('side_lengths').value

        # Plan = list of (duration [s], linear v, angular w)
        self.plan = [(self.get_parameter('start_delay').value, 0.0, 0.0)]
        for i, length in enumerate(sides):
            self.plan.append((length / v, v, 0.0))            # straight
            if i < len(sides) - 1:
                self.plan.append(((math.pi / 2) / w, 0.0, w))  # 90 deg left

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.start_time = None
        self.done = False
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info(f'Drive plan has {len(self.plan)} segments')

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now == 0.0:          # simulation clock not received yet
            return
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time

        cmd = Twist()
        t_end = 0.0
        for duration, v, w in self.plan:
            t_end += duration
            if elapsed < t_end:
                cmd.linear.x = v
                cmd.angular.z = w
                break
        else:
            if not self.done:
                self.get_logger().info('Trajectory complete, robot stopped')
                self.done = True
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

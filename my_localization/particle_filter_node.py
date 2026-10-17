"""Particle filter (Monte Carlo) localization.

Inputs (the ONLY data used to estimate the pose):
  /noisy_odom    nav_msgs/Odometry          noisy dead-reckoning pose
  /landmark_obs  std_msgs/Float64MultiArray [id, range, bearing, ...]
plus the known map of landmark positions.
Ground truth is never used here.

Output:
  /pf_pose       nav_msgs/Odometry          weighted-mean pose estimate

Algorithm (Thrun et al., Probabilistic Robotics, ch. 4 and 8):
  1. Initialise N particles around the known start pose.
  2. Predict: on each odometry message, apply the odometry increment
     (rot1, trans, rot2) to every particle with sampled noise.
  3. Update: on each landmark reading, weight every particle by the
     Gaussian likelihood of the measured range and bearing.
  4. Resample (low-variance) when the effective sample size drops
     below N/2, then add a small jitter to keep particle diversity.
  5. Estimate: weighted mean of x, y and circular mean of heading.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Known map: landmark (x, y) positions, matching worlds/landmark_world.sdf
LANDMARKS = np.array([(-2.0, -2.0), (3.0, -2.5), (8.0, -2.0), (8.0, 2.0),
                      (8.0, 6.0), (3.0, 6.5), (-2.0, 6.0), (-2.0, 2.0)])


def wrap(angle):
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ParticleFilter:
    """Pure-numpy particle filter, independent of ROS."""

    def __init__(self, n, init_pose, init_std, alphas, range_std,
                 bearing_std, jitter, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.range_std = range_std
        self.bearing_std = bearing_std
        self.jitter = jitter                      # (xy std, theta std)
        self.p = np.empty((n, 3))                 # particles: x, y, theta
        self.p[:, 0] = self.rng.normal(init_pose[0], init_std[0], n)
        self.p[:, 1] = self.rng.normal(init_pose[1], init_std[0], n)
        self.p[:, 2] = wrap(self.rng.normal(init_pose[2], init_std[1], n))
        self.w = np.full(n, 1.0 / n)              # weights

    def predict(self, rot1, trans, rot2):
        """Sample from the odometry motion model for every particle."""
        n = self.n
        s_r1 = self.a1 * abs(rot1) + self.a2 * abs(trans)
        s_t = self.a3 * abs(trans) + self.a4 * (abs(rot1) + abs(rot2))
        s_r2 = self.a1 * abs(rot2) + self.a2 * abs(trans)
        r1 = rot1 + self.rng.normal(0.0, s_r1, n)
        t = trans + self.rng.normal(0.0, s_t, n)
        r2 = rot2 + self.rng.normal(0.0, s_r2, n)
        self.p[:, 0] += t * np.cos(self.p[:, 2] + r1)
        self.p[:, 1] += t * np.sin(self.p[:, 2] + r1)
        self.p[:, 2] = wrap(self.p[:, 2] + r1 + r2)

    def update(self, obs):
        """Weight particles by landmark likelihood. obs = [(id, r, b), ...]"""
        if not obs:
            return
        log_w = np.log(self.w + 1e-300)
        for lid, r, b in obs:
            lx, ly = LANDMARKS[int(lid)]
            dx = lx - self.p[:, 0]
            dy = ly - self.p[:, 1]
            r_hat = np.hypot(dx, dy)
            b_hat = wrap(np.arctan2(dy, dx) - self.p[:, 2])
            log_w += -0.5 * ((r - r_hat) / self.range_std) ** 2
            log_w += -0.5 * (wrap(b - b_hat) / self.bearing_std) ** 2
        log_w -= log_w.max()                      # numerical stability
        self.w = np.exp(log_w)
        self.w /= self.w.sum()
        if self.n_eff() < self.n / 2.0:
            self.resample()

    def n_eff(self):
        return 1.0 / np.sum(self.w ** 2)

    def resample(self):
        """Low-variance (systematic) resampling + small jitter."""
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(self.w), positions)
        idx = np.minimum(idx, self.n - 1)
        self.p = self.p[idx]
        self.p[:, 0:2] += self.rng.normal(0.0, self.jitter[0], (self.n, 2))
        self.p[:, 2] = wrap(
            self.p[:, 2] + self.rng.normal(0.0, self.jitter[1], self.n))
        self.w = np.full(self.n, 1.0 / self.n)

    def estimate(self):
        """Weighted mean pose and x-y covariance."""
        x = np.sum(self.w * self.p[:, 0])
        y = np.sum(self.w * self.p[:, 1])
        th = math.atan2(np.sum(self.w * np.sin(self.p[:, 2])),
                        np.sum(self.w * np.cos(self.p[:, 2])))
        cov = np.cov(self.p[:, 0:2].T, aweights=self.w)
        return x, y, th, cov


class ParticleFilterNode(Node):

    def __init__(self):
        super().__init__('particle_filter_node')
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('init_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('init_std', [0.1, 0.05])       # m, rad
        # Filter motion noise: larger than the true random noise so the
        # particles also cover the unknown systematic odometry error.
        self.declare_parameter('alphas', [0.1, 0.05, 0.1, 0.05])
        # Filter sensor noise: slightly larger than the true sensor noise
        self.declare_parameter('range_std', 0.2)               # m
        self.declare_parameter('bearing_std', 0.1)             # rad
        self.declare_parameter('jitter', [0.01, 0.005])        # m, rad
        self.declare_parameter('seed', 0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.pf = ParticleFilter(
            n=p('num_particles'), init_pose=p('init_pose'),
            init_std=p('init_std'), alphas=p('alphas'),
            range_std=p('range_std'), bearing_std=p('bearing_std'),
            jitter=p('jitter'), seed=p('seed'))
        self.prev_odom = None

        self.create_subscription(Odometry, 'noisy_odom', self.odom_cb, 50)
        self.create_subscription(
            Float64MultiArray, 'landmark_obs', self.obs_cb, 10)
        self.pub = self.create_publisher(Odometry, 'pf_pose', 10)
        self.get_logger().info(
            f'Particle filter started with {p("num_particles")} particles')

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        th = yaw_from_quat(msg.pose.pose.orientation)
        if self.prev_odom is not None:
            px, py, pth = self.prev_odom
            dx, dy = x - px, y - py
            trans = math.hypot(dx, dy)
            if trans < 1e-3:
                rot1 = 0.0
            else:
                rot1 = float(wrap(math.atan2(dy, dx) - pth))
                if abs(rot1) > math.pi / 2:          # driving backwards
                    rot1 = float(wrap(rot1 - math.pi))
                    trans = -trans
            rot2 = float(wrap(th - pth - rot1))
            if abs(trans) > 1e-6 or abs(rot1) > 1e-6 or abs(rot2) > 1e-6:
                self.pf.predict(rot1, trans, rot2)
        self.prev_odom = (x, y, th)
        self.publish(msg.header.stamp)

    def obs_cb(self, msg):
        d = msg.data
        obs = [(d[i], d[i + 1], d[i + 2]) for i in range(0, len(d), 3)]
        self.pf.update(obs)

    def publish(self, stamp):
        x, y, th, cov = self.pf.estimate()
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = 'world'
        out.child_frame_id = 'base_link'
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation.z = math.sin(th / 2.0)
        out.pose.pose.orientation.w = math.cos(th / 2.0)
        c = [0.0] * 36
        c[0], c[1], c[6], c[7] = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
        out.pose.covariance = [float(v) for v in c]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

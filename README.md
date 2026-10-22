# Particle Filter Localization in Gazebo (ROS 2)

Intelligent Robotics, Assignment 1. A differential-drive robot drives a
6 m x 4 m loop in Gazebo. Its pose is estimated with a particle filter
that uses only noisy odometry and noisy range-bearing observations of
8 known landmarks. Ground truth is used only for evaluation.

## Requirements
- Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress
- `ros_gz_sim`, `ros_gz_bridge`, `python3-numpy`, `python3-matplotlib`

## Build
```bash
cd ~/loc_ws
colcon build --packages-select my_localization
source install/setup.bash
```

## Run
```bash
ros2 launch my_localization sim.launch.py
```
Wait for `Trajectory complete, robot stopped` (about 70 s), then press Ctrl+C.

## Plot results
```bash
cd ~/loc_ws/src/my_localization
python3 scripts/plot_results.py results/trajectory_log.csv
```

## Nodes
| Node | Role |
| --- | --- |
| `drive_node` | Drives the rectangular trajectory (`/cmd_vel`) |
| `noisy_odom_node` | Adds systematic + random error to odometry (`/noisy_odom`) |
| `landmark_sensor_node` | Noisy range-bearing landmark readings (`/landmark_obs`) |
| `particle_filter_node` | 500-particle Monte Carlo localization (`/pf_pose`) |
| `logger_node` | Logs ground truth, dead reckoning and PF poses to CSV |

## Results (driving period, 66.4 s)
| Method | Position RMSE (m) | Final error (m) |
| --- | --- | --- |
| Dead reckoning | 0.466 | 0.851 |
| Particle filter | 0.061 | 0.033 |

Plots: `results/trajectory_plot.png`, `results/error_plot.png`.
Report: `report/report.pdf`.

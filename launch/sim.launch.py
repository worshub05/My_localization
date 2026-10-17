import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('my_localization')
    world = os.path.join(pkg, 'worlds', 'landmark_world.sdf')

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch',
            'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items())

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/ground_truth@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        ])

    sim_time = {'use_sim_time': True}

    noisy_odom = Node(
        package='my_localization', executable='noisy_odom_node',
        output='screen', parameters=[sim_time])

    drive = Node(
        package='my_localization', executable='drive_node',
        output='screen', parameters=[sim_time])

    return LaunchDescription([gz, bridge, noisy_odom, drive])

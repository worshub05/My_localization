import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'my_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shub_2605',
    maintainer_email='worshub05@gmail.com',
    description='Particle filter localization of a diff-drive robot in Gazebo',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'drive_node = my_localization.drive_node:main',
            'noisy_odom_node = my_localization.noisy_odom_node:main',
            'landmark_sensor_node = '
            'my_localization.landmark_sensor_node:main',
            'particle_filter_node = '
            'my_localization.particle_filter_node:main',
        ],
    },
)

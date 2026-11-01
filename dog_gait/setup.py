from glob import glob

from setuptools import setup

package_name = 'dog_gait'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/policies', glob('policies/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='willie',
    maintainer_email='willie040600@gmail.com',
    description='Trot gait + leg IK for the 8-DOF robot dog',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gait_node = dog_gait.gait_node:main',
            'rl_policy_node = dog_gait.rl_policy_node:main',
            'odom_footprint_node = dog_gait.odom_footprint_node:main',
            'scan_ground_filter_node = dog_gait.scan_ground_filter_node:main',
        ],
    },
)

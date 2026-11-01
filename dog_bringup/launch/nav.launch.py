"""Simulation + Nav2.

  slam:=True  (default)  slam_toolbox builds the map while navigating
  slam:=False map:=<yaml> map_server + AMCL on a saved map
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('dog_bringup')
    nav2_bringup = get_package_share_directory('nav2_bringup')

    slam = LaunchConfiguration('slam')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    # scoped so the rviz:=false passed to sim.launch.py doesn't leak back and
    # disable this file's own RViz
    sim = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'sim.launch.py')),
        launch_arguments={
            'gui': gui, 'rviz': 'false', 'controller': LaunchConfiguration('controller'),
            'rl_policy': LaunchConfiguration('rl_policy'),
            'rl_stand_when_idle': LaunchConfiguration('rl_stand_when_idle'),
        }.items(),
    )])

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'slam': slam,
            'map': map_yaml,
            'params_file': params_file,
            'use_sim_time': 'True',
            'autostart': 'True',
            'use_composition': 'False',
        }.items(),
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', os.path.join(bringup, 'rviz', 'dog.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('slam', default_value='True'),
        DeclareLaunchArgument('map', default_value=os.path.join(bringup, 'maps', 'dog_world.yaml')),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(bringup, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('controller', default_value='gait', description="'gait' or 'rl'"),
        DeclareLaunchArgument('rl_policy', default_value='dog_flat_v7.npz'),
        DeclareLaunchArgument('rl_stand_when_idle', default_value='true'),
        sim,
        # give the dog time to spawn and stand up before Nav2 starts
        TimerAction(period=6.0, actions=[nav2]),
        rviz_node,
    ])

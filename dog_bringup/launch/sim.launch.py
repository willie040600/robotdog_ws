"""Gazebo Harmonic simulation of the dog: world, robot, bridge, state publisher, gait."""
import os
import re

import xacro
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Passive joint damping used in simulation. The joint PID controllers (dog_sim.urdf.xacro) run
# with d_gain = 0, so torque = kp * (target - q) - JOINT_DAMPING * qdot: an ideal PD like the
# Isaac Lab training (kp 40, kd 0.6). A PID d-term on the *error* would kick the joint every time
# the 50 Hz policy steps its target.
JOINT_DAMPING = 0.6


def generate_launch_description():
    bringup = get_package_share_directory('dog_bringup')
    xacro_file = os.path.join(bringup, 'urdf', 'dog_sim.urdf.xacro')
    default_world = os.path.join(bringup, 'worlds', 'dog_world.sdf')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    controller = LaunchConfiguration('controller')

    robot_description = xacro.process_file(xacro_file).toxml()
    robot_description = re.sub(r'(<dynamics damping=")[0-9.eE+-]+(")',
                               rf'\g<1>{JOINT_DAMPING}\g<2>', robot_description)

    # lets Gazebo resolve package://dog_description/meshes/...
    resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_prefix('dog_description'), 'share'))

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            # -r: start running; -s: server only (no GUI window)
            'gz_args': PythonExpression([
                "'-r -v 2 ", world, "' + ('' if '", gui, "' == 'true' else ' -s')"]),
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # spawned a little later so the bridge and gait node are already publishing
    # the ready pose when the joint controllers come up
    spawn = TimerAction(period=3.0, actions=[Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description', '-name', 'dog',
                   '-x', '0', '-y', '0', '-z', '0.26'],
    )])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        parameters=[{
            'config_file': os.path.join(bringup, 'config', 'gz_bridge.yaml'),
            'use_sim_time': True,
        }],
    )

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    # odom -> base_footprint -> base_link TF and planar /odom from the simulator's 3D odometry
    odom_footprint = Node(
        package='dog_gait', executable='odom_footprint_node', output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[('odom_3d', '/dog/odom_3d')],
    )

    # /scan -> /scan_filtered without the floor returns of a tilted body (used by SLAM/AMCL/Nav2)
    scan_filter = Node(
        package='dog_gait', executable='scan_ground_filter_node', output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # controller:=gait -> hand-written trot (dog_gait/gait_node)
    # controller:=rl   -> Isaac Lab trained policy (dog_gait/rl_policy_node)
    gait = Node(
        package='dog_gait', executable='gait_node', output='screen',
        parameters=[os.path.join(bringup, 'config', 'gait.yaml'), {'use_sim_time': True}],
        condition=IfCondition(PythonExpression(["'", controller, "' == 'gait'"])),
    )
    rl_policy = Node(
        package='dog_gait', executable='rl_policy_node', output='screen',
        parameters=[{
            'use_sim_time': True,
            'policy_path': PathJoinSubstitution(
                [FindPackageShare('dog_gait'), 'policies', LaunchConfiguration('rl_policy')]),
            'stand_when_idle': ParameterValue(LaunchConfiguration('rl_stand_when_idle'), value_type=bool),
        }],
        condition=IfCondition(PythonExpression(["'", controller, "' == 'rl'"])),
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', os.path.join(bringup, 'rviz', 'dog.rviz'), '-f', 'odom'],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('controller', default_value='gait', description="'gait' or 'rl'"),
        DeclareLaunchArgument('rl_policy', default_value='dog_flat_v7.npz',
                              description='policy file in dog_gait/policies (controller:=rl)'),
        DeclareLaunchArgument('rl_stand_when_idle', default_value='true',
                              description='hold the standing pose when the command is zero (controller:=rl)'),
        resource_path,
        gz_sim,
        bridge,
        rsp,
        odom_footprint,
        scan_filter,
        spawn,
        gait,
        rl_policy,
        rviz_node,
    ])

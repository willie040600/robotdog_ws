"""Isaac Lab articulation config for the 8-DOF robot dog (dog_description/urdf/dog.urdf).

The USD was generated with:
    uv run --extra isaacsim python scripts/tools/convert_urdf.py \
        ~/robotdog_ws/dog_description/urdf/dog.urdf ~/robotdog_ws/dog_rl/assets/dog/dog.usd \
        --ros_package_path dog_description ~/robotdog_ws/dog_description \
        --merge_joints --joint_stiffness 40.0 --joint_damping 0.6
(the converter writes a folder; it was renamed to assets/dog/usd).

Actuator gains and limits match the Gazebo simulation (dog_bringup/urdf/dog_sim.urdf.xacro:
JointPositionController p_gain 40, d_gain 0.6, cmd_max 5) so a policy trained here can be
run in Gazebo with the same PD position targets.
"""
import os

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

DOG_USD_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "dog", "usd", "dog", "dog.usda")

# Standing pose used by the Gazebo gait node (hip-to-foot 0.155 m -> body ~0.17 m above ground).
DOG_STAND_THIGH = 0.869
DOG_STAND_CALF = -1.737
DOG_STAND_HEIGHT = 0.17

DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.abspath(DOG_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DOG_STAND_HEIGHT + 0.02),
        joint_pos={
            ".*_thigh_joint": DOG_STAND_THIGH,
            ".*_calf_joint": DOG_STAND_CALF,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Implicit PD: the leg links are very light (~7e-4 kg*m^2 about the knee), so an explicit
        # PD with d=0.6 would be numerically unstable at the 200 Hz physics rate.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint", ".*_calf_joint"],
            stiffness=40.0,
            damping=0.6,
            joint_effort_limit=5.0,
            actuator_effort_limit=5.0,
            actuator_velocity_limit=10.0,
        ),
    },
)

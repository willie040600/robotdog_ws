"""Velocity-tracking locomotion task for the 8-DOF dog on flat ground.

Built on Isaac Lab's shared velocity task (isaaclab_tasks.core.velocity), following the Unitree Go2
flat config, with changes for this robot:
  - the legs only pitch, so commands are forward speed + yaw rate only (lin_vel_y = 0), matching
    what Nav2 sends in Gazebo;
  - the policy sees only what the robot can measure (IMU angular velocity + gravity direction,
    joint states, last action, command) -- no base linear velocity, no height scan;
  - smaller speeds, heights, pushes and CoM offsets for a 3 kg, 26 cm long robot;
  - PD gains / friction randomized around the Gazebo values for sim-to-sim transfer.
Policy runs at 50 Hz (physics 200 Hz, decimation 4).
"""
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace

from isaaclab_tasks.contrib.velocity.config.spot import mdp as spot_mdp
from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    EventsCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

from dog_rl.dog_cfg import DOG_CFG
from dog_rl.tasks import mdp as dog_mdp
from dog_rl.tasks.actions import DelayedJointPositionActionCfg
from dog_rl.tasks.commands import DogVelocityCommandCfg

STAND_HEIGHT = 0.166  # measured base height while standing (scripts/stand_test.py)
GAIT_PERIOD = 0.4  # s per trot cycle (2.5 steps/s per foot pair); the Gazebo gait uses 0.36 s


@configclass
class DogObservations(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # gait clock (sin, cos), appended last -> obs layout: ang_vel, gravity, cmd, q, dq, last action, clock
        gait_phase = ObsTerm(func=dog_mdp.gait_phase, params={"period": GAIT_PERIOD})

    policy: PolicyCfg = PolicyCfg()


@configclass
class DogRewards(RewardsCfg):
    """Shared velocity rewards plus terms that shape a proper trot.

    v1 of this task (rewards above only) learned to shuffle with tiny steps and some envs walked
    on their knees, so v2 adds diagonal-pair gait sync, foot clearance and foot-slip terms
    (adapted from Isaac Lab's Spot task) and a knee termination (see DogTerminations).
    v2 then trotted with very fast tiny steps, so v3 adds a gait clock observation with a
    fixed-period contact schedule reward and stronger smoothness penalties.
    """

    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-60.0, params={"target_height": STAND_HEIGHT})
    # FL+RR and FR+RL touch down / lift off together, the two pairs alternate
    trot_gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=1.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("FL_calf", "RR_calf"), ("FR_calf", "RL_calf")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # swinging feet should lift ~3.5 cm (the Gazebo gait uses 4 cm)
    foot_clearance = RewTerm(
        func=dog_mdp.foot_clearance,
        weight=0.75,
        params={"asset_cfg": dog_mdp.calf_cfg("robot"), "target_height": 0.035, "std": 0.005, "tanh_mult": 2.0},
    )
    # follow the gait clock: fixes the step frequency to 1 / GAIT_PERIOD
    contact_schedule = RewTerm(
        func=dog_mdp.contact_schedule,
        weight=1.0,
        params={
            "sensor_cfg": dog_mdp.calf_cfg("contact_forces"),
            "period": GAIT_PERIOD,
            "command_name": "base_velocity",
            "threshold": 1.0,
        },
    )
    # v5: with a zero command, stand still (v1-v4 kept trotting and crept backwards ~0.06 m/s)
    stand_still_joints = RewTerm(
        func=dog_mdp.stand_still_joints,
        weight=-0.5,
        params={"command_name": "base_velocity", "threshold": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    stand_still_drift = RewTerm(
        func=dog_mdp.stand_still_drift,
        weight=-2.0,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )
    # v6: turning in place must not translate (v5 crept into obstacles under Nav2)
    turn_in_place_drift = RewTerm(
        func=dog_mdp.turn_in_place_drift,
        weight=-2.0,
        params={"command_name": "base_velocity", "vx_threshold": 0.05, "wz_threshold": 0.1},
    )
    # slow, smooth leg motion
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-4)
    # feet on the ground must not drag
    foot_slip = RewTerm(
        func=dog_mdp.foot_slip,
        weight=-0.5,
        params={
            "asset_cfg": dog_mdp.calf_cfg("robot"),
            "sensor_cfg": dog_mdp.calf_cfg("contact_forces"),
            "threshold": 1.0,
        },
    )


@configclass
class DogTerminations(TerminationsCfg):
    """Shared terminations plus: kneeling ends the episode."""

    knee_contact = DoneTerm(
        func=dog_mdp.knee_too_low, params={"asset_cfg": dog_mdp.calf_cfg("robot"), "min_height": 0.035}
    )


@configclass
class DogEvents(EventsCfg):
    """Shared events plus PD gain randomization around the Gazebo gains (kp 40, kd 0.6)."""

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.7, 1.3),   # v5: wider (v4 0.8-1.2 / 0.7-1.3)
            "damping_distribution_params": (0.5, 1.5),
        },
    )
    # v5: unmodelled joint friction and rotor inertia (Gazebo/DART and real servos differ from PhysX)
    joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.05),
            "armature_distribution_params": (0.0, 0.0005),
            "operation": "add",
            "distribution": "uniform",
        },
    )


@configclass
class DogFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: DogObservations = DogObservations()
    rewards: DogRewards = DogRewards()
    events: DogEvents = DogEvents()
    terminations: DogTerminations = DogTerminations()

    def __post_init__(self):
        super().__post_init__()

        # --- physics: Isaac Sim PhysX (validated with scripts/stand_test.py). The "physx" auto preset
        # would pick the kit-less ovphysx package, which the isaacsim extra doesn't install.
        self.sim.physics.default = self.sim.physics.isaacsim_physx

        # --- scene: flat plane and our robot. The shared contact sensor ("Robot/[^/]*") searches the
        # whole robot subtree for contact-reporting bodies, so the nested USD links are found as-is.
        self.scene.env_spacing = 1.0
        self.scene.robot = replace(DOG_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # --- commands: forward speed + yaw rate (no sideways stepping possible)
        # v4: 25% of commands are pure turn-in-place (v3 turned in place at only ~20% of the command)
        self.commands.base_velocity = DogVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.2,      # v5: 0.1 -> 0.2
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            rel_turn_in_place_envs=0.25,
            turn_in_place_min_wz=0.15,  # v7: 0.3 -> 0.15, also practise slow turns
            ranges=DogVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.3, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )

        # --- observations: only what the real robot / Gazebo IMU + encoders provide
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        # --- actions: joint position offsets around the standing pose
        # v5: half of the envs apply their targets one policy step (20 ms) late
        self.actions.joint_pos = DelayedJointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, delay_prob=0.5
        )

        # --- domain randomization scaled to a 3 kg robot
        ev = self.events
        # v5: much wider friction (v4: 0.6-1.25 / 0.4-1.0 turned in place well in PhysX but not in Gazebo)
        ev.physics_material.params["static_friction_range"] = (0.3, 1.5)
        ev.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
        ev.add_base_mass.params["asset_cfg"].body_names = "base_link"
        ev.base_com.params["asset_cfg"].body_names = "base_link"
        ev.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.015, 0.015), "z": (-0.005, 0.005)}
        ev.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        ev.reset_base.params["pose_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)}
        ev.reset_base.params["velocity_range"] = {
            "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
        }
        ev.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        ev.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

        # --- rewards
        rw = self.rewards
        rw.track_lin_vel_xy_exp.weight = 1.5
        rw.track_lin_vel_xy_exp.params["std"] = 0.25          # speeds are ~half of Go2's
        rw.track_ang_vel_z_exp.weight = 1.5                    # v4: 0.75 -> 1.5, turning was weak
        # v7: std 0.5 -> 0.25. With 0.5 a 0.2 rad/s yaw error still earned 85% of the reward, so v6
        # ignored the small corrections Nav2's path follower sends (drifted 25 cm into a wall).
        rw.track_ang_vel_z_exp.params["std"] = 0.25
        rw.dof_torques_l2.weight = -2.0e-4
        rw.action_rate_l2.weight = -0.02                       # smooth actions (v2: -0.01 -> fast steps)
        rw.feet_air_time.weight = 1.0                          # v1 used 0.25 -> shuffling
        rw.feet_air_time.params["sensor_cfg"].body_names = ".*_calf"   # the foot sphere is on the calf
        rw.feet_air_time.params["threshold"] = 0.15            # swing lasts GAIT_PERIOD / 2 = 0.2 s
        rw.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        rw.flat_orientation_l2.weight = -2.5
        rw.dof_pos_limits.weight = -1.0

        # --- terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class DogFlatEnvCfg_PLAY(DogFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

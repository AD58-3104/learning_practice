# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg,ImuCfg
from isaaclab.actuators import  ImplicitActuatorCfg,IdealPDActuatorCfg


from . import mdp

##
# ArticulationCfg for booster T1
##
from pathlib import Path

PROJECT_HOME_DIR = Path.home() / "learning_practice" / "booster_train"

BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_HOME_DIR}/t1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72 - 0.0841),
        joint_pos={
            "AAHead_yaw" : 0.0, 
            "Head_pitch" : 0.0, 
            "Waist" : 0.0, 
            "Left_Shoulder_Pitch" : 0.0, 
            "Right_Shoulder_Pitch" : 0.0,   # ここはラジアンで指定するっぽい
            "Left_Shoulder_Roll" : -0.7853981634, 
            "Left_Elbow_Pitch" : 0.0, 
            "Left_Elbow_Yaw" : 0.0, 
            "Right_Shoulder_Roll" : 0.7853981634, 
            "Right_Elbow_Pitch" : 0.5, 
            "Right_Elbow_Yaw" : 0.0, 

            "Left_Hip_Pitch" : -0.2, 
            "Left_Hip_Roll" : 0.0, 
            "Left_Hip_Yaw" : 0.0, 
            "Left_Knee_Pitch" : 0.4, 
            "Left_Ankle_Pitch" : -0.25, 
            "Left_Ankle_Roll" : 0.0, 
            
            "Right_Hip_Pitch" : -0.2, 
            "Right_Hip_Roll" : 0.0, 
            "Right_Hip_Yaw" : 0.0, 
            "Right_Knee_Pitch" : 0.4, 
            "Right_Ankle_Pitch" : -0.25, 
            "Right_Ankle_Roll" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*" : 45.0,
                ".*Hip_Roll": 30.0,
                ".*Hip_Yaw.*": 30.0,
                ".*_Knee_Pitch": 60.0,
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*Hip_Pitch.*" : 12.5, #rad/s
                ".*Hip_Roll": 10.9,
                ".*Hip_Yaw.*": 10.9,
                ".*_Knee_Pitch": 11.7,
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness={
                ".*Hip_Yaw.*": 200.0,
                ".*Hip_Roll": 200.0,
                ".*Hip_Pitch.*": 200.0,
                ".*_Knee_Pitch": 200.0,
                ".*_Ankle_.*": 50.0,
            },
            damping={
                ".*Hip_Yaw.*": 5.0,
                ".*Hip_Roll": 5.0,
                ".*Hip_Pitch.*": 5.0,
                ".*_Knee_Pitch": 5.0,
                ".*_Ankle_.*": 1.0,
            },
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=100,
            velocity_limit=50.0,
            stiffness=40.0,
            damping=10.0,
        ),
        "bodies": IdealPDActuatorCfg(
            joint_names_expr=["Waist","AAHead_yaw", "Head_pitch"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0,
        )
    },
)


##
# Scene definition
##


@configclass
class BoosterTrainSceneCfg(InteractiveSceneCfg):
    """Configuration for a booster t1 scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # contact sensor(単に終了等を判断するために必要)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    base_imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Trunk",gravity_bias=(0,0,0),debug_vis=True)
##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=False,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-0.0, 0.4), lin_vel_y=(-0.0, 0.1), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi)
    #     ),
    # )
    base_velocity = mdp.CurriculumCommandCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=BOOSTER_T1_CFG.actuators["legs"].joint_names_expr , scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        imu_orientation = ObsTerm(func=mdp.imu_orientation,
                                    params={"asset_cfg": SceneEntityCfg("base_imu")},     #, body_names="Trunk"
                                  )
        imu_angular_velocity = ObsTerm(func=mdp.imu_ang_vel,
                                        params={"asset_cfg": SceneEntityCfg("base_imu")},     #, body_names="Trunk"
                                       )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # これはkinematicsとorientationから計算するらしい。多分stanceの順運動学からやるんだろ
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w) # ワールド座標系での速度。これ現実でどうやって取るのか知らん。意味不明
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset
    reset_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=BOOSTER_T1_CFG.actuators["legs"].joint_names_expr),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.01, 0.01),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    
    # 姿勢に関する正則化項
    pose_regularization = RewTerm(func=mdp.pose_regularization, weight=4.0)
    # コマンド追従
    command_tracking = RewTerm(func=mdp.command_tracking,params = {"command_name":"base_velocity"}, weight=42.0)
    # 長く生き残る方が報酬が高い
    # これはterminationsConfigの方で生き残り基準を設定する。基準はroll,pitch角度とbaseの高さが規定を満たす事
    is_alive = RewTerm(func=mdp.is_alive, weight=1e-4 * 4)
    # 足上げ高さに関する報酬
    foot_clearance = RewTerm(func=mdp.foot_clearance,weight=18.0)

    # パラメータに関するメモ
    # 姿勢以外の値を大きくしたらめちゃくちゃ足を開いてしまった (これは論文の重みを見る前の試行の話)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height_below_minimum = DoneTerm(mdp.root_height_below_minimum,params={"minimum_height": 0.6})
    bad_orientation = DoneTerm(mdp.bad_orientation,params={"limit_angle": 1.2}) # 120度くらいを超えたら終わり


##
# Environment configuration
##


@configclass
class BoosterTrainEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BoosterTrainSceneCfg = BoosterTrainSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # sensor update period
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BoosterTraisEnvCfg_PLAY(BoosterTrainEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.biped_s54 import BIPED_S54_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 1.2


@configclass
class LiteRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # energy = RewTerm(func=mdp.energy, weight=-1e-3)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                # Removed "base_link" to be consistent with termination condition
                # Penalize contacts on intermediate limb segments (not feet, not base)
                "contact_sensor", body_names=["leg_l4_link", "leg_r4_link", "zarm_l2_link", "zarm_r2_link", "zarm_l4_link", "zarm_r4_link"]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        # Use dummy_link as it's the actual root body recognized by Isaac Lab
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=["dummy_link"])}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "leg_l1_joint",
                    "leg_r1_joint",
                    "zarm_l1_joint",
                    "zarm_r1_joint",
                    "zarm_l4_joint",
                    "zarm_r4_joint",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_l2_joint", "zarm_r2_joint", "zarm_l3_joint", "zarm_r3_joint"])},
    )
    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.02,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "leg_l3_joint",
    #                 "leg_r3_joint",
    #                 "leg_l4_joint",
    #                 "leg_r4_joint",
    #                 "leg_l5_joint",
    #                 "leg_r5_joint",
    #                 "leg_l6_joint",
    #                 "leg_r6_joint",
    #             ],
    #         )
    #     },
    # )

    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    # ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    # ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    # hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    # hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)


@configclass
class Kuavo5WalkFlatEnvCfg:
    # 可以添加多个 visualization 文件用于测试播放
    amp_motion_files_display = [
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/侧移_右_低速_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/侧移_左_低速_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/原地踏步_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/原地转圈_中速_逆时针_Skeleton_001_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/原地转圈_中速_顺时针_000_Skeleton_001_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/原地转圈_低速_逆时针_Skeleton_001_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/原地转圈_低速_顺时针_Skeleton_001_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/后退_中速_小摆手_000_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/后退_低速_小摆手_000_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/后退_低速_小摆手_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/圆_2_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/弧线_小右_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/弧线_小左_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/直行_中速_小摆手_1_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/直行_中速_小摆手_2_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/直行_中速_小摆手_3_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/直行_低速_小摆手_1_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/直行_低速_小摆手_2_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_visualization/静止站立_LiuKe_Skeleton_retargeted.txt",
        "legged_lab/envs/kuavo5/datasets/motion_visualization/kuavo5_walk.txt",
    ]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=BIPED_S54_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=5,
        # Height scanner uses dummy_link (root) as reference - similar to TienKung using pelvis
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="dummy_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        # Removed "base_link" from termination contacts - it was causing premature episode termination
        # Only terminate when non-foot lower leg/arm links touch the ground
        terminate_contacts_body_names=["leg_l3_link", "leg_r3_link", "dummy_link", "zarm_l5_link", "zarm_r5_link", "zarm_l6_link", "zarm_r6_link", "zarm_l7_link", "zarm_r7_link"],
        feet_body_names=["leg_l6_link", "leg_r6_link"],
    )
    reward = LiteRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            # Randomize dummy_link mass for robustness (similar to TienKung's pelvis mass randomization)
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="dummy_link"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
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
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Kuavo5WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "kuavo5_walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "kuavo5_walk"
    wandb_project = "kuavo5_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter - 支持多个 expert motion 文件
    amp_reward_coef = 0.3
    # 可以添加多个 expert motion 用于训练（建议包含多种动作模式）
    amp_motion_files = [
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/侧移_右_低速_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/侧移_左_低速_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/原地踏步_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/原地转圈_中速_逆时针_Skeleton_001_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/原地转圈_中速_顺时针_000_Skeleton_001_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/原地转圈_低速_逆时针_Skeleton_001_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/原地转圈_低速_顺时针_Skeleton_001_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/后退_中速_小摆手_000_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/后退_低速_小摆手_000_Skeleton_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/后退_低速_小摆手_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/圆_2_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/弧线_小右_Skeleton_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/弧线_小左_Skeleton_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/直行_中速_小摆手_1_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/直行_中速_小摆手_2_s52.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/直行_中速_小摆手_3_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/直行_低速_小摆手_1_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/直行_低速_小摆手_2_s52.txt",
        "legged_lab/envs/kuavo5/datasets/motion_amp_expert/静止站立_LiuKe_Skeleton_retargeted.txt",
        # "legged_lab/envs/kuavo5/datasets/motion_amp_expert/kuavo5_walk.txt",
    ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    # min_normalized_std 的长度应该等于关节数量（不是观测值维度）
    # 用于控制每个关节的归一化标准差最小值
    # Kuavo5 有 27 个关节：左腿 6 + 右腿 6 + 腰部 1 + 左臂 7 + 右臂 7
    min_normalized_std = [0.05] * 27  # 27 个关节

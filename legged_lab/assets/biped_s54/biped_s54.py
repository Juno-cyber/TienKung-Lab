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

"""Configuration for Kuavo5 (biped_s54) robot.

The following configurations are available:

* :obj:`BIPED_S54_CFG`: Kuavo5 bipedal robot with full collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

BIPED_S54_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAAC_ASSET_DIR}/biped_s54/urdf/biped_s54.urdf",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.97),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.1,
            "leg_[l,r]4_joint": 0.25,
            "leg_[l,r]5_joint": -0.15,
            "leg_[l,r]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_l[1-4]_joint",
                "leg_r[1-4]_joint",
            ],
            effort_limit_sim={
                "leg_[l,r]1_joint": 107.0,
                "leg_[l,r]2_joint": 51,
                "leg_[l,r]3_joint": 112.0,
                "leg_[l,r]4_joint": 200.0,
            },
            velocity_limit_sim={
                "leg_[l,r]1_joint": 8.4,
                "leg_[l,r]2_joint": 8.7,
                "leg_[l,r]3_joint": 10.7,
                "leg_[l,r]4_joint": 10.4,
            },
            stiffness={
                "leg_[l,r]1_joint": 700.0,
                "leg_[l,r]2_joint": 700.0,
                "leg_[l,r]3_joint": 700.0,
                "leg_[l,r]4_joint": 700.0,
            },
            damping={
                "leg_[l,r]1_joint": 10.0,
                "leg_[l,r]2_joint": 10.0,
                "leg_[l,r]3_joint": 10.0,
                "leg_[l,r]4_joint": 10.0,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 102.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 8.7,
            },
            stiffness={
                "waist_yaw_joint": 30.0,
            },
            damping={
                "waist_yaw_joint": 3.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_l[5-6]_joint",
                "leg_r[5-6]_joint",
            ],
            effort_limit_sim={
                "leg_[l,r]5_joint": 40.0,
                "leg_[l,r]6_joint": 40.0,
            },
            velocity_limit_sim={
                "leg_[l,r]5_joint": 10.8,
                "leg_[l,r]6_joint": 10.8,
            },
            stiffness={
                "leg_[l,r]5_joint": 500.0,
                "leg_[l,r]6_joint": 500.0,
            },
            damping={
                "leg_[l,r]5_joint": 7.5,
                "leg_[l,r]6_joint": 7.5,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "zarm_l[1-7]_joint",
                "zarm_r[1-7]_joint",
            ],
            effort_limit_sim={
                "zarm_[l,r]1_joint": 66.0,
                "zarm_[l,r]2_joint": 75.0,
                "zarm_[l,r]3_joint": 57.0,
                "zarm_[l,r]4_joint": 75.0,
                "zarm_[l,r]5_joint": 14.1,
                "zarm_[l,r]6_joint": 14.1,
                "zarm_[l,r]7_joint": 14.1,
            },
            velocity_limit_sim={
                "zarm_[l,r]1_joint": 18.8,
                "zarm_[l,r]2_joint": 8.0,
                "zarm_[l,r]3_joint": 7.5,
                "zarm_[l,r]4_joint": 8.0,
                "zarm_[l,r]5_joint": 17.5,
                "zarm_[l,r]6_joint": 17.5,
                "zarm_[l,r]7_joint": 17.5,
            },
            stiffness={
                "zarm_[l,r]1_joint": 30.0,
                "zarm_[l,r]2_joint": 30.0,
                "zarm_[l,r]3_joint": 15.0,
                "zarm_[l,r]4_joint": 30.0,
                "zarm_[l,r]5_joint": 15.0,
                "zarm_[l,r]6_joint": 15.0,
                "zarm_[l,r]7_joint": 15.0,
            },
            damping={
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
        ),
    },
)

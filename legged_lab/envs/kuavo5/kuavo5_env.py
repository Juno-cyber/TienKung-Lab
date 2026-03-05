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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.kuavo5.run_cfg import Kuavo5RunFlatEnvCfg
from legged_lab.envs.kuavo5.walk_cfg import Kuavo5WalkFlatEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class Kuavo5Env(VecEnv):
    def __init__(
        self,
        cfg: Kuavo5RunFlatEnvCfg | Kuavo5WalkFlatEnvCfg,
        headless,
    ):
        self.cfg: Kuavo5RunFlatEnvCfg | Kuavo5WalkFlatEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]
        
        # Initialize end-effector visualization markers
        self._init_ee_visualizer()

    def _init_ee_visualizer(self):
        """Initialize visualization markers for end-effector positions (hands and feet)."""
        # Create sphere markers for end-effectors
        ee_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/EE_Visualization",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),  # Red color by default
                        opacity=0.8,
                    ),
                )
            },
        )
        self.ee_visualizer = VisualizationMarkers(ee_visualizer_cfg)
        self.ee_visualizer.set_visibility(True)

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["leg_l6_link", "leg_r6_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["zarm_l4_link", "zarm_r4_link"], preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_l1_joint",
                "leg_l2_joint",
                "leg_l3_joint",
                "leg_l4_joint",
                "leg_l5_joint",
                "leg_l6_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_r1_joint",
                "leg_r2_joint",
                "leg_r3_joint",
                "leg_r4_joint",
                "leg_r5_joint",
                "leg_r6_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "zarm_l1_joint",
                "zarm_l2_joint",
                "zarm_l3_joint",
                "zarm_l4_joint",
                "zarm_l5_joint",
                "zarm_l6_joint",
                "zarm_l7_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "zarm_r1_joint",
                "zarm_r2_joint",
                "zarm_r3_joint",
                "zarm_r4_joint",
                "zarm_r5_joint",
                "zarm_r6_joint",
                "zarm_r7_joint",
            ],
            preserve_order=True,
        )
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=["waist_yaw_joint"],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["leg_l5_joint", "leg_l6_joint", "leg_r5_joint", "leg_r6_joint"],
            preserve_order=True,
        )
        
        # DEBUG: Print joint mapping information
        print(f"\n{'='*80}")
        print(f"[DEBUG] Joint Mapping Verification for Kuavo5")
        print(f"{'='*80}")
        print(f"Total joints: {self.robot.num_joints}")
        print(f"All joint names: {self.robot.joint_names}")
        print(f"\nJoint group indices:")
        print(f"  left_leg_ids ({len(self.left_leg_ids)}): {self.left_leg_ids}")
        print(f"  right_leg_ids ({len(self.right_leg_ids)}): {self.right_leg_ids}")
        print(f"  waist_ids ({len(self.waist_ids)}): {self.waist_ids}")
        print(f"  left_arm_ids ({len(self.left_arm_ids)}): {self.left_arm_ids}")
        print(f"  right_arm_ids ({len(self.right_arm_ids)}): {self.right_arm_ids}")
        print(f"\nExpected AMP data order (27 joints):")
        print(f"  [0-5]: Left leg (6 joints)")
        print(f"  [6-11]: Right leg (6 joints)")
        print(f"  [12]: Waist (1 joint)")
        print(f"  [13-19]: Left arm (7 joints)")
        print(f"  [20-26]: Right arm (7 joints)")
        print(f"{'='*80}\n")

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        # Init gait parameter
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        # Kuavo5 has 27 joints: 6 left leg + 6 right leg + 1 waist + 7 left arm + 7 right arm
        # NPZ data order: [leg_l1-6, leg_r1-6, waist_yaw, zarm_l1-7, zarm_r1-7]
        # AMP format: [root_pos(3), root_rot(3), dof_pos(27), root_lin_vel(3), root_ang_vel(3), dof_vel(27)]
        # Index breakdown in AMP frame:
        #   [0:3]   - root_pos
        #   [3:6]   - root_rot (euler)
        #   [6:12]  - left_leg_pos (6 joints: leg_l1-6)
        #   [12:18] - right_leg_pos (6 joints: leg_r1-6)
        #   [18:19] - waist_pos (1 joint: waist_yaw)
        #   [19:26] - left_arm_pos (7 joints: zarm_l1-7)
        #   [26:33] - right_arm_pos (7 joints: zarm_r1-7)
        #   [33:36] - root_lin_vel
        #   [36:39] - root_ang_vel
        #   [39:45] - left_leg_vel (6 joints)
        #   [45:51] - right_leg_vel (6 joints)
        #   [51:52] - waist_vel (1 joint)
        #   [52:59] - left_arm_vel (7 joints)
        #   [59:66] - right_arm_vel (7 joints)
        
        # Debug: print AMPLoader constants
        from rsl_rl.utils.motion_loader import AMPLoader
        # print(f"[DEBUG] AMPLoader.JOINT_POS_SIZE: {AMPLoader.JOINT_POS_SIZE}")
        # print(f"[DEBUG] AMPLoader.JOINT_VEL_END_IDX: {AMPLoader.JOINT_VEL_END_IDX}")
        # print(f"[DEBUG] AMPLoader.END_POS_END_IDX: {AMPLoader.END_POS_END_IDX}")
        # print(f"[DEBUG] visual_motion_frame length: {len(visual_motion_frame)}")
        
        # DEBUG: Print joint position mapping for first frame
        if self.sim_step_counter < 10:  # Only print first few frames
            print(f"\n[DEBUG t={time:.3f}] AMP frame joint positions:")
            print(f"  Left leg (indices 6-12):   {visual_motion_frame[6:12].cpu().numpy()}")
            print(f"  Right leg (indices 12-18): {visual_motion_frame[12:18].cpu().numpy()}")
            print(f"  Waist (index 18-19):       {visual_motion_frame[18:19].cpu().numpy()}")
            print(f"  Left arm (indices 19-26):  {visual_motion_frame[19:26].cpu().numpy()}")
            print(f"  Right arm (indices 26-33): {visual_motion_frame[26:33].cpu().numpy()}")
            
            # Map to Isaac Lab joint indices and print
            test_dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
            test_dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
            test_dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
            test_dof_pos[:, self.waist_ids] = visual_motion_frame[18:19]
            test_dof_pos[:, self.left_arm_ids] = visual_motion_frame[19:26]
            test_dof_pos[:, self.right_arm_ids] = visual_motion_frame[26:33]
            
            print(f"\n[DEBUG t={time:.3f}] Mapped to Isaac Lab joints:")
            for i, joint_name in enumerate(self.robot.joint_names):
                print(f"  [{i:2d}] {joint_name:25s}: {test_dof_pos[0, i]:7.4f}")
            print()
        
        dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
        dof_pos[:, self.waist_ids] = visual_motion_frame[18:19]
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[19:26]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[26:33]

        dof_vel[:, self.left_leg_ids] = visual_motion_frame[39:45]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[45:51]
        dof_vel[:, self.waist_ids] = visual_motion_frame[51:52]
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[52:59]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[59:66]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.5

        # The euler angles were converted from quaternion using 'XYZ' order in convert_kuavo5_npz.py
        # So we should use 'XYZ' order to convert back
        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[32:35].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        
        # Print end-effector positions for debugging
        # print(f"[EE Pos] Time: {time:.3f}s | "
        #       f"L_Hand: [{left_hand_pos[0, 0]:.3f}, {left_hand_pos[0, 1]:.3f}, {left_hand_pos[0, 2]:.3f}] | "
        #       f"R_Hand: [{right_hand_pos[0, 0]:.3f}, {right_hand_pos[0, 1]:.3f}, {right_hand_pos[0, 2]:.3f}] | "
        #       f"L_Foot: [{left_foot_pos[0, 0]:.3f}, {left_foot_pos[0, 1]:.3f}, {left_foot_pos[0, 2]:.3f}] | "
        #       f"R_Foot: [{right_foot_pos[0, 0]:.3f}, {right_foot_pos[0, 1]:.3f}, {right_foot_pos[0, 2]:.3f}]")
        
        # Update 3D visualization markers for end-effectors
        self._update_ee_visualization(left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos, root_pos)

        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids] 
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.waist_dof_pos = dof_pos[:, self.waist_ids]  # Add waist joint
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids] 
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.waist_dof_vel = dof_vel[:, self.waist_ids]  # Add waist joint
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids] 
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids] 
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        return torch.cat(
            (
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                self.waist_dof_pos,          # Add waist joint
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                self.waist_dof_vel,          # Add waist joint
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 27 (kuavo5 joints: 6L leg + 6R leg + 1 waist + 7L arm + 7R arm)
                joint_vel * self.obs_scales.joint_vel,  # 27
                action * self.obs_scales.actions,  # 27
                torch.sin(2 * torch.pi * self.gait_phase),  # 2
                torch.cos(2 * torch.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            dim=-1,
        )
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]

            # (num_envs, height, width, 1) --> (num_envs, height * width)
            flattened_depth = depth_image.view(self.num_envs, -1)

            # Append the flattened depth data to the end of the actor and critic observation vectors.
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos
                
        # DEBUG: Print action and joint position target for first few steps
        if self.sim_step_counter < 20:
            print(f"\n[DEBUG Step] Action stats: min={self.action.min():.3f}, max={self.action.max():.3f}, mean={self.action.mean():.3f}")
            print(f"[DEBUG Step] Action by group:")
            print(f"  Left leg (ids {self.left_leg_ids}): {self.action[0, self.left_leg_ids].cpu().numpy()}")
            print(f"  Right leg (ids {self.right_leg_ids}): {self.action[0, self.right_leg_ids].cpu().numpy()}")
            print(f"  Waist (ids {self.waist_ids}): {self.action[0, self.waist_ids].cpu().numpy()}")
            print(f"  Left arm (ids {self.left_arm_ids}): {self.action[0, self.left_arm_ids].cpu().numpy()}")
            print(f"  Right arm (ids {self.right_arm_ids}): {self.action[0, self.right_arm_ids].cpu().numpy()}")
            print(f"[DEBUG Step] Processed action stats: min={processed_actions.min():.3f}, max={processed_actions.max():.3f}, mean={processed_actions.mean():.3f}")
            print(f"[DEBUG Step] Default joint pos: {self.robot.data.default_joint_pos[0].cpu().numpy()}")

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        
        # DEBUG: Print torque and reward for first few steps
        if self.sim_step_counter < 20:
            print(f"[DEBUG Reward] Applied torque stats: min={self.robot.data.applied_torque.min():.3f}, max={self.robot.data.applied_torque.max():.3f}, mean={torch.abs(self.robot.data.applied_torque).mean():.3f}")
            print(f"[DEBUG Reward] Torque by group:")
            print(f"  Left leg (ids {self.left_leg_ids}): {self.robot.data.applied_torque[0, self.left_leg_ids].cpu().numpy()}")
            print(f"  Right leg (ids {self.right_leg_ids}): {self.robot.data.applied_torque[0, self.right_leg_ids].cpu().numpy()}")
            print(f"  Waist (ids {self.waist_ids}): {self.robot.data.applied_torque[0, self.waist_ids].cpu().numpy()}")
            print(f"  Left arm (ids {self.left_arm_ids}): {self.robot.data.applied_torque[0, self.left_arm_ids].cpu().numpy()}")
            print(f"  Right arm (ids {self.right_arm_ids}): {self.robot.data.applied_torque[0, self.right_arm_ids].cpu().numpy()}")
            print(f"[DEBUG Reward] Reward buf: {reward_buf[0].cpu().numpy()}")
        
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[9:12] = 0
            noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos  # 27 joints
            noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            noise_vec[12 + self.num_actions * 3 : 18 + self.num_actions * 3] = 0.0  # gait phase and ratio
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy for Kuavo5 robot"""
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        
        # Extract joint positions and velocities for Kuavo5 (27 joints total, including waist)
        self.left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        self.waist_dof_pos = self.robot.data.joint_pos[:, self.waist_ids]  # Add waist joint
        self.left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        self.waist_dof_vel = self.robot.data.joint_vel[:, self.waist_ids]  # Add waist joint
        self.left_arm_dof_pos = self.robot.data.joint_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = self.robot.data.joint_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]
        
        # Return AMP observation vector with all 27 joints + end-effectors
        return torch.cat(
            (
                self.right_arm_dof_pos,      # 7
                self.left_arm_dof_pos,       # 7
                self.right_leg_dof_pos,      # 6
                self.left_leg_dof_pos,       # 6
                self.waist_dof_pos,          # 1 (waist joint)
                self.right_arm_dof_vel,      # 7
                self.left_arm_dof_vel,       # 7
                self.right_leg_dof_vel,      # 6
                self.left_leg_dof_vel,       # 6
                self.waist_dof_vel,          # 1 (waist joint)
                left_hand_pos,               # 3
                right_hand_pos,              # 3
                left_foot_pos,               # 3
                right_foot_pos,              # 3
            ),
            dim=-1,
        )  # Total: 66 dimensions (27 joint pos + 27 joint vel + 12 end-effector pos)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _update_ee_visualization(self, left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos, root_pos):
        """
        Update 3D visualization markers for end-effector positions.
        
        Args:
            left_hand_pos: Left hand position in root frame [num_envs, 3]
            right_hand_pos: Right hand position in root frame [num_envs, 3]
            left_foot_pos: Left foot position in root frame [num_envs, 3]
            right_foot_pos: Right foot position in root frame [num_envs, 3]
            root_pos: Root position in world frame [num_envs, 3]
        """
        # Transform end-effector positions from root frame to world frame
        left_hand_world = quat_apply(self.robot.data.root_state_w[:, 3:7], left_hand_pos) + root_pos
        right_hand_world = quat_apply(self.robot.data.root_state_w[:, 3:7], right_hand_pos) + root_pos
        left_foot_world = quat_apply(self.robot.data.root_state_w[:, 3:7], left_foot_pos) + root_pos
        right_foot_world = quat_apply(self.robot.data.root_state_w[:, 3:7], right_foot_pos) + root_pos
        
        # Stack all end-effector positions [num_envs * 4, 3]
        ee_positions = torch.cat([
            left_hand_world,   # [num_envs, 3]
            right_hand_world,  # [num_envs, 3]
            left_foot_world,   # [num_envs, 3]
            right_foot_world,  # [num_envs, 3]
        ], dim=0)
        
        # Update visualization markers
        if hasattr(self, "ee_visualizer"):
            self.ee_visualizer.visualize(translations=ee_positions)

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import torch
import math

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask

body_ids_offsets = {
    # axis: x 0 y 1 z 2

    # left leg
    # hip
    2: { "offset" : 20, "size": 1, 'axis': 1},
    #2: { "offset" : 18, "size": 1, 'axis': 2},
    #2: { "offset" : 19, "size": 1, 'axis': 0},
    # knee
    8: { "offset" : 21, "size": 1, 'axis': 1},
    # foot
    16: { "offset" : 22, "size": 1, 'axis': 1},
    #20: { "offset" : 23, "size": 1, 'axis': 0},

    # right leg
    # hip
    3: { "offset" : 26, "size": 1, 'axis': 1},
    #3: { "offset" : 24, "size": 1, 'axis': 2},
    #3: { "offset" : 25, "size": 1, 'axis': 0},
    # knee
    9: { "offset" : 27, "size": 1, 'axis': 1},
    # foot
    18: { "offset" : 28, "size": 1, 'axis': 1},
    #22: { "offset" : 29, "size": 1, 'axis': 0},

    # torso
    1: { "offset" : 0, "size": 3, 'axis': -1},
    1: { "offset" : 1, "size": 0, 'axis': -1},
    1: { "offset" : 2, "size": 0, 'axis': -1},
    # head
    12: { "offset" : 10, "size": 0, 'axis': -1},

    # left arm
    10: { "offset" : 4, "size": 3, 'axis': 0},
    #19: { "offset" : 3, "size": 1, 'axis': 2},
    25: { "offset" : 5, "size": 1, 'axis': 1},
    25: { "offset" : 6, "size": 1, 'axis': 0},
    29: { "offset" : 7, "size": 1, 'axis': 1},
    29: { "offset" : 8, "size": 1, 'axis': 0},
    # right arm
    13: { "offset" : 12, "size": 3, 'axis': 0},
    #13: { "offset" : 11, "size": 1, 'axis': 2},
    26: { "offset" : 13, "size": 1, 'axis': 1},
    26: { "offset" : 14, "size": 1, 'axis': 0},
    30: { "offset" : 15, "size": 1, 'axis': 1},
    30: { "offset" : 16, "size": 1, 'axis': 0},
}

KEY_BODY_NAMES = ["r_foot", "l_foot"]
#KEY_BODY_NAMES = ["r_hand", "l_hand", "r_foot", "l_foot"]

NUM_OBS = 98 + (len(KEY_BODY_NAMES) * 3) #13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 30

class AtlasAMPBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self.body_ids_map = {
            10: 11, # l_hand - left_hand
            18: 14, # r_hand - right_hand
            24: 3, # l_foot - left_foot
            30: 6 #r_foot - left_foot
        }

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.task_enabled = self.cfg["task"]["enable"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.task_enabled:
            # reward scales
            self.rew_scales = {}
            self.rew_scales["termination"] = self.cfg["task"]["learn"]["terminalReward"]
            self.rew_scales["lin_vel_xy"] = self.cfg["task"]["learn"]["linearVelocityXYRewardScale"]
            self.rew_scales["ang_vel_z"] = self.cfg["task"]["learn"]["angularVelocityZRewardScale"]

            #command ranges
            self.command_x_range = self.cfg["task"]["randomCommandVelocityRanges"]["linear_x"]
            #self.command_y_range = self.cfg["task"]["randomCommandVelocityRanges"]["linear_y"]
            self.command_yaw_range = self.cfg["task"]["randomCommandVelocityRanges"]["yaw"]

        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "r_arm_shx")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "l_arm_shx")
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self.max_episode_length_s = self.cfg["env"]["episodeLength"]
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)

        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "ang_vel_z": torch_zeros(), "total_reward": torch_zeros()}

        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)

        if self.task_enabled:
            self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
            #self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
            self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
            self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1141
        asset_options.angular_damping = 0.4
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.01
        asset_options.thickness = 0.01
        asset_options.flip_visual_attachments = True
        if (self._pd_control):
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        else:
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = False
        asset_options.override_com = True
        #asset_options.override_inertia = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        dof_props = self.gym.get_asset_dof_properties(humanoid_asset)

        motor_efforts = [prop[5] for prop in dof_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "r_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "l_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        for j in range(self.num_dof):
            dof_props['effort'][j] = math.ceil(dof_props['effort'][j] * self.power_scale)
            #dof_props['velocity'][j] = math.ceil(dof_props['velocity'][j] * self.power_scale)
            #dof_props['stiffness'][j] = self.cfg["env"]["control"]["stiffness"]
            #dof_props['damping'][j] = self.cfg["env"]["control"]["damping"]

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.95, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0

            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "atlas", i, 1, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        print("Body names")
        self.body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        for j, body_name in enumerate(self.body_names):
            print(j, body_name)

        print("DOF names")
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        for j, dof_name in enumerate(self.dof_names):
            print(f"""DOF props: {dof_name}
                  hasLimits {dof_props[j]['hasLimits']} lower {dof_props[j]['lower']} upper {dof_props[j]['upper']}
                  driveMode {dof_props[j]['driveMode']} stiffness {dof_props[j]['stiffness']} damping {dof_props[j]['damping']}
                  velocity {dof_props[j]['velocity']} effort {dof_props[j]['effort']} friction {dof_props[j]['friction']} armature {dof_props[j]['armature']}""")

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "utorso")

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(body_ids_offsets.keys())

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for body_id in body_ids_offsets.keys():
            dof_offset = body_ids_offsets[body_id]['offset']
            dof_size = body_ids_offsets[body_id]['size']

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):

        if self.task_enabled:
            base_quat = self._root_states[:, 3:7]
            base_lin_vel = quat_rotate_inverse(base_quat, self._root_states[:, 7:10])
            base_ang_vel = quat_rotate_inverse(base_quat, self._root_states[:, 10:13])

            lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
            ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
            rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]
            rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

            self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z

            self.rew_buf[:] += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

            # log episode reward sums
            self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
            self.episode_sums["total_reward"] += self.rew_buf[:]
            self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        else:
            compute_humanoid_reward(self.obs_buf)

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos[:, self.base_index, :], self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        if self.task_enabled:
            base_quat = self._root_states[:, 3:7]
            base_lin_vel = quat_rotate_inverse(base_quat, self._root_states[:, 7:10])
            base_ang_vel = quat_rotate_inverse(base_quat, self._root_states[:, 10:13])

            task_obs = torch.cat((self.commands[:, :3],), dim=-1)
        else:
            task_obs = torch.zeros_like((self.commands[:, :3]))

        if (env_ids is None):
            #self.obs_buf[:] = obs
            self.obs_buf[:] = torch.hstack([humanoid_obs, task_obs])
        else:
            #self.obs_buf[env_ids] = obs
            self.obs_buf[env_ids] = torch.hstack([humanoid_obs, task_obs[[env_ids]]])

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

        key_body_pos = torch.zeros_like(key_body_pos).to(self.device)
        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        #assert not actions.isnan().any()
        #self.actions = torch.zeros_like(actions)
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reset()
        self._compute_reward(self.actions)

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, mode="rgb_array"):
        if self.viewer and self.camera_follow:
            self._update_camera()

        return super().render(mode)

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 52

    body_ids_offsets = {
        # axis: x 0 y 1 z 2

        # left leg
        # hip
        2: { "offset" : 20, "size": 1, 'axis': 1},
        # knee
        8: { "offset" : 21, "size": 1, 'axis': 1},
        # foot
        16: { "offset" : 22, "size": 1, 'axis': 1},

        # right leg
        # hip
        3: { "offset" : 26, "size": 1, 'axis': 1},
        # knee
        9: { "offset" : 27, "size": 1, 'axis': 1},
        # foot
        18: { "offset" : 28, "size": 1, 'axis': 1},

        # torso
        1: { "offset" : 0, "size": 3, 'axis': -1},
        1: { "offset" : 1, "size": 0, 'axis': -1},
        1: { "offset" : 2, "size": 0, 'axis': -1},
        # head
        12: { "offset" : 10, "size": 0, 'axis': -1},

        # left arm
        19: { "offset" : 4, "size": 0, 'axis': 0},
        25: { "offset" : 5, "size": 0, 'axis': 1},
        29: { "offset" : 8, "size": 0, 'axis': 0},
        # right arm
        21: { "offset" : 12, "size": 0, 'axis': 0},
        26: { "offset" : 13, "size": 0, 'axis': 1},
        30: { "offset" : 16, "size": 0, 'axis': 0},
    }

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for body_id in body_ids_offsets.keys():
        dof_offset = body_ids_offsets[body_id]['offset']
        dof_size = body_ids_offsets[body_id]['size']
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        elif (dof_size == 1):
            joint_dof_obs = joint_pose
            dof_obs_size = 1
        elif (dof_size == 0):
            joint_dof_obs = joint_pose
            dof_obs_size = 0
        else:
            print("Unsupported joint type")
            joint_dof_obs = joint_pose
            dof_obs_size = 0
            assert(False)
        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos)
    #print(dof_obs)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    #assert not obs.isnan().any()
    return obs

@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        #feet_no_contact = torch.all(masked_contact_buf[:, contact_body_ids, :] <= 0.0, dim=-1)
        #feet_no_contact = torch.all(feet_no_contact, dim=-1)

        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        #print(body_height[0])
        fall_height = body_height < termination_height
        fly_height = body_height > termination_height*3
        #fall_height[:, contact_body_ids] = False
        #fall_height = torch.any(fall_height, dim=-1)

        #feet_no_contact *= (progress_buf > 3)
        has_fallen_or_flying = torch.logical_or(fall_height, fly_height)
        #feet_in_the_air_body_contact = torch.logical_or(feet_no_contact, fall_contact)

        terminate = torch.logical_or(has_fallen_or_flying, fall_contact)

        #if feet_no_contact[0]:
        #    print("feet_no_contact")
        #if fall_contact[0]:
        #    print("fall_contact")
        #if fall_height[0]:
        #    print("fall_height")
        #if fly_height[0]:
        #    print("fly_height")
        #if terminate[0]:
        #    print("terminate")
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        terminate *= (progress_buf > 1)
        terminated = torch.where(terminate, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
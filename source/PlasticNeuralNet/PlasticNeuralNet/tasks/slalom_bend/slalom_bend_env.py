import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor

import math
import torch
import numpy as np

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import isaacsim.core.utils.torch as torch_utils

from PlasticNeuralNet.assets.robots.slalom import SLALOM_CFG
from .slalom_bend_env_cfg import SlalomBendEnvCfg


class SlalomBendLocomotionTask(DirectRLEnv):
    cfg: SlalomBendEnvCfg

    def __init__(self, cfg: SlalomBendEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Define number of spaces
        self.num_actions = self.cfg.action_space
        self.num_observations = self.cfg.observation_space
        self.action_scale = self.cfg.action_scale
        
        # set action
        self.actions = torch.zeros((self.num_envs, self.num_actions) , device=self.sim.device )
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions) , device=self.sim.device)

        # define joint dof index
        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        # print(_)
        # print("--------------------")
        # print(self._joint_dof_idx)
        # set dof limit
        self.dof_limits_lower = self.robot.data.soft_joint_pos_limits[0, :, 0] 
        self.dof_limits_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]


        # define target pos to forward in X-axis
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins

        # robot posture vector
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1)) # for Upright posture
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat( # for heading posture
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone() # for heading
        self.basis_vec1 = self.up_vec.clone() # for upright posture

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

        # Foot Contact
        self.foot_contact_force = torch.zeros(self.num_envs , 4 , dtype=torch.float32, device=self.sim.device)
        self.foot_contact_status = torch.zeros(self.num_envs , 4 , dtype=torch.float32, device=self.sim.device)
        # Foot indices
        self.foot_indices , _ = self.robot.find_bodies("foot_.*")  # force_links = ["foot_lf", "foot_rf", "foot_lh", "foot_rh"]
       
        # Extras for Log / Analyze
        # Logging

        self._episode_reward = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_reward",
                "heading_reward",
                "up_reward",
            ]
        }

        self.vel_loc = torch.zeros(self.num_envs , 3 , dtype=torch.float32, device=self.sim.device)
        self.heading_proj = torch.zeros(self.num_envs , dtype=torch.float32, device=self.sim.device)
        self.up_proj = torch.zeros(self.num_envs  , dtype=torch.float32, device=self.sim.device)

    def _setup_scene(self):
        # get robot cfg
        self.robot = Articulation(self.cfg.robot)
        # Add contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):

        # pos = self.action_scale * self.joint_gears * self.actions

        self.actions = 0.1*self.actions + 0.9*self.prev_actions
        self.prev_actions = self.actions
        self.robot.set_joint_position_target(self.actions, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        # get pose in world frame
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        # get vel
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        # get joint position and joint velocity
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        
        # compute some useful value 
        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        foot_status = self._get_foot_status()
        obs = torch.cat(
            (
                self.dof_pos ,          # inx 0 - 18
                normalize_angle(self.roll).unsqueeze(-1),       # inx 19
                normalize_angle(self.pitch).unsqueeze(-1),      # inx 20
                normalize_angle(self.yaw).unsqueeze(-1),        # inx 21
                foot_status ,                                   # inx 22 - 25
            ),
            dim=-1
        )
        observations = {"policy" : obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # speed reward
        lin_vel_reward = self.vel_loc[:, 0] * self.cfg.lin_vel_weight

        # heading rewards    
        heading_reward = torch.where(self.heading_proj > 0.95 , 0 , -0.5)    *   self.cfg.heading_weight

        # aligning up axis of robot and environment
        up_reward = torch.where(torch.abs(self.up_proj) < 0.45, 0 , -0.5)  *   self.cfg.up_weight

        rewards = {
            "lin_vel_reward" : lin_vel_reward , 
            "heading_reward" : heading_reward ,
            "up_reward" : up_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # for key, value in rewards.items():
        #     self._episode_reward[key] += value

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        # died = self.torso_position[:, 2] < self.cfg.termination_height
        died = torch.zeros_like(time_out, dtype=torch.bool)
        return died , time_out 
        # return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)


        # ------------------ Reset Action ------------------ #
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        # ------------------ Reset Robot ------------------ # 
        # joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_vel = self.robot.data.default_joint_vel[env_ids]
        joint_pos = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.2,0.2)
        joint_pos[:] = torch.clamp(self.robot.data.default_joint_pos[env_ids] + joint_pos , self.robot.data.joint_pos_limits[: , : , 0], self.robot.data.joint_pos_limits[: , : , 1])
        
        joint_vel = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.1,0.1)

        # Get Root Pose
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Set Root Pose/Vel
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # Set Joint State
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        # Logging Reward

        self.extras["log"] = dict()
        extras = dict() # Temporary Buffer
        for key in self._episode_reward.keys():
            episodic_sum_avg = torch.mean(self._episode_reward[key][env_ids])
            extras[key] = episodic_sum_avg / self.max_episode_length
            self._episode_reward[key][env_ids] = 0.0
        self.extras["log"].update(extras)
        # # Logging Value
        extras = dict()
        extras["lin_vel"] = torch.mean(self.vel_loc[:,0])
        extras["heading"] = torch.mean(self.heading_proj)
        extras["up_right"] = torch.mean(self.up_proj)
        self.extras["log"].update(extras)

        self._compute_intermediate_values()

    # def _get_foot_status(self):
    #     self.foot_contact_force[:,0] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,0])
    #     self.foot_contact_force[:,1] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,1])
    #     self.foot_contact_force[:,2] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,2])
    #     self.foot_contact_force[:,3] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,3])
        
    #     self.foot_contact_status[:,0] = 1 if self.foot_contact_force[:,0] > 1 else 0
    #     self.foot_contact_status[:,1] = 1 if self.foot_contact_force[:,0] > 1 else 0
    #     self.foot_contact_status[:,2] = 1 if self.foot_contact_force[:,0] > 1 else 0
    #     self.foot_contact_status[:,3] = 1 if self.foot_contact_force[:,0] > 1 else 0

    #     return torch.stack((self.foot_contact_status[:,0] , self.foot_contact_status[:,1] , self.foot_contact_status[:,2] , self.foot_contact_status[:,3]),dim=1)
    
    def _get_foot_status(self):
        f = self.scene["contact_sensor"].data.net_forces_w  # shape (num_envs, 4, 3)
        # compute per-foot norm
        foot_force_norm = torch.norm(f, dim=-1)            # (num_envs, 4)
        # threshold at 1.0
        foot_status = (foot_force_norm > 1.0).float()      # (num_envs, 4)
        return foot_status


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position # in world frame
    to_target[:, 2] = 0.0

    
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    # change global frame to local frame
    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    # normalize if you need
    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))
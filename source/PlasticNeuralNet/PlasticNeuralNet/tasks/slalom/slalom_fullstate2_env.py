import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

import math
import torch
import numpy as np

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate , quat_rotate
import isaacsim.core.utils.torch as torch_utils

from PlasticNeuralNet.assets.robots.slalom import SLALOM_CFG
from .slalom_fullstate_env_cfg import SlalomFullStateEnvCfg
    
class SlalomFullState2LocomotionTask(DirectRLEnv):
    cfg: SlalomFullStateEnvCfg

    def __init__(self, cfg: SlalomFullStateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Define number of spaces
        self.num_actions = self.cfg.action_space
        self.num_observations = self.cfg.observation_space
        self.action_scale = self.cfg.action_scale
        # self.angular_velocity_scale = self.cfg.angular_velocity_scale
        
        # set action
        self.actions = torch.zeros((self.num_envs, self.num_actions) , device=self.sim.device )
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions) , device=self.sim.device)

        # init gear ratio of (Gecko no use for now)
        # self.joint_gears = torch.tensor(np.repeat(8,self.cfg.action_space), dtype=torch.float32, device=self.device)
        # self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        
        # define joint dof index
        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        # set dof limit
        self.dof_limits_lower = self.robot.data.soft_joint_pos_limits[0, :, 0] 
        self.dof_limits_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]


        # define target pos to forward in X-axis
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
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
        # Foot indices
        self.foot_indices , _ = self.robot.find_bodies("foot_.*")
        # Undesired contact indices
        pattern = r"^(?!foot_).*"
        self.undesired_indices, _ = self.robot.find_bodies(pattern)
        print(_)
        # self.foot_indices = torch.tensor(
        #     [self.robot.data.body_names.index(n) for n in self.foot_names],
        #     dtype=torch.long,
        #     device=self.sim.device,
        # )
        
        # Logging
        self._episode_sums = {
        key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for key in [
            "trck_lin_vel",
            "heading",
            "upright",
            "torque",
            "feet_air_time",
            "undesired_contacts",
        ]
    }

    

    def _setup_scene(self):
        # get robot cfg
        self.robot = Articulation(self.cfg.robot)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # Add contact sensor to scene
        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # self.contact_sensor_lf = ContactSensor(self.cfg.contact_force_lf)
        # self.contact_sensor_rf = ContactSensor(self.cfg.contact_force_rf)
        # self.contact_sensor_lh = ContactSensor(self.cfg.contact_force_lh)
        # self.contact_sensor_rh = ContactSensor(self.cfg.contact_force_rh)
        # self.scene.sensors["contact_sensor_lf"] = self.contact_sensor_lf
        # self.scene.sensors["contact_sensor_rf"] = self.contact_sensor_rf
        # self.scene.sensors["contact_sensor_lh"] = self.contact_sensor_lh
        # self.scene.sensors["contact_sensor_rh"] = self.contact_sensor_rh

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
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
        # get joint position , joint velocity , joint effort 
        self.dof_pos, self.dof_vel, self.dof_effort = self.robot.data.joint_pos, self.robot.data.joint_vel , self.robot.data.applied_torque
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
        foot_pos = self._get_foot_pos_local()
        foot_contact = self._get_foot_contact()
        obs = torch.cat(
            (
                self.dof_effort ,       # inx 0 - 15    joint_torque
                self.dof_vel ,          # inx 16 - 31   joint_vel
                self.dof_pos ,          # inx 32 - 47   joint_pos
                self.torso_position ,   # inx 48 49 50  base_pos
                foot_pos,               # inx 51 - 62   foot position
                foot_contact,           # inx 63 - 66   foot contact
                self.angvel_loc,        # inx 67 - 69   base_angular vel [local]
                self.vel_loc,           # inx 70 - 72   base_lin vel [local]
                self.robot.data.projected_gravity_b,               # inx 73 - 75 Projected Gravity
            ),
            dim=-1
        )
        observations = {"policy" : obs} # 76 term

        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = self.vel_loc[:, 0]
        heading_proj = self.heading_proj
        up_proj = self.up_proj
        joint_torques = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        first_contact = self.contact_sensor.compute_first_contact(self.step_dt)[:, self.foot_indices]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.foot_indices]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)

        # speed reward
        lin_vel_reward = lin_vel * self.cfg.lin_vel_weight
        # heading rewards    
        heading_reward = torch.where(heading_proj > 0.95 , 0 , 1.0)    *   self.cfg.heading_weight
        # aligning up axis of robot and environment
        up_reward = torch.where(torch.abs(up_proj) < 0.45, 0 , 1.0)  *   self.cfg.up_weight
        # up_reward = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1) *   self.cfg.up_weight
        # torque
        torque_reward =  joint_torques   *   self.cfg.joint_torque_weight
        # Air 
        feet_air_time_reward = air_time * self.cfg.feet_air_time_weight
        # undesired contact
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_indices], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        undesired_contacts = contacts * self.cfg.undesired_contact_weight
        
        rewards = {
            "trck_lin_vel" : lin_vel_reward ,
            "heading" : heading_reward , 
            "upright" : up_reward ,
            "torque" : torque_reward ,
            "feet_air_time" : feet_air_time_reward ,
            "undesired_contacts" : undesired_contacts
        }
        
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # total_reward = lin_vel_reward + heading_reward + up_reward + torque_reward

        for key, value in rewards.items():
            self._episode_sums[key] += value
        # print(total_reward)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        # died = self.torso_position[:, 2] < self.cfg.termination_height
        died = torch.zeros_like(time_out, dtype=torch.bool)
        died = time_out

        return died , time_out 

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # ------------------ Reset Action ------------------ #
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        # ------------------ Reset Robot ------------------ # 
        default_pos = self.robot.data.default_joint_pos[env_ids]

        rand_pos = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.2,0.2)
        joint_pos = torch.clamp(default_pos + rand_pos , self.robot.data.joint_pos_limits[env_ids , : , 0], self.robot.data.joint_pos_limits[env_ids , : , 1])
        # joint_vel = torch.empty((num_reset, self.robot.num_joints), device=self.device).uniform_(-0.1, 0.1)
        joint_vel = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.05,0.05)

        #
        #  Get Root Pose
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

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self._compute_intermediate_values()

    def _get_foot_pos_local(self) -> torch.Tensor:
        """Return (num_envs, 4, 3) tensor of foot positions in the robot-base frame."""
        # ------------------------------------------------------------------
        foot_pos_w  = self.robot.data.body_pos_w[:, self.foot_indices]      # (N,4,3)
        base_pos_w  = self.torso_position.unsqueeze(1)                      # (N,1,3)
        delta_w     = foot_pos_w - base_pos_w                               # (N,4,3)

        base_quat_w = self.torso_rotation.unsqueeze(1)                      # (N,1,4)
        base_quat_conj = quat_conjugate(base_quat_w).expand(-1, 4, -1)      # (N,4,4)

        # ---------- reshape เป็น ----------
        B, F = base_quat_conj.shape[:2]          # B = num_envs, F = 4 feet
        q_flat = base_quat_conj.reshape(B*F, 4)  # (B·F, 4)
        v_flat = delta_w.reshape(B*F, 3)         # (B·F, 3)

        # ---------- rotate ----------
        v_rot  = quat_rotate(q_flat, v_flat)     # (B·F, 3)

        # ---------- reshape ----------
        foot_pos_b = v_rot.view(B, F, 3)         # (N, 4, 3)
        foot_flat = foot_pos_b.reshape(foot_pos_b.shape[0], -1)  # (N, 12)
        return foot_flat
    
    def _get_foot_contact(self):
        self.foot_contact_force[:,0] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,0])
        self.foot_contact_force[:,1] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,1])
        self.foot_contact_force[:,2] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,2])
        self.foot_contact_force[:,3] = torch.norm(self.scene["contact_sensor"].data.net_forces_w[:,3])

        return torch.stack((self.foot_contact_force[:,0] , self.foot_contact_force[:,1] , self.foot_contact_force[:,2] , self.foot_contact_force[:,3]),dim=1)
        # print(self.scene["contact_sensor_lf"])
        # print("Received contact force of: ", self.scene["contact_sensor_lf"].data.net_forces_w)
        # print("Norm : ", torch.norm(self.scene["contact_sensor_lf"].data.net_forces_w))

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
    to_target = targets - torso_position
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
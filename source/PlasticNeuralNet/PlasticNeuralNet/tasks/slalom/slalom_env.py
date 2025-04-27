import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg ,PhysxCfg , RigidBodyMaterialCfg, RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg


import math
import torch
import numpy as np

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import isaacsim.core.utils.torch as torch_utils

from PlasticNeuralNet.assets.robots.slalom import SLALOM_CFG

@configclass
class SlalomEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        use_fabric = True,
        enable_scene_query_support = False,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        ) ,
        physx= PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
            # GPU cap (optional)
            # gpu_max_rigid_contact_count=524288,
            # gpu_max_rigid_patch_count=81920,
            # gpu_found_lost_pairs_capacity=8192,
            # gpu_found_lost_aggregate_pairs_capacity=262144,
            # gpu_total_aggregate_pairs_capacity=8192,
            # gpu_heap_capacity=1048576,
            # gpu_temp_buffer_capacity=1048576,
            # gpu_max_num_partitions=67108864,
            # gpu_max_soft_body_contacts=16777216,
            # gpu_max_particle_contacts=8,
        )
    )

    # robot
    robot : ArticulationCfg = SLALOM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot")
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # scene
    scene : InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs = 4096, env_spacing = 2.0, replicate_physics=True
    )
    
    # env
    decimation = 2 # controlFrequencyInv
    episode_length_s = 9999
    action_space = 16 # for gecko
    observation_space = 55 # for gecko
    state_space = 0
    action_scale = 1
    angular_velocity_scale = 1

    # noise (no noise for now :) )
    
    
    # reward scale
    heading_weight= 0.5
    up_weight= 0.5
    lin_vel_weight = 2
    
    # cost parameter
    # actionsCost= 0.005
    # energyCost= 0.05
    # dofVelocityScale= 0.2
    # angularVelocityScale= 1.0
    # contactForceScale= 0.1
    # jointsAtLimitCost= 0.1
    # deathCost= -2.0
    # terminationHeight= 0.31
    # alive_reward_scale= 0.5

class SlalomLocomotionTask(DirectRLEnv):
    cfg: SlalomEnvCfg

    def __init__(self, cfg: SlalomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Define number of spaces
        self.num_actions = self.cfg.action_space
        self.num_observations = self.cfg.observation_space
        self.action_scale = self.cfg.action_scale
        # self.angular_velocity_scale = self.cfg.angular_velocity_scale
        
        # set action
        self.actions = torch.zeros((self.num_envs, self.num_actions) , device=self.sim.device )
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions) , device=self.sim.device)

        # init gear ratio of (Gecko no use fornow)
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
        # sensor
        # force_links = ["foot_lf", "foot_rf", "foot_lh", "foot_rh"]
        # self._sensor_indices = torch.tensor(
        #     [self._ants._body_indices[j] for j in force_links], device=self._device, dtype=torch.long
        # )

    def _setup_scene(self):
        # get robot cfg
        self.robot = Articulation(self.cfg.robot)
        
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

        #------------- default -------------#
        # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
        # obs = torch.cat(
        #     (
        #         self.torso_position[:, 2].view(-1, 1),                      # idx 0
        #         self.vel_loc,                                               # idx 1 2 3
        #         self.angvel_loc * self.cfg.angular_velocity_scale,          # idx 4 5 6 
        #         normalize_angle(self.yaw).unsqueeze(-1),                    # idx 7
        #         normalize_angle(self.roll).unsqueeze(-1),                   # idx 8
        #         normalize_angle(self.angle_to_target).unsqueeze(-1),        # idx 9
        #         self.up_proj.unsqueeze(-1),                                 # idx 10
        #         self.heading_proj.unsqueeze(-1),                            # idx 11
        #         self.dof_pos_scaled,                                        # 12 - 
        #         self.dof_vel * self.cfg.dof_vel_scale,
        #         self.actions,
        #     ),
        #     dim=-1,
        # )
        # observations = {"policy": obs}

        #------------- configs -------------#
        # obs = torch.cat(
        #     (
        #         self.dof_pos ,          # inx 0 - 15
        #         self.dof_vel ,          # inx 16 - 31
        #         normalize_angle(self.roll).unsqueeze(-1),       # inx 32
        #         normalize_angle(self.pitch).unsqueeze(-1),      # inx 33
        #         normalize_angle(self.yaw).unsqueeze(-1),        # inx 34
        #         self.actions ,                                  # inx 35-50 ***
        #         self.vel_loc ,                                  # inx 51 52 53
        #         self.up_proj.unsqueeze(-1) ,                    # inx 54
        #     ),
        #     dim=-1
        # )

        obs = torch.cat(
            (
                self.dof_pos ,          # inx 0 - 15
                normalize_angle(self.roll).unsqueeze(-1),       # inx 16
                normalize_angle(self.pitch).unsqueeze(-1),      # inx 17
                normalize_angle(self.yaw).unsqueeze(-1),        # inx 18
                self.vel_loc ,                                  # inx 19
            ),
            dim=-1
        )
        observations = {"policy" : obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.lin_vel_weight,
            self.cfg.heading_weight,
            self.cfg.up_weight,
            self.vel_loc[:, 0],
            self.heading_proj,
            self.up_proj,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # died = self.torso_position[:, 2] < self.cfg.termination_height
        died = torch.zeros_like(time_out, dtype=torch.bool)
        return died , time_out 
        # return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    lin_vel_weight : float,
    heading_weight: float,
    up_weight: float,
    lin_vel : torch.Tensor,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
):
    '''
    lin_vel_weight : float,
    heading_weight: float,
    up_weight: float,
    lin_vel : float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    '''
    # speed reward
    lin_vel_reward = lin_vel * lin_vel_weight

    # heading rewards    
    heading_reward = torch.where(heading_proj > 0.95 , 0 , -0.5)    *   heading_weight

    # aligning up axis of robot and environment
    up_reward = torch.where(torch.abs(up_proj) < 0.45, 0 , -0.5)  *   up_weight

    total_reward = lin_vel_reward + heading_reward + up_reward
    return total_reward


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
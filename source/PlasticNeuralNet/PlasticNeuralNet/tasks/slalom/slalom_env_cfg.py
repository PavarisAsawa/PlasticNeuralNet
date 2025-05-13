import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg , ViewerCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg ,PhysxCfg , RigidBodyMaterialCfg, RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg , ContactSensor

import math
import torch
import numpy as np

import isaacsim.core.utils.torch as torch_utils

from PlasticNeuralNet.assets.robots.slalom import SLALOM_CFG

@configclass
class SlalomEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
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
            ## GPU cap (optional)
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
    decimation = 4 # controlFrequencyInv
    episode_length_s = 30
    action_space = 16 # for gecko
    observation_space = 39 # 76 # for gecko
    state_space = 0
    action_scale = 1
    angular_velocity_scale = 1

    contact_debug_vis = False
    contact_force = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/foot_.*",
        update_period=0.0,
        history_length=6,
        debug_vis=contact_debug_vis,
        track_air_time = True
        )
    
    # Set View
    viewer = ViewerCfg(eye=(2.0, 2.0, 2.0))
    
    # reward scale
    lin_vel_weight = 2
    heading_weight= 0.5
    up_weight= 0.5 # -0.75
    joint_torque_weight = -2.5e-5
    feet_air_time_weight = 0.0
    undesired_contact_weight = 0.0 # -3.0
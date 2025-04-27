import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg , ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as ERewTerm
from PlasticNeuralNet.tasks.slalom.slalom_env_cfg import SlalomEnvCfg

import math
import torch
import numpy as np

import PlasticNeuralNet.tasks.locomotion.velocity.mdp as mdp
from PlasticNeuralNet.assets.robots.slalom import SLALOM_CFG


class SlalomLocomotionTask(DirectRLEnv):
    cfg: SlalomEnvCfg

    def __init__(self, cfg: SlalomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Define number of spaces
        self.num_actions = self.cfg.action_space
        self.num_observations = self.cfg.observation_space
        
        # init gear ratio of (Gecko)
        self.joint_gears = torch.tensor(np.repeat(8,self.cfg.action_space), dtype=torch.float32, device=self.device)
        
    def _setup_scene(self):
        # get robot cfg
        self.robot = Articulation(self.cfg.robot)
        
        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.num_envs = self.scene.cfg.num_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    
    # Next TODO : add reward function
    
    # def _get_rewards(self) -> torch.Tensor:
    #     total_reward = compute_reward(
            
    #     )
    
    def compute_reward(
        obs_buf
    )
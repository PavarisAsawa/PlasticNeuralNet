"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gymnasium as gym

from . import agents
from .slalom_env_cfg import SlalomEnvCfg 
##
# Register Gym environments.
##

gym.register(
    id='Isaac-slalom',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":SlalomEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gymnasium as gym

from . import agents
##
# Register Gym environments.
##


gym.register(
    id='default',
    entry_point=f"{__name__}.slalom_env:SlalomLocomotionTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slalom_env_cfg:SlalomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "es_cfg_entry_point": f"{agents.__name__}:es_cfg.yaml",
    },
)

gym.register(
    id='fullstate',
    entry_point=f"{__name__}.slalom_fullstate_env:SlalomFullStateLocomotionTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slalom_env_cfg:SlalomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "es_cfg_entry_point": f"{agents.__name__}:es_cfg.yaml",
    },
)

gym.register(
    id='fullstate2',
    entry_point=f"{__name__}.slalom_env:SlalomFullState2LocomotionTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slalom_env_cfg:SlalomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "es_cfg_entry_point": f"{agents.__name__}:es_cfg.yaml",
    },
)
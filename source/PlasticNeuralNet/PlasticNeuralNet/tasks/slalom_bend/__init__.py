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
    id='bend',
    entry_point=f"{__name__}.slalom_bend_env:SlalomBendLocomotionTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slalom_bend_env_cfg:SlalomBendEnvCfg",
        "es_cfg_entry_point": f"{agents.__name__}:es_cfg.yaml",
    },
)
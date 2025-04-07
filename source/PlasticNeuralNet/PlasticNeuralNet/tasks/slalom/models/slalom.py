# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
##
# Configuration
##


script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
slalom_path = os.path.join(script_dir, "slalom_fixedbody_16dof.usd")  # Build the full path
SLAMLOM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=slalom_path,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        )
    ),
        actuators={
            "dummy" : ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1.0,
                damping=0.0,
                effort_limit=4.1,
            )
        }
)

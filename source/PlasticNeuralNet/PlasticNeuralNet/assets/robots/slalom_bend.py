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
slalom_path = os.path.normpath(
    os.path.join(script_dir, "..", "models", "slalom_bendbody_19dof2.usd")
)

SLALOM_BEND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=slalom_path,
        activate_contact_sensors = True,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,           
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.01,
            stabilization_threshold=0.001,
        )
    ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0 ,0.0 ,0.075)
        ),
        actuators={
            "dummy" : ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1.0,
                damping=0.0,
                effort_limit=4.1,
            )
        }
        # actuators={
        #     "dummy" : ImplicitActuatorCfg(
        #         joint_names_expr=[r"^(?!joint0_).*"],
        #         stiffness=1.0,
        #         damping=0.0,
        #         effort_limit=4.1,
        #     ) ,
        #     "body" : ImplicitActuatorCfg(
        #         joint_names_expr=["joint0_.*"],
        #         stiffness=1.0,
        #         damping=0.0,
        #         effort_limit=1.0,
        #     ) ,
        # }
)
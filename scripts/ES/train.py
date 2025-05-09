import argparse
import sys

from isaaclab.app import AppLauncher
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# parser.add_argument(
#     "--train",
#     action="store_true",
#     default=False,
#     help="when given, run in training mode"
# )

parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="when given, run in test/play mode"
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="when given, log in wandb"
)


parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime
import pickle

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import PlasticNeuralNet.tasks.slalom.slalom_env
import PlasticNeuralNet.tasks.slalom.slalom_fullstate_env
import PlasticNeuralNet.tasks.slalom.slalom_fullstate2_env
import PlasticNeuralNet.tasks.anymal.anymal_c_env

from utils.ES_classes import *
from utils.feedforward_neural_net_gpu import *
from utils.hebbian_neural_net import *
from utils.LSTM_neural_net import * 
from utils.ES_agent import *
# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, "es_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with ES agent."""
    # --------------------------------- Setting up Agent --------------------------------- #
    # Initialize ES parameters
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    agent_cfg["ES_params"]["POPSIZE"] = args_cli.num_envs if args_cli.num_envs is not None else agent_cfg["POPSIZE"]
    agent_cfg["USE_TRAIN_PARAM"] = True if args_cli.test else False
    agent_cfg["wandb"]["wandb_activate"] = args_cli.wandb
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    # Set seed
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # Set Epochs (max number of episodes/iterations)
    agent_cfg["EPOCHS"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["EPOCHS"]
    )
    
    # Set Task name
    agent_cfg["task_name"] = args_cli.task if args_cli.task is not None else agent_cfg["task_name"]
    agent_cfg["num_envs"] = args_cli.num_envs if args_cli.num_envs is not None else agent_cfg["num_envs"]
    
    # Set checkpoint path
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "es", agent_cfg["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # # specify directory for logging runs
    # log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # # set directory into agent config
    # # logging directory path: <train_dir>/<full_experiment_name>
    # agent_cfg["params"]["config"]["train_dir"] = log_root_path
    # agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # new — use agent_cfg itself
    log_dir = agent_cfg.get("experiment", None)
    if not log_dir:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    agent_cfg["train_dir"] = log_root_path
    agent_cfg["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["rl_device"]
    clip_obs = agent_cfg.get("clip_observations", math.inf)
    clip_actions = agent_cfg.get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        
    # wrap around environment for rl-games
    # env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    # # register the environment to rl-games registry
    # # note: in agents configuration: environment name must be "rlgpu"
    # vecenv.register(
    #     "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    # )
    # env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    agent = ESAgent(agent_cfg)
    
    # simulate environment
    while simulation_app.is_running():
        agent.run(env=env, test=args_cli.test)
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

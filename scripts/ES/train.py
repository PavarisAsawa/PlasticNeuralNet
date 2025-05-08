import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
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
# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, "es_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with ES agent."""
    # --------------------------------- Setting up Agent --------------------------------- #
    # Initialize ES parameters
    POPSIZE             = env_cfg.num_envs
    RANK_FITNESS        = env_cfg.ES_params.rank_fitness
    ANTITHETIC          = env_cfg.ES_params.antithetic
    LEARNING_RATE       = env_cfg.ES_params.learning_rate
    LEARNING_RATE_DECAY = env_cfg.ES_params.learning_rate_decay
    SIGMA_INIT          = env_cfg.ES_params.sigma_init
    SIGMA_DECAY         = env_cfg.ES_params.sigma_decay
    LEARNING_RATE_LIMIT = env_cfg.ES_params.learning_rate_limit
    SIGMA_LIMIT         = env_cfg.ES_params.sigma_limit

    # Models
    ARCHITECTURE_NAME   = env_cfg.model
    ARCHITECTURE_TYPE   = env_cfg.model_type
    FF_ARCHITECTURE     = env_cfg.FF_ARCHITECTURE
    HEBB_ARCHITECTURE   = env_cfg.HEBB_ARCHITECTURE
    LSTM_ARCHITECTURE   = env_cfg.LSTM_ARCHITECTURE
    HEBB_init_wnoise    = env_cfg.HEBB_init_wnoise
    HEBB_norm           = env_cfg.HEBB_norm
    USE_TRAIN_HEBB      = env_cfg.USE_TRAIN_HEBB
    
    # Training parameters
    EPOCHS                  = env_cfg.EPOCHS
    EPISODE_LENGTH_TRAIN    = env_cfg.EPISODE_LENGTH_TRAIN
    EPISODE_LENGTH_TEST     = env_cfg.EPISODE_LENGTH_TEST
    SAVE_EVERY              = env_cfg.SAVE_EVERY
    USE_TRAIN_PARAM         = env_cfg.USE_TRAIN_PARAM

    # General info
    TASK = env_cfg.task_name
    TEST = env_cfg.test
    if TEST:
        USE_TRAIN_PARAM = True
    train_ff_path = env_cfg.train_ff_path
    train_hebb_path = env_cfg.train_hebb_path
    train_lstm_path = env_cfg.train_lstm_path


    # Initialize model &
    if ARCHITECTURE_NAME == 'ff':
        models = FeedForwardNet(popsize=POPSIZE,
                                sizes=FF_ARCHITECTURE,
                                )
        dir_path = 'runs_ES/'+TASK+'/ff/'
    elif ARCHITECTURE_NAME == 'hebb':
        models = HebbianNet(popsize=POPSIZE, 
                            sizes=HEBB_ARCHITECTURE,
                            init_noise=HEBB_init_wnoise,
                            norm_mode=HEBB_norm,
                            )
        dir_path = 'runs_ES/'+TASK+'/hebb/'
    elif ARCHITECTURE_NAME == 'lstm':
        models = LSTMs(popsize=POPSIZE, 
                    arch=LSTM_ARCHITECTURE,
                    )
        dir_path = 'runs_ES/'+TASK+'/lstm/'

    n_params_a_model = models.get_n_params_a_model()

    # Initialize OpenES Evolutionary Strategy Optimizer
    solver = OpenES(n_params_a_model,
                    popsize=POPSIZE,
                    rank_fitness=RANK_FITNESS,
                    antithetic=ANTITHETIC,
                    learning_rate=LEARNING_RATE,
                    learning_rate_decay=LEARNING_RATE_DECAY,
                    sigma_init=SIGMA_INIT,
                    sigma_decay=SIGMA_DECAY,
                    learning_rate_limit=LEARNING_RATE_LIMIT,
                    sigma_limit=SIGMA_LIMIT)
    solver.set_mu(models.get_a_model_params())
    
    # Use train rbf params
    # 1. solver 2. copy.deepcopy(models)  3. pop_mean_curve 4. best_sol_curve,
    if USE_TRAIN_PARAM:
        if env_cfg.model == 'ff':
            trained_data = pickle.load(open(dir_path+train_ff_path, 'rb'))
        if env_cfg.model == 'hebb':
            trained_data = pickle.load(open(dir_path+train_hebb_path, 'rb'))
        if env_cfg.model == 'lstm':
            trained_data = pickle.load(open(dir_path+train_lstm_path, 'rb'))

        train_params = trained_data[0].best_param()
        solver = trained_data[0]
        print('train_params number: ', len(train_params))

        # print('--- Used train RBF params ---')
    print('file_name: ', train_hebb_path)

    # Initialize VecEnvRLGames
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --------------------------------- Setting up Environment --------------------------------- # 
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
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
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

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
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs


    # create runner from rl-games
    # ES code
    # Log data initialize
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)
    
    # initial time to measure time of training loop 
    # initial_time = timeit.default_timer()
    # print("initial_time", initial_time)

    # Testing Loop ----------------------------------
    if TEST:
        # sample params from ES and set model params
        # solutions = solver.ask()
        # models.set_models_params(solutions)        
        
        models.set_a_model_params(train_params)
        obs = env.reset()

        # Epoch rewards
        total_rewards = torch.zeros(env_cfg.num_envs)
        total_rewards = total_rewards.cuda()
        rew = torch.zeros(env_cfg.num_envs).cuda()

        

        # rollout 
        for sim_step in range(EPISODE_LENGTH_TEST):
            actions = models.forward(obs['obs'][:,:51])
            # actions = 0.3*actions + 0.7*prev_actions
            # prev_actions = actions
            obs, reward, done, info = env.step(
                actions
            )
            
            total_rewards += reward/EPISODE_LENGTH_TEST*100
    
        # update reward arrays to ES
        total_rewards_cpu = total_rewards.cpu().numpy()
        fitlist = list(total_rewards_cpu)
        fit_arr = np.array(fitlist)
        # np.save('analysis/weights/total_rewards_Limu_'+cfg.model+'_max.npy', total_rewards_cpu)

        print('mean', fit_arr.mean(), 
              "best", fit_arr.max(), )

    else:
        # Training Loop epoch ###################################
        for epoch in range(EPOCHS):
            # sample params from ES and set model params
            solutions = solver.ask()
            models.set_models_params(solutions)
            obs = env.reset()

            # Epoch rewards
            total_rewards = torch.zeros(env_cfg.num_envs)
            total_rewards = total_rewards.cuda()

            # rollout 
            for sim_step in range(EPISODE_LENGTH_TRAIN):
                # Random actions array for testing
                # actions = torch.zeros(cfg.num_envs, env.action_space.shape[0])
                actions = models.forward(obs['obs'][:,:51])

                # print("observation", obs['obs'].shape)
                # print("action", actions[0, :])
                obs, reward, done, info = env.step(actions)

                total_rewards += reward/EPISODE_LENGTH_TRAIN*100


            # update reward arrays to ES
            total_rewards_cpu = total_rewards.cpu().numpy()
            fitlist = list(total_rewards_cpu)
            solver.tell(fitlist)

            fit_arr = np.array(fitlist)

            print('epoch', epoch, 'mean', fit_arr.mean(), 
                  'best', fit_arr.max(), )


            pop_mean_curve[epoch] = fit_arr.mean()
            best_sol_curve[epoch] = fit_arr.max()

            # WanDB Log data -------------------------------
            if env_cfg.wandb_activate:
                wandb.log({"epoch": epoch,
                            "mean" : np.mean(fitlist),
                            "best" : np.max(fitlist),
                            "worst": np.min(fitlist),
                            "std"  : np.std(fitlist)
                            })
            # -----------------------------------------------

            # Save model params and OpenES params
            if (epoch + 1) % SAVE_EVERY == 0:
                print('saving..')
                pickle.dump((
                    solver,
                    copy.deepcopy(models),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open(dir_path+TASK+'_'+env_cfg.model+'_' + env_cfg.wandb_group + '_' + str(n_params_a_model) +'_' + str(epoch) + '_' + str(pop_mean_curve[epoch])[:6] + '.pickle', 'wb'))



    env.close()

    if env_cfg.wandb_activate and global_rank == 0:
        wandb.finish()



    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    # run the main function
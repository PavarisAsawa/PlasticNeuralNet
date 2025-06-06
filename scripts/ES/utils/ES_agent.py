from .ES_classes import *
from .feedforward_neural_net_gpu import *
from .hebbian_neural_net import *
from .LSTM_neural_net import *
from tqdm import tqdm
import wandb
import pickle
import copy
import os
import datetime

class ESAgent:
    def __init__(self,agent_cfg):
        # Initialize ES parameters
        # self.POPSIZE             = agent_cfg["num_envs"]
        self.POPSIZE             = agent_cfg["ES_params"]["POPSIZE"]
        self.RANK_FITNESS        = agent_cfg["ES_params"]["rank_fitness"]
        self.ANTITHETIC          = agent_cfg["ES_params"]["antithetic"]
        self.LEARNING_RATE       = agent_cfg["ES_params"]["learning_rate"]
        self.LEARNING_RATE_DECAY = agent_cfg["ES_params"]["learning_rate_decay"]
        self.SIGMA_INIT          = agent_cfg["ES_params"]["sigma_init"]
        self.SIGMA_DECAY         = agent_cfg["ES_params"]["sigma_decay"]
        self.LEARNING_RATE_LIMIT = agent_cfg["ES_params"]["learning_rate_limit"]
        self.SIGMA_LIMIT         = agent_cfg["ES_params"]["sigma_limit"]

        # Models
        self.ARCHITECTURE_NAME   = agent_cfg["model"]
        self.ARCHITECTURE_TYPE   = agent_cfg["model_type"]
        self.FF_ARCHITECTURE     = agent_cfg["FF_ARCHITECTURE"]
        self.HEBB_ARCHITECTURE   = agent_cfg["HEBB_ARCHITECTURE"]
        self.LSTM_ARCHITECTURE   = agent_cfg["LSTM_ARCHITECTURE"]
        self.HEBB_init_wnoise    = agent_cfg["HEBB_init_wnoise"]
        self.HEBB_norm           = agent_cfg["HEBB_norm"]
        self.USE_TRAIN_HEBB      = agent_cfg["USE_TRAIN_HEBB"]
        
        # Training parameters
        self.EPOCHS                  = agent_cfg["EPOCHS"]
        self.EPISODE_LENGTH_TRAIN    = agent_cfg["EPISODE_LENGTH_TRAIN"]
        self.EPISODE_LENGTH_TEST     = agent_cfg["EPISODE_LENGTH_TEST"]
        self.SAVE_EVERY              = agent_cfg["SAVE_EVERY"]
        self.USE_TRAIN_PARAM         = agent_cfg["USE_TRAIN_PARAM"]
        
        # General Information
        self.TASK              = agent_cfg["task_name"]
        self.TEST              = agent_cfg["test"]
        if self.TEST:
            self.USE_TRAIN_PARAM = True 
        self.experiment     = agent_cfg["experiment"] # Name of experiemtn

        self.train_ff_path = agent_cfg["train_ff_path"]
        self.train_hebb_path = agent_cfg["train_hebb_path"]
        self.train_lstm_path = agent_cfg["train_lstm_path"]
        # Debug WanDB 
        self.wandb_activate = agent_cfg["wandb"]["wandb_activate"]
        self.wandb_name = agent_cfg["wandb"]["wandb_name"]
        self.wandb_group = agent_cfg["wandb"]["wandb_group"]
        self.wandb_project = agent_cfg["wandb"]["wandb_project"]

        # device
        
        self.device = agent_cfg["rl_device"]

        if self.wandb_activate:
            # run_name = f"{self.wandb_name}_{self.ARCHITECTURE_NAME}_{self.wandb_group}"
            run_name = f"{self.experiment}"
            wandb.init(
                project=self.wandb_project,
                group=self.wandb_group,
                config=agent_cfg,
                name=run_name,
            )
            
        # Initialize model
        # dir path : set from experiment
        # {model} path : set from checkpoint
        if self.ARCHITECTURE_NAME == 'ff':
            self.models = FeedForwardNet(popsize=self.POPSIZE,
                                    sizes=self.FF_ARCHITECTURE,
                                    )
            self.dir_path = 'logs/'+'es/'+self.TASK+'/ff/'+self.experiment
        elif self.ARCHITECTURE_NAME == 'hebb':
            self.models = HebbianNet(popsize=self.POPSIZE, 
                                sizes=self.HEBB_ARCHITECTURE,
                                init_noise=self.HEBB_init_wnoise,
                                norm_mode=self.HEBB_norm,
                                )
            self.dir_path = 'logs/'+'es/'+self.TASK+'/hebb/'+self.experiment
        elif self.ARCHITECTURE_NAME == 'lstm':
            self.models = LSTMs(popsize=self.POPSIZE, 
                        arch=self.LSTM_ARCHITECTURE,
                        )
            self.dir_path = 'logs/'+'es/'+self.TASK+'/lstm/'+self.experiment
        
        # Get *Number* of Param from model
        self.n_params_a_model = self.models.get_n_params_a_model()
    
        # Initialize OpenES Evolutionary Strategy Optimizer
        if not self.TEST:
            self.solver = OpenES(self.n_params_a_model,
                    popsize=self.POPSIZE,
                    rank_fitness=self.RANK_FITNESS,
                    antithetic=self.ANTITHETIC,
                    learning_rate=self.LEARNING_RATE,
                    learning_rate_decay=self.LEARNING_RATE_DECAY,
                    sigma_init=self.SIGMA_INIT,
                    sigma_decay=self.SIGMA_DECAY,
                    learning_rate_limit=self.LEARNING_RATE_LIMIT,
                    sigma_limit=self.SIGMA_LIMIT)
            self.solver.set_mu(self.models.get_a_model_params())
            pass

        if self.USE_TRAIN_PARAM:
            if self.ARCHITECTURE_NAME == 'ff':
                trained_data = pickle.load(open(self.dir_path+"/model/"+self.train_ff_path, 'rb'))
            if self.ARCHITECTURE_NAME == 'hebb':
                trained_data = pickle.load(open(self.dir_path+"/model/"+self.train_hebb_path, 'rb'))
            if self.ARCHITECTURE_NAME == 'lstm':
                trained_data = pickle.load(open(self.dir_path+"/model/"+self.train_lstm_path, 'rb'))

            self.train_params = trained_data[0].best_param()
            self.solver = trained_data[0]
            print('train_params number: ', len(self.train_params))
            
    
    def run(self,env, test=False):
        """
        Run the ES agent on the environment.
        :param env: The environment to run the agent on.
        :param train: Whether to train the agent or not.
        """        
        # ES code
        # Log data initialized

        if test:   # Trainig Loop
            self.run_play(env=env)
        else:       # Playing Loop
            self.run_train(env=env)
        
 
    def run_train(self,env):
        pop_mean_curve = np.zeros(self.EPOCHS)
        best_sol_curve = np.zeros(self.EPOCHS)
        eval_curve = np.zeros(self.EPOCHS)

        log = dict()
        log["reward"] = []
        log["lin_vel"] = []
        log["heading"] = []
        log["up_right"] = []

        for epoch in tqdm(range(self.EPOCHS)):
            # sample params from ES and set model params
            solutions = self.solver.ask()
            self.models.set_models_params(solutions)
            total_rewards = torch.zeros(self.POPSIZE, device=self.device)
            cumulative_reward = torch.zeros(self.POPSIZE , device=self.device)
            obs , _ = env.reset()


            log_upright = torch.zeros(self.POPSIZE, device=self.device)
            log_lin_vel = torch.zeros(self.POPSIZE, device=self.device)
            log_heading = torch.zeros(self.POPSIZE, device=self.device)
            log_lin_vel_rew = torch.zeros(self.POPSIZE, device=self.device)
            log_upright_rew = torch.zeros(self.POPSIZE, device=self.device)
            log_heading_rew = torch.zeros(self.POPSIZE, device=self.device)

            # Rollout
            for timse_step in range(self.EPISODE_LENGTH_TRAIN):
                actions = self.models.forward(obs["policy"])
                next_obs, reward, terminated, truncated, extras = env.step(actions)
                cumulative_reward += reward
                done = torch.logical_or(terminated, truncated)
                obs = next_obs
                # Set Objective Function to ES
                total_rewards += reward/self.EPISODE_LENGTH_TRAIN*100

                # Logging
                log_lin_vel += extras["log"]["lin_vel"]
                log_upright += extras["log"]["up_right"]
                log_heading += extras["log"]["heading"]

                log_lin_vel_rew += extras["log"]["lin_vel_reward"]
                log_upright_rew += extras["log"]["up_reward"]
                log_heading_rew += extras["log"]["heading_reward"]

                
            # Update to ES
            total_rewards_cpu = total_rewards.cpu().numpy()

            
            fitlist = list(total_rewards_cpu)
            self.solver.tell(fitlist)
            fit_arr = np.array(fitlist)

            # print('epoch', epoch, 'mean', fit_arr.mean(dtype=np.float64), 
            print('epoch', epoch, 'mean', np.nanmean(fit_arr), 
                  'best', fit_arr.max(), )
            pop_mean_curve[epoch] = fit_arr.mean()
            best_sol_curve[epoch] = fit_arr.max()

            # log
            lin_vel = np.mean((log_lin_vel/self.EPISODE_LENGTH_TRAIN).cpu().numpy())
            heading = np.mean((log_upright/self.EPISODE_LENGTH_TRAIN).cpu().numpy())
            upright = np.mean((log_heading/self.EPISODE_LENGTH_TRAIN).cpu().numpy())

            lin_vel_rew = np.mean((log_lin_vel_rew/self.EPISODE_LENGTH_TRAIN).cpu().numpy())
            heading_rew = np.mean((log_upright_rew/self.EPISODE_LENGTH_TRAIN).cpu().numpy())
            upright_rew = np.mean((log_heading_rew/self.EPISODE_LENGTH_TRAIN).cpu().numpy())

           # Save value
            value_dir = os.path.join(self.dir_path, "value")
            os.makedirs(value_dir, exist_ok=True)
            log["reward"].append(fit_arr.mean())
            log["lin_vel"].append(lin_vel)
            log["heading"].append(heading)
            log["up_right"].append(upright)

            np.save( os.path.join(value_dir, "metrics.npy"),log ,allow_pickle=True )           
             # np.save(os.path.join(value_dir, "lin_vel.npy"),np.stack(log["lin_vel"], axis=0))
            # np.save(os.path.join(value_dir, "heading.npy"),np.stack(log["heading"], axis=0))
            # np.save(os.path.join(value_dir, "up_right.npy"),np.stack(log["up_right"], axis=0))
            # np.save(os.path.join(value_dir, "reward.npy"),np.stack(log["reward"], axis=0))


            
            print(f"Now time : {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")



            # WanDB Log data -------------------------------
            if self.wandb_activate:
                wandb.log({"epoch": epoch,
                            "mean" : np.mean(fitlist),
                            "best" : np.max(fitlist),
                            "worst": np.min(fitlist),
                            "std"  : np.std(fitlist),
                            "reward" : np.mean(cumulative_reward.cpu().numpy()),
                            "lin_vel_value" : np.mean(lin_vel),
                            "heading_value" : np.mean(heading),
                            "up_right" : np.mean(upright),
                            "lin_vel_reward" : np.mean(lin_vel_rew),
                            "heading_reward" : np.mean(heading_rew),
                            "up_right_reward" : np.mean(upright_rew)
                            })
 
            # Save model params and OpenES params
            if (epoch + 1) % self.SAVE_EVERY == 0:
                print('saving..')
                # Create Folder
                save_path = os.path.join(self.dir_path, "model",f"model_{epoch}.pickle" )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Dump file
                pickle.dump((
                    self.solver,
                    copy.deepcopy(self.models),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open(save_path, 'wb'))
        env.close()
        if self.wandb_activate:
            wandb.finish()
        
    
    def run_play(self,env):
    
        for epoch in tqdm(range(self.EPOCHS)):
            # sample params from ES and set model params
            self.models.set_a_model_params(self.train_params)
            obs , _ = env.reset()
            
            # Rollout
            for timse_step in range(self.EPISODE_LENGTH_TRAIN):
                
                actions = self.models.forward(obs['policy'])
                next_obs, reward, terminated, truncated, _ = env.step(actions)
                obs = next_obs
                
            # Update to ES

            # print('epoch', epoch, 'mean', fit_arr.mean(), 
            #       'best', fit_arr.max(), )
        env.close()
        if self.wandb_activate:
            wandb.finish()


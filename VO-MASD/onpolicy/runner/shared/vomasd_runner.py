import time, pickle
from functools import reduce
from tqdm import tqdm

import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
from gym import spaces
from runner.shared.hier_runner import HierRunner

def _t2n(x):
    return x.detach().cpu().numpy()

class VOMASDRunner(HierRunner):
    def __init__(self, config, data_path):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        self.data_path = data_path

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        
        self.c_step = self.all_args.c_step
        self.skill_dim = self.all_args.skill_dim

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.algorithm_name == "vomasd-hier":
            from algorithms.vomasd.vomasd_hier import VOMASDHier as TrainAlgo
            from algorithms.vomasd.vomasd_hier_policy import VOMASDHierPolicy as Policy
            
        else:
            from algorithms.vomasd.vomasd_3d import VOMASD3d as TrainAlgo
            from algorithms.vomasd.vomasd_3d_policy import VOMASD3dPolicy as Policy
        
        skill_space = spaces.Box(low=-1.0, high=1.0, shape=(self.skill_dim, ), dtype=np.float32)

        if self.all_args.map_name in ["3m", "5m", "7m", "10m", "12m"]:
            obs_shapes = {"3m": [100, [2, 5, 21], [3, 5], [1, 4], [1, 5, 21, 3, 0]], "5m": [164, [4, 5, 21], [5, 5], [1, 4], [1, 5, 21, 5, 0]],
                          "10m": [324, [9, 5, 21], [10, 5], [1, 4], [1, 5, 21, 10, 0]], "7m": [228, [6, 5, 21], [7, 5], [1, 4], [1, 5, 21, 7, 0]]}
            state_shapes = {"3m": [117, [2, 8, 21], [3, 8], [1, 4], [1, 7, 21, 3, 0]], "5m": [193, [4, 8, 21], [5, 8], [1, 4], [1, 7, 21, 5, 0]],
                          "10m": [383, [9, 8, 21], [10, 8], [1, 4], [1, 7, 21, 10, 0]], "7m": [269, [6, 8, 21], [7, 8], [1, 4], [1, 7, 21, 7, 0]]}

            if self.all_args.is_med:
                self.pretrain_task_list = ["3m-med", "5m-med"]
            elif self.all_args.is_mixed:
                self.pretrain_task_list = ["3m", "5m", "3m-med", "5m-med"]
            else:
                self.pretrain_task_list = ["3m", "5m"]

            self.max_group_size = 10
            self.max_skill_size = 5
        elif self.all_args.map_name in ["MMM", "MMM2"]:
            obs_shapes = {"MMM": [384, [9, 8, 21], [10, 8], [1, 4], [1, 8, 21, 10, 0]] , "MMM2": [400, [9, 8, 21], [12, 8], [1, 4], [1, 8, 21, 10, 0]]}
            state_shapes = {"MMM": [443, [9, 11, 21], [10, 11], [1, 4], [1, 10, 21, 10, 0]], "MMM2": [465, [9, 11, 21], [12, 11], [1, 4], [1, 10, 21, 10, 0]]}

            if self.all_args.is_med:
                self.pretrain_task_list = ["MMM-med", "MMM2-med"]
            elif self.all_args.is_mixed:
                self.pretrain_task_list = ["MMM", "MMM2", "MMM-med", "MMM2-med"]
            else:
                self.pretrain_task_list = ["MMM", "MMM2"]
                
            self.max_group_size = 10
            self.max_skill_size = 10
        
        elif self.all_args.map_name in ["terran_3", "terran_5", "terran_7"]:
            obs_shapes = {"terran_3": [116, [2, 8, 21], [3, 8], [1, 4], [1, 6, 21, 3, 0]] , "terran_5": [192, [4, 8, 21], [5, 8], [1, 4], [1, 6, 21, 5, 0]],
                          "terran_7": [268, [6, 8, 21], [7, 8], [1, 4], [1, 6, 21, 7, 0]]}
            state_shapes = {"terran_3": [109, [2, 7, 21], [3, 6], [1, 4], [1, 7, 21, 3, 0]], "terran_5": [179, [4, 7, 21], [5, 6], [1, 4], [1, 7, 21, 5, 0]],
                            "terran_7": [249, [6, 7, 21], [7, 6], [1, 4], [1, 7, 21, 7, 0]]}
            self.pretrain_task_list = ["terran_3", "terran_5"]
            self.max_group_size = 7
            self.max_skill_size = 5
            
        else:
            raise NotImplementedError
        
        self.share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            self.share_observation_space,
                            self.envs.action_space[0],
                            obs_shapes,
                            state_shapes,
                            skill_space,
                            self.max_group_size,
                            self.max_skill_size,
                            device = self.device) 

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # if self.model_dir is not None:
        #     self.trainer.restore()

        # buffer for training the online policy
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space[0],
                                         self.share_observation_space,
                                         skill_space)
    
    @torch.no_grad()
    def execute_low_level_policy(self, skills, step):
        states = self.buffer.share_obs[step]
        states = np.expand_dims(states, axis=1)
        skills = np.expand_dims(skills, axis=1)
        skills = self.policy.forward_code(states, skills, self.all_args.map_name, test_mode=True, second_stage=True).squeeze(dim=1)
        # print(skills.shape) # torch.Size([1, 3, 10])
        skills = skills.reshape(self.n_rollout_threads * self.num_agents, -1)
        cumu_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for t in range(self.c_step):
            real_actions, self.low_rnn_states = self.trainer.policy.get_real_actions(skills, self.low_obses, \
                                                                                     self.low_rnn_states, self.low_masks, \
                                                                                     self.low_avail_actions) # (3 * 8, 1), (3 * 8, 1, 64)
            real_actions = np.array(np.split(_t2n(real_actions), self.n_rollout_threads)) # (8, 3, 1)
            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(real_actions)
            # (1, 3, 100) (1, 3, 117) (1, 3, 1) (1, 3) (1, 3) (1, 3, 21)
            cumu_rewards += rewards
            self.low_obses = obs.copy().reshape(self.n_rollout_threads * self.num_agents, -1)
            self.low_avail_actions = available_actions.copy().reshape(self.n_rollout_threads * self.num_agents, -1)

            dones_env = np.all(dones, axis=1)
            self.low_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.low_masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.low_masks = self.low_masks.reshape(self.n_rollout_threads * self.num_agents, -1)

            # print(obs.shape, share_obs.shape, rewards.shape, dones.shape, infos.shape, available_actions.shape)
            # print(dones_env, (dones_env == True).sum(), rewards, cumu_rewards) # [False] 0

            if (dones_env == True).sum() > 0:
                break

        return obs, share_obs, cumu_rewards, dones, infos
    
    @torch.no_grad()
    def eval_execute_low_level_policy(self, eval_skills, eval_states):
        eval_states = np.expand_dims(eval_states, axis=1)
        eval_skills = np.expand_dims(eval_skills, axis=1)
        
        eval_skills = self.policy.forward_code(eval_states, eval_skills, self.all_args.map_name, \
                                               test_mode=True, second_stage=True).squeeze(dim=1)
        eval_skills = eval_skills.reshape(self.n_eval_rollout_threads * self.num_agents, -1)

        eval_cumu_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for t in range(self.c_step):
            eval_real_actions, self.low_eval_rnn_states = self.trainer.policy.get_real_actions(eval_skills, self.low_eval_obses, \
                                                                                               self.low_eval_rnn_states, self.low_eval_masks, \
                                                                                               self.low_eval_avail_actions) 
            eval_real_actions = np.array(np.split(_t2n(eval_real_actions), self.n_eval_rollout_threads)) 
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_real_actions)
            # (1, 3, 100) (1, 3, 117) (1, 3, 1) (1, 3) (1, 3) (1, 3, 21)
            eval_cumu_rewards += eval_rewards
            self.low_eval_obses = eval_obs.copy().reshape(self.n_eval_rollout_threads * self.num_agents, -1)
            self.low_eval_avail_actions = eval_available_actions.copy().reshape(self.n_eval_rollout_threads * self.num_agents, -1)

            eval_dones_env = np.all(eval_dones, axis=1)
            self.low_eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.low_eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.low_eval_masks = self.low_eval_masks.reshape(self.n_eval_rollout_threads * self.num_agents, -1)

            # print(obs.shape, share_obs.shape, rewards.shape, dones.shape, infos.shape, available_actions.shape)
            # print(dones_env, (dones_env == True).sum(), rewards, cumu_rewards) # [False] 0

            if (eval_dones_env == True).sum() > 0:
                break

        return eval_obs, eval_share_obs, eval_cumu_rewards, eval_dones, eval_infos

    def pretrain_sequential(self): # offline skill discovery
        train_tasks = self.pretrain_task_list
        task2offlinedata = {}
        print(train_tasks)
        for task in train_tasks:
            with open(str(self.data_path) + '/' + str(task) + '/data.pkl', 'rb') as f:
                task2offlinedata[task] = pickle.load(f)
        
        main_args = self.all_args
        ########## start training ##########
        t_max = main_args.pretrain_steps
        batch_size_train = main_args.pretrain_batch_size # 32
        if self.all_args.map_name in ["MMM", "MMM2"]:
            batch_size_train = 16
            
        for t_env in tqdm(range(t_max)):
            # shuffle tasks
            np.random.shuffle(train_tasks) # train_tasks is pretrain_tasks when pretrain is true
            # multi-task training
            for task in train_tasks:
                data_size = task2offlinedata[task]["state"].shape[0]
                indices = np.random.permutation(data_size)[:batch_size_train]
                # print(data_size, indices)
                episode_sample = {}
                for key in task2offlinedata[task]:
                    episode_sample[key] = task2offlinedata[task][key][indices] # 32 trajectories
                    # print(episode_sample[key].shape) # (32, 27, 3, 117), (32, 27, 3, 100)

                if '-' in task:
                    task = task[:-4]
                loss_info = self.trainer.pretrain(episode_sample, task) # skill discovery, only calculate the gradient
            
            self.trainer.pretrain_update() 

            # train the grouper through RL
            np.random.shuffle(train_tasks)
            for task in train_tasks:
                data_size = task2offlinedata[task]["state"].shape[0]
                if self.all_args.map_name in ["MMM", "MMM2"]:
                    indices = np.random.permutation(data_size)[:batch_size_train]
                else:
                    indices = np.random.permutation(data_size)[:batch_size_train * 2] # big batches for RL training
                # print(data_size, indices)
                episode_sample = {}
                for key in task2offlinedata[task]:
                    episode_sample[key] = task2offlinedata[task][key][indices] 

                if '-' in task:
                    task = task[:-4]
                rl_loss_info = self.trainer.pretrain_rl(episode_sample, task) # skill discovery, only calculate the gradient

            if t_env % self.log_interval == 0:
                self.log_pretrain(loss_info, t_env)
                self.log_pretrain(rl_loss_info, t_env)
                print(loss_info, "\n", rl_loss_info)
        
        self.save_pretrain()
    

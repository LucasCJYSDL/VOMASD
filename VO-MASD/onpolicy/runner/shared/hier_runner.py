import time, pickle
from functools import reduce
from tqdm import tqdm

import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
from gym import spaces
from runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class HierRunner(Runner):
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

        if self.algorithm_name == "odis":
            from algorithms.odis.odis import ODIS as TrainAlgo
            from algorithms.odis.odis_policy import ODISPolicy as Policy

            skill_space = spaces.Discrete(self.skill_dim)
        elif self.algorithm_name == "vomasd-single":
            from algorithms.vomasd.vomasd_single import VOMASDSingle as TrainAlgo
            from algorithms.vomasd.vomasd_single_policy import VOMASDSinglePolicy as Policy

            skill_space = spaces.Box(low=-1.0, high=1.0, shape=(self.skill_dim, ), dtype=np.float32)
        else:
            raise NotImplementedError
        
        if self.all_args.map_name in ["3m", "5m", "7m", "10m", "12m"]:
            obs_shapes = {"3m": [100, [2, 5, 21], [3, 5], [1, 4], [1, 5, 21, 3, 0]], "5m": [164, [4, 5, 21], [5, 5], [1, 4], [1, 5, 21, 5, 0]],
                          "10m": [324, [9, 5, 21], [10, 5], [1, 4], [1, 5, 21, 10, 0]], "7m": [228, [6, 5, 21], [7, 5], [1, 4], [1, 5, 21, 7, 0]]}
            state_shapes = {"3m": [117, [2, 8, 21], [3, 8], [1, 4], [1, 7, 21, 3, 0]], "5m": [193, [4, 8, 21], [5, 8], [1, 4], [1, 7, 21, 5, 0]],
                          "10m": [383, [9, 8, 21], [10, 8], [1, 4], [1, 7, 21, 10, 0]], "7m": [269, [6, 8, 21], [7, 8], [1, 4], [1, 7, 21, 7, 0]]}
            self.pretrain_task_list = ["3m", "5m"]

        elif self.all_args.map_name in ["MMM", "MMM2"]:
            obs_shapes = {"MMM": [384, [9, 8, 21], [10, 8], [1, 4], [1, 8, 21, 10, 0]] , "MMM2": [400, [9, 8, 21], [12, 8], [1, 4], [1, 8, 21, 10, 0]]}
            state_shapes = {"MMM": [443, [9, 11, 21], [10, 11], [1, 4], [1, 10, 21, 10, 0]], "MMM2": [465, [9, 11, 21], [12, 11], [1, 4], [1, 10, 21, 10, 0]]}
            self.pretrain_task_list = ["MMM", "MMM2"]

        elif self.all_args.map_name in ["terran_3", "terran_5", "terran_7"]:
            obs_shapes = {"terran_3": [116, [2, 8, 21], [3, 8], [1, 4], [1, 6, 21, 3, 0]] , "terran_5": [192, [4, 8, 21], [5, 8], [1, 4], [1, 6, 21, 5, 0]],
                          "terran_7": [268, [6, 8, 21], [7, 8], [1, 4], [1, 6, 21, 7, 0]]}
            state_shapes = {"terran_3": [109, [2, 7, 21], [3, 6], [1, 4], [1, 7, 21, 3, 0]], "terran_5": [179, [4, 7, 21], [5, 6], [1, 4], [1, 7, 21, 5, 0]],
                            "terran_7": [249, [6, 7, 21], [7, 6], [1, 4], [1, 7, 21, 7, 0]]}
            self.pretrain_task_list = ["terran_3", "terran_5"]

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
    def collect(self, step):
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            avail_actions = None
        else:
            avail_actions = np.concatenate(self.buffer.available_actions[step])
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            available_actions=avail_actions)
        
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    @torch.no_grad()
    def execute_low_level_policy(self, skills, step):
        skills = self.policy.forward_code(skills)
        # print(skills.shape) # torch.Size([1, 3, 5])

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

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions=None)

    def run(self):
        # offline skill discovery stage
        self.pretrain_sequential()
    
        # online learning stage
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # print(values.shape, actions.shape, action_log_probs.shape, rnn_states.shape)
                # (8, 3, 1) (8, 3, 1/5) (8, 3, 1) (8, 3, 1, 64)
                # Obser reward and next obs
                # obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                obs, share_obs, rewards, dones, infos = self.execute_low_level_policy(actions, step)

                if self.all_args.use_sparse_reward:
                    rewards = np.zeros_like(rewards)
                    dones_env = np.all(dones, axis=1) # (n_roll, )
                    # print("1: ", rewards.shape, dones_env.shape)
       
                    for t in range(self.n_rollout_threads):
                        if dones_env[t]:
                            if infos[t][0].get("won", False):
                                rewards[t] = np.ones_like(rewards[t]) * 20.0
                    # print("2: ", rewards, rewards.shape)

                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)
                
            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads      
            print(self.episode_length, self.n_rollout_threads, episode, total_num_steps)     
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2" or self.env_name == "StarCraft2_v2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))

                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, low_eval_avail_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        self.low_eval_obses = eval_obs.copy().reshape(self.n_eval_rollout_threads * self.num_agents, -1)
        self.low_eval_rnn_states = np.zeros((self.n_eval_rollout_threads * self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.low_eval_masks = np.ones((self.n_eval_rollout_threads * self.num_agents, 1), dtype=np.float32)
        self.low_eval_avail_actions = low_eval_avail_actions.reshape(self.n_eval_rollout_threads * self.num_agents, -1)

        if self.buffer.available_actions is None:
            eval_available_actions = None
        else:
            eval_available_actions = np.ones((self.n_eval_rollout_threads, self.num_agents, self.skill_dim), dtype=np.float32)
            eval_available_actions = np.concatenate(eval_available_actions)
        

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        available_actions=eval_available_actions,
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_execute_low_level_policy(eval_actions, eval_share_obs)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                # if self.use_wandb:
                #     wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                # else:
                self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
    
    @torch.no_grad()
    def eval_execute_low_level_policy(self, eval_skills, eval_states):
        
        eval_skills = self.policy.forward_code(eval_skills)
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
    
    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        # self.buffer.available_actions[0] = available_actions.copy()
        # now actions are actually skills, so available_actions are all ones
        # for low-level rollout
        self.low_obses = obs.copy().reshape(self.n_rollout_threads * self.num_agents, -1)
        self.low_rnn_states = np.zeros((self.n_rollout_threads * self.num_agents, self.recurrent_N,\
                                         self.hidden_size), dtype=np.float32)
        self.low_masks = np.ones((self.n_rollout_threads * self.num_agents, 1), dtype=np.float32)
        self.low_avail_actions = available_actions.copy().reshape(self.n_rollout_threads * self.num_agents, -1)
    
    def pretrain_sequential(self): # offline skill discovery
        train_tasks = self.pretrain_task_list
        task2offlinedata = {}
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

                loss_info = self.trainer.pretrain(episode_sample, task) # skill discovery, only calculate the gradient
            
            self.trainer.pretrain_update() 

            if t_env % self.log_interval == 0:
                self.log_pretrain(loss_info, t_env)
                # print(loss_info)
        
        self.save_pretrain()
    
    def log_pretrain(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    def save_pretrain(self):
        self.policy.save_pretrain(self.save_dir)

    def restore_pretrain(self):
        self.policy.restore_pretrain(self.model_dir)
    
    def save(self):
        super(HierRunner, self).save()
        self.policy.save(self.save_dir)
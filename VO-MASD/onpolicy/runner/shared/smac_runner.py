import time
# import wandb
import numpy as np
from functools import reduce
import torch
from runner.shared.base_runner import Runner

from utils.shared_buffer import SharedReplayBuffer
from tqdm import tqdm
import pickle

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
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
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                if self.all_args.use_sparse_reward:
                    rewards = np.zeros_like(rewards)
                    dones_env = np.all(dones, axis=1) # (n_roll, )
                    # print("1: ", rewards.shape, dones_env.shape)
                    for t in range(self.n_rollout_threads):
                        if dones_env[t]:
                            if infos[t][0].get("won", False):
                                rewards[t] = np.ones_like(rewards[t]) * 20.0
                    # print("2: ", rewards, rewards.shape)
                    

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # print(self.episode_length, self.n_rollout_threads, episode, total_num_steps)

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
                    # if self.use_wandb:
                    #     wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    # else:
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)
            
            if incre_win_rate > 0.95:
                self.save_with(incre_win_rate)
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
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
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            # if self.use_wandb:
            #     wandb.log({k: v}, step=total_num_steps)
            # else:
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
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
    def data_collect(self, step):
        self.trainer.prep_rollout()
        
        action, rnn_state = self.trainer.policy.act(np.concatenate(self.buffer.obs[step]),
                                                                np.concatenate(self.buffer.rnn_states[step]),
                                                                np.concatenate(self.buffer.masks[step]),
                                                                np.concatenate(self.buffer.available_actions[step]),
                                                                deterministic=True)
        # [self.envs, agents, dim]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))

        return actions, rnn_states

    def data_insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, actions, rnn_states= data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.data_insert(share_obs, obs, rnn_states, actions, rewards, masks, bad_masks, active_masks, available_actions)

    @torch.no_grad()
    def collect_data(self, load_path, down_path, episode_number):
        policy_actor_state_dict = torch.load(str(load_path) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        data_frame = {"state": [], "obs": [], "action": [], "reward": [], "mask": [], "avail_action": []}
        
        aver_win_rate = 0
        for i in tqdm(range(episode_number)):
            self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.envs.observation_space[0], 
                                             self.share_observation_space, self.envs.action_space[0])
            
            self.warmup()
            for step in range(self.episode_length):
                # Sample actions
                actions, rnn_states = self.data_collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, actions, rnn_states
                
                # insert data into buffer
                self.data_insert(data)
                # print("0:", infos)
                # [[{'battles_won': 20, 'battles_game': 21, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}
                #   {'battles_won': 20, 'battles_game': 21, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}
                #   {'battles_won': 20, 'battles_game': 21, 'battles_draw': 0, 'restarts': 0, 'bad_transition': False, 'won': False}]]
            
            data_dict = self.buffer.get_data()
            for k in data_frame:
                data_frame[k].append(data_dict[k])
                # print(data_dict[k].shape) # (401, 1, 3, 81), (401, 1, 3, 64), (400, 1, 3, 1), (400, 1, 3, 1), (401, 1, 3, 1), (401, 1, 3, 9)
            
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
            aver_win_rate += incre_win_rate

            last_battles_game = battles_game
            last_battles_won = battles_won
        
        print("Average win rate is {}.".format(aver_win_rate / float(episode_number)))
        for k in data_frame:
            data_frame[k] = np.concatenate(data_frame[k], axis=1)
            print(data_frame[k].shape) # (401, 10, 3, 81), ......

        # post process
        masks = data_frame["mask"]
        seq_len, data_len = masks.shape[0], masks.shape[1]
        segments = []
        max_traj_len = 0
        traj_num = 0
       
        for i in range(data_len): # we discard the last incomplete trajectory
            temp_segments = [0]
            for j in range(seq_len):
                if masks[j][i].sum() < 1:
                    temp_segments.append(j)
                    traj_num += 1
                    if max_traj_len < temp_segments[-1] - temp_segments[-2]:
                        max_traj_len = temp_segments[-1] - temp_segments[-2]
            segments.append(temp_segments)
        
        # print(max_traj_len, traj_num, segments)
        num_agents = masks.shape[2]
        share_obs_shape, obs_shape, act_shape = data_frame["state"].shape[-1], data_frame["obs"].shape[-1], data_frame["avail_action"].shape[-1]

        b_share_obs = np.zeros((traj_num, max_traj_len, num_agents, share_obs_shape), dtype=np.float32)
        b_obs = np.zeros((traj_num, max_traj_len, num_agents, obs_shape), dtype=np.float32)
        b_available_actions = np.ones((traj_num, max_traj_len, num_agents, act_shape), dtype=np.float32)
        b_actions = np.zeros((traj_num, max_traj_len, num_agents, 1), dtype=np.float32)
        b_rewards = np.zeros((traj_num, max_traj_len, num_agents, 1), dtype=np.float32)
        b_filled = np.zeros((traj_num, max_traj_len, num_agents, 1), dtype=np.float32)

        new_data_frame = {"state": b_share_obs, "obs": b_obs, "action": b_actions, 
                          "reward": b_rewards, "filled": b_filled, "avail_action": b_available_actions}
        
        copy_list = ["state", "obs", "action", "reward", "avail_action"]
        traj_idx = 0
        for i in range(data_len):
            temp_segments = segments[i]
            temp_len = len(temp_segments)
            for j in range(temp_len - 1):
                s_id = temp_segments[j]
                e_id = temp_segments[j+1]
                cur_traj_len = e_id - s_id
                for k in copy_list:
                    new_data_frame[k][traj_idx][:cur_traj_len] = data_frame[k][s_id:e_id, i]
                new_data_frame["filled"][traj_idx][:cur_traj_len] = 1.0
                traj_idx += 1

        for k in new_data_frame:
            print(new_data_frame[k].shape) # (160, 24, 3, 117), (160, 24, 3, 100), ...
        with open(str(down_path) + '/data.pkl', 'wb') as f:
            pickle.dump(new_data_frame, f)

import time
import wandb
import numpy as np
from functools import reduce
import torch
from hmasd.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class AliceBobRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(AliceBobRunner, self).__init__(config)

    def run2(self):
        for episode in range(1):
            self.eval(episode)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = np.zeros(self.n_rollout_threads, dtype=np.float32)
        train_episode_steps = np.zeros(self.n_rollout_threads, dtype=np.float32)
        done_episodes_rewards = []
        train_episode_team_intri_rewards = np.zeros(self.n_rollout_threads, dtype=np.float32)
        done_episodes_team_intri_rewards = []
        train_episode_indi_intri_rewards = np.zeros(self.n_rollout_threads, dtype=np.float32)
        done_episodes_indi_intri_rewards = []
        battles_game_num = 0
        battles_won_num = 0
        battles_key0 = 0
        battles_key1 = 0
        battles_step = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.h_trainer.policy.lr_decay(episode, episodes)
                self.l_trainer.policy.lr_decay(episode, episodes)
                self.d_trainer.policy.lr_decay(episode, episodes)

            h_reward = 0

            for step in range(self.episode_length):
                if step % self.skill_interval == 0:
                    h_values, h_actions, h_action_log_probs = self.h_collect(step // self.skill_interval)
                    # (n_roll, n_agent+1, 1), (n_roll, n_agent+1, skill_num), (n_roll, n_agent+1, skill_num)
                    team_skill = h_actions[:, 0] 
                    team_skill = np.expand_dims(team_skill, 1).repeat(self.num_agents, 1)  # (n_roll, n_agent, skill_num)
                    indi_skill = h_actions[:, 1:] # (n_roll, n_agent, skill_num)
        
                self.l_buffer.team_skill[step] = team_skill.copy()
                self.l_buffer.indi_skill[step] = indi_skill.copy()

                # Sample actions
                l_values, l_actions, l_action_log_probs, rnn_states, rnn_states_critic = self.l_collect(step)
                # (n_roll, n_agent, 1), (n_roll, n_agent, act_num), (n_roll, n_agent, act_num), (n_roll, n_agent, recurrent_N, hidden_size), (n_roll, n_agent, recurrent_N, hidden_size)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(l_actions)
                # obs: (n_roll, n_agent, obs_dim)
                # share_obs: (n_roll, n_agent, state_dim)
                # rewards: (n_roll, n_agent, 1)
                # dones: (n_roll, n_agent)
                # infos: tuple len=n_roll
                # available_actions: (n_roll, n_agent, n_action)  

                dones_env = np.all(dones, axis=1) # (n_roll, )
                reward_env = np.mean(rewards, axis=1).flatten() # (n_roll, )
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    train_episode_steps[t] += 1
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        if infos[t][0]["battle_won"]:
                            battles_won_num += 1
                        if infos[t][0]["key0"]:
                            battles_key0 += 1
                        if infos[t][0]["key1"]:
                            battles_key1 += 1
                        battles_step += train_episode_steps[t]
                        train_episode_steps[t] = 0
                        battles_game_num += 1

                h_reward = h_reward + rewards

                # store high-level policy data
                if step % self.skill_interval == self.skill_interval - 1:
                    h_rewards = np.expand_dims(h_reward.mean(axis=1), 1).repeat(self.num_agents + 1, 1) # (n_roll, n_agent+1, 1)
                    h_data = obs, share_obs, h_rewards, dones, h_values, h_actions, h_action_log_probs
                    self.h_insert(h_data)
                    h_reward = 0

                # store discriminator data
                if step == 0:
                    rnn_team_states, rnn_indi_states = self.state_skill.rnn_team_states[0].copy(), self.state_skill.rnn_indi_states[0].copy()
                d_data = obs, share_obs, team_skill, indi_skill, dones, rnn_team_states, rnn_indi_states
                self.d_insert(d_data, step)

                # compute intrinsic reward
                team_intri_rew, indi_intri_rew, rnn_team_states, rnn_indi_states = self.d_collect(step)
                # (n_roll, n_agent, 1), (n_roll, n_agent, 1), (n_roll, n_agent, recurrent_N, hidden_size), (n_roll, n_agent, recurrent_N, hidden_size)
                  
                # store low-level policy data
                l_rewards = self.all_args.lambda_env * rewards + self.all_args.lambda_team * team_intri_rew + self.all_args.lambda_indi * indi_intri_rew

                l_data = obs, share_obs, l_rewards, dones, infos, available_actions, \
                         l_values, l_actions, l_action_log_probs, rnn_states, rnn_states_critic 
                self.l_insert(l_data)

                # the ratio of intrinsic reward in total reward                
                dones_env = np.all(dones, axis=1) # (n_roll, )
                reward_team_intri = np.mean(team_intri_rew, axis=1).flatten() # (n_roll, )
                reward_indi_intri = np.mean(indi_intri_rew, axis=1).flatten() # (n_roll, )
                train_episode_team_intri_rewards += reward_team_intri
                train_episode_indi_intri_rewards += reward_indi_intri
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_team_intri_rewards.append(train_episode_team_intri_rewards[t])
                        train_episode_team_intri_rewards[t] = 0
                        done_episodes_indi_intri_rewards.append(train_episode_indi_intri_rewards[t])
                        train_episode_indi_intri_rewards[t] = 0

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    ave_train_episode_rewards = np.mean(done_episodes_rewards)
                    ave_train_episode_team_intri_rewards = np.mean(done_episodes_team_intri_rewards)
                    ave_train_episode_indi_intri_rewards = np.mean(done_episodes_indi_intri_rewards)
                    ave_train_win_rate = battles_won_num / battles_game_num
                    ave_train_key0 = battles_key0 / battles_game_num
                    ave_train_key1 = battles_key1 / battles_game_num
                    ave_train_episode_step = battles_step / battles_game_num
                    done_episodes_rewards = []
                    done_episodes_team_intri_rewards = []
                    done_episodes_indi_intri_rewards = []
                    battles_step = 0
                    battles_won_num = 0
                    battles_key0 = 0
                    battles_key1 = 0
                    battles_game_num = 0 

                    print('ave_train_episode_env_rewards', ave_train_episode_rewards, 
                    'ave_train_episode_team_intri_rewards', ave_train_episode_team_intri_rewards, 
                    'ave_train_episode_indi_intri_rewards', ave_train_episode_indi_intri_rewards, 
                    'ave_train_win_rate', ave_train_win_rate, 
                    'ave_train_key0', ave_train_key0, 
                    'ave_train_key1', ave_train_key1, 
                    'ave_train_episode_step', ave_train_episode_step)
                    if self.use_wandb:
                        wandb.log({"ave_train_episode_env_rewards": ave_train_episode_rewards}, step=total_num_steps)
                        wandb.log({"ave_train_episode_team_intri_rewards": ave_train_episode_team_intri_rewards}, step=total_num_steps)
                        wandb.log({"ave_train_episode_indi_intri_rewards": ave_train_episode_indi_intri_rewards}, step=total_num_steps)
                        wandb.log({"ave_train_win_rate": ave_train_win_rate}, step=total_num_steps)
                        wandb.log({"ave_train_key0": ave_train_key0}, step=total_num_steps)
                        wandb.log({"ave_train_key1": ave_train_key1}, step=total_num_steps)
                        wandb.log({"ave_train_episode_step": ave_train_episode_step}, step=total_num_steps)
                        wandb.log({"h_entropy_coef": self.h_trainer.entropy_coef}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("ave_train_episode_env_rewards", {"ave_train_episode_env_rewards": ave_train_episode_rewards}, total_num_steps)
                        self.writter.add_scalars("ave_train_episode_team_intri_rewards", {"ave_train_episode_team_intri_rewards": ave_train_episode_team_intri_rewards}, total_num_steps)
                        self.writter.add_scalars("ave_train_episode_indi_intri_rewards", {"ave_train_episode_indi_intri_rewards": ave_train_episode_indi_intri_rewards}, total_num_steps)
                        self.writter.add_scalars("ave_train_win_rate", {"ave_train_win_rate": ave_train_win_rate}, total_num_steps)
                        self.writter.add_scalars("ave_train_key0", {"ave_train_key0": ave_train_key0}, total_num_steps)
                        self.writter.add_scalars("ave_train_key1", {"ave_train_key1": ave_train_key1}, total_num_steps)
                        self.writter.add_scalars("ave_train_episode_step", {"ave_train_episode_step": ave_train_episode_step}, total_num_steps)
                        self.writter.add_scalars("h_entropy_coef", {"h_entropy_coef": self.h_trainer.entropy_coef}, total_num_steps)
                        
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # (n_roll, n_agent, obs_dim), (n_roll, n_agent, state_dim), (n_roll, n_agent, n_action)

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.h_buffer.share_obs[0] = share_obs.copy()
        self.h_buffer.obs[0] = obs.copy()

        self.l_buffer.share_obs[0] = share_obs.copy()
        self.l_buffer.obs[0] = obs.copy()
        self.l_buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def h_collect(self, step):
        self.h_trainer.prep_rollout()
        value, action, action_log_prob = self.h_trainer.policy.get_actions(np.concatenate(self.h_buffer.share_obs[step]),
                                                                           np.concatenate(self.h_buffer.obs[step]))
        # input: (n_roll*n_agent, state_dim), (n_roll*n_agent, obs_dim)
        # output: (n_roll*(n_agent+1), 1), (n_roll*(n_agent+1), act_num), (n_roll*(n_agent+1), act_num)

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads)) # (n_roll, n_agent+1, 1)
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads)) # (n_roll, n_agent+1, act_num)
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads)) # (n_roll, n_agent+1, act_num)

        return values, actions, action_log_probs
        # (n_roll, n_agent+1, 1), (n_roll, n_agent+1, act_num), (n_roll, n_agent+1, act_num)

    @torch.no_grad()
    def l_collect(self, step):
        self.l_trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.l_trainer.policy.get_actions(np.concatenate(self.l_buffer.share_obs[step]),
                                                np.concatenate(self.l_buffer.obs[step]),
                                                np.concatenate(self.l_buffer.team_skill[step]),
                                                np.concatenate(self.l_buffer.indi_skill[step]),
                                                np.concatenate(self.l_buffer.rnn_states[step]),
                                                np.concatenate(self.l_buffer.rnn_states_critic[step]),
                                                np.concatenate(self.l_buffer.masks[step]),
                                                np.concatenate(self.l_buffer.available_actions[step]))
        # (n_roll*n_agent, 1), (n_roll*n_agent, act_num), (n_roll*n_agent, act_num), (n_roll*n_agent, recurrent_N, hidden_size), (n_roll*n_agent, recurrent_N, hidden_size)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads)) # (n_roll, n_agent, 1)
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads)) # (n_roll, n_agent, act_num)
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads)) # (n_roll, n_agent, act_num)
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads)) # (n_roll, n_agent, recurrent_N, hidden_size)
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads)) # (n_roll, n_agent, recurrent_N, hidden_size)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
        # (n_roll, n_agent, 1), (n_roll, n_agent, act_num), (n_roll, n_agent, act_num), (n_roll, n_agent, recurrent_N, hidden_size), (n_roll, n_agent, recurrent_N, hidden_size)

    @torch.no_grad()
    def d_collect(self, step):
        self.d_trainer.prep_rollout()
        team_intri_rew, indi_intri_rew, rnn_team_states, rnn_indi_states \
            = self.d_trainer.policy.get_intrinsic_reward(np.concatenate(self.state_skill.share_obs[step]),
                                                         np.concatenate(self.state_skill.obs[step]),
                                                         np.concatenate(self.state_skill.team_skill[step]),
                                                         np.concatenate(self.state_skill.indi_skill[step]),
                                                         np.concatenate(self.state_skill.rnn_team_states[step]),
                                                         np.concatenate(self.state_skill.rnn_indi_states[step]),
                                                         np.concatenate(self.state_skill.masks[step]))
        # (n_roll*n_agent, 1), (n_roll*n_agent, 1), (n_roll*n_agent, recurrent_N, hidden_size), (n_roll*n_agent, recurrent_N, hidden_size)
        # [self.envs, agents, dim]
        team_intri_rew = np.array(np.split(_t2n(team_intri_rew), self.n_rollout_threads)) # (n_roll, n_agent, 1)
        indi_intri_rew = np.array(np.split(_t2n(indi_intri_rew), self.n_rollout_threads)) # (n_roll, n_agent, 1)
        rnn_team_states = np.array(np.split(_t2n(rnn_team_states), self.n_rollout_threads)) # (n_roll, n_agent, recurrent_N, hidden_size)
        rnn_indi_states = np.array(np.split(_t2n(rnn_indi_states), self.n_rollout_threads)) # (n_roll, n_agent, recurrent_N, hidden_size)

        return team_intri_rew, indi_intri_rew, rnn_team_states, rnn_indi_states
        # (n_roll, n_agent, 1), (n_roll, n_agent, 1), (n_roll, n_agent, recurrent_N, hidden_size), (n_roll, n_agent, recurrent_N, hidden_size)

    def h_insert(self, data):
        obs, share_obs, rewards, dones, values, actions, action_log_probs = data
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents + 1, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents + 1, 1), dtype=np.float32)
        if not self.use_centralized_V:
            share_obs = obs
        self.h_buffer.insert(share_obs, obs, actions, action_log_probs, values, rewards, masks)

    def l_insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = None
        
        if not self.use_centralized_V:
            share_obs = obs

        self.l_buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)
        
    def d_insert(self, data, step):
        obs, share_obs, team_skill, indi_skill, dones, rnn_team_states, rnn_indi_states = data

        if step % self.skill_interval == 0:
            dones = np.ones_like(dones)

        dones_env = np.all(dones, axis=1) # (n_roll, )

        rnn_team_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_indi_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        if not self.use_centralized_V:
            share_obs = obs

        self.state_skill.insert(share_obs, obs, team_skill, indi_skill, rnn_team_states, rnn_indi_states, masks)

    def log_train(self, train_infos, total_num_steps):
        train_infos["h_average_step_rewards"] = np.mean(self.h_buffer.rewards)
        train_infos["l_average_step_rewards"] = np.mean(self.l_buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        assert self.n_eval_rollout_threads == 1
        eval_battles_won = 0
        eval_battles_key0 = 0
        eval_battles_key1 = 0
        eval_battles_step = 0
        eval_episode = self.all_args.eval_episodes

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        for _ in range(eval_episode):
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            step = 0
            while True:
                if step % self.skill_interval == 0:
                    self.h_trainer.prep_rollout()
                    h_actions = self.h_trainer.policy.act(np.concatenate(eval_share_obs),
                                                          np.concatenate(eval_obs),
                                                          deterministic=True) # (n_roll*(n_agent+1), skill_num)
                    h_actions = np.array(np.split(_t2n(h_actions), self.n_eval_rollout_threads)) # (n_roll, n_agent+1, skill_num)
                    team_skill = h_actions[:, 0] 
                    team_skill = np.expand_dims(team_skill, 1).repeat(self.num_agents, 1)  # (n_roll, n_agent, skill_num)
                    indi_skill = h_actions[:, 1:] # (n_roll, n_agent, skill_num)

                self.l_trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    self.l_trainer.policy.act(np.concatenate(eval_obs),
                                              np.concatenate(team_skill),
                                              np.concatenate(indi_skill),
                                              np.concatenate(eval_rnn_states),
                                              np.concatenate(eval_masks),
                                              np.concatenate(eval_available_actions),
                                              deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads)) # (n_roll, n_agent, act_num)
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)) # (n_roll, n_agent, recurrent_N, hidden_size)
                
                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
                # eval_obs: (n_roll, n_agent, obs_dim)
                # eval_share_obs: (n_roll, n_agent, state_dim)
                # eval_rewards: (n_roll, n_agent, 1)
                # eval_dones: (n_roll, n_agent)
                # eval_infos: tuple len=n_roll
                # eval_available_actions: (n_roll, n_agent, n_action)

                one_episode_rewards.append(eval_rewards) # (eplen, n_roll, n_agent, 1)
                step += 1

                eval_dones_env = np.all(eval_dones, axis=1) # (n_roll, )

                if eval_dones_env[0]:
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[0][0]['battle_won']:
                        eval_battles_won += 1
                    if eval_infos[0][0]['key0']:
                        eval_battles_key0 += 1
                    if eval_infos[0][0]['key1']:
                        eval_battles_key1 += 1
                    eval_battles_step += step
                    break

        # self.eval_envs.save_replay()
        eval_episode_rewards = np.array(eval_episode_rewards) # (n_episode, n_roll, n_agent, 1)
        eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
        self.log_env(eval_env_infos, total_num_steps)
        eval_win_rate = eval_battles_won/eval_episode
        eval_key0 = eval_battles_key0/eval_episode
        eval_key1 = eval_battles_key1/eval_episode
        eval_ave_step = eval_battles_step/eval_episode
        print("eval win rate is {}.".format(eval_win_rate), "eval ave step is {}.".format(eval_ave_step), "eval key0 rate is {}.".format(eval_key0), "eval key1 rate is {}.".format(eval_key1))
        if self.use_wandb:
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_key0": eval_key0}, step=total_num_steps)
            wandb.log({"eval_key1": eval_key1}, step=total_num_steps)
            wandb.log({"eval_ave_step": eval_ave_step}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
            self.writter.add_scalars("eval_key0", {"eval_key0": eval_key0}, total_num_steps)
            self.writter.add_scalars("eval_key1", {"eval_key1": eval_key1}, total_num_steps)
            self.writter.add_scalars("eval_ave_step", {"eval_ave_step": eval_ave_step}, total_num_steps)

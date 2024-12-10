import torch
import numpy as np
from hmasd.utils.util import get_shape_from_obs_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class StateSkillDataset(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        if args.skill_type == 'Discrete':
            skill_num = 1
        else:
            skill_num = args.team_skill_dim
        
        self.team_skill = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, skill_num), dtype=np.float32)
        self.indi_skill = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, skill_num), dtype=np.float32)

        self.share_obs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_team_states = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_indi_states = np.zeros_like(self.rnn_team_states)

        self.masks = np.ones((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.step = 0

    def insert(self, share_obs, obs, team_skill, indi_skill, rnn_team_states, rnn_indi_states, masks):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.team_skill[self.step] = team_skill.copy()
        self.indi_skill[self.step] = indi_skill.copy()
        self.rnn_team_states[self.step] = rnn_team_states.copy()
        self.rnn_indi_states[self.step] = rnn_indi_states.copy()
        self.masks[self.step] = masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.share_obs.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs.reshape(-1, *self.share_obs.shape[3:]) # (eplen*n_roll*n_agent, state_dim)
        obs = self.obs.reshape(-1, *self.obs.shape[3:])
        team_skill = self.team_skill.reshape(-1, *self.team_skill.shape[3:]) # (eplen*n_roll*n_agent, skill_num)
        indi_skill = self.indi_skill.reshape(-1, *self.indi_skill.shape[3:])
        rnn_team_states = self.rnn_team_states.reshape(-1, *self.rnn_team_states.shape[3:])
        # # (eplen*n_roll*n_agent, recurrent_N, hidden_size)
        rnn_indi_states = self.rnn_indi_states.reshape(-1, *self.rnn_indi_states.shape[3:])
        masks = self.masks.reshape(-1, 1)


        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices] # (mini_batch_size, state_dim)
            obs_batch = obs[indices]
            team_skill_batch = team_skill[indices]
            indi_skill_batch = indi_skill[indices]
            rnn_team_states_batch = rnn_team_states[indices] # (mini_batch_size, recurrent_N, hidden_size)
            rnn_indi_states_batch = rnn_indi_states[indices]
            masks_batch = masks[indices]

            yield share_obs_batch, obs_batch, team_skill_batch, indi_skill_batch, \
                  rnn_team_states_batch, rnn_indi_states_batch, masks_batch

    def recurrent_generator(self, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.share_obs.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs.transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs.transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs) # (n_roll*n_agent*eplen, state_dim)
            obs = _cast(self.obs)

        team_skill = _cast(self.team_skill) # (n_roll*n_agent*eplen, skill_num)
        indi_skill = _cast(self.indi_skill)
        masks = _cast(self.masks)
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_team_states = self.rnn_team_states.transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_team_states.shape[3:])
        # (n_roll*n_agent*eplen, recurrent_N, hidden_size)
        rnn_indi_states = self.rnn_indi_states.transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_indi_states.shape[3:])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            team_skill_batch = []
            indi_skill_batch = []
            rnn_team_states_batch = []
            rnn_indi_states_batch = []
            masks_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length]) # (data_chunk_length, state_dim)
                obs_batch.append(obs[ind:ind + data_chunk_length])
                team_skill_batch.append(team_skill[ind:ind + data_chunk_length])
                indi_skill_batch.append(indi_skill[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_team_states_batch.append(rnn_team_states[ind])
                rnn_indi_states_batch.append(rnn_indi_states[ind])

            # share_obs_batch: (mini_batch_size, data_chunk_length, state_dim)
            # rnn_team_states_batch: (mini_batch_size, recurrent_N, hidden_size)

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1) # (data_chunk_length, mini_batch_size, state_dim)
            obs_batch = np.stack(obs_batch, axis=1)
            team_skill_batch = np.stack(team_skill_batch, axis=1)
            indi_skill_batch = np.stack(indi_skill_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # States is just a (N, -1) from_numpy
            rnn_team_states_batch = np.stack(rnn_team_states_batch).reshape(N, *self.rnn_team_states.shape[3:])
            # (mini_batch_size, recurrent_N, hidden_size)
            rnn_indi_states_batch = np.stack(rnn_indi_states_batch).reshape(N, *self.rnn_indi_states.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch) # (data_chunk_length*mini_batch_size, state_dim)
            obs_batch = _flatten(L, N, obs_batch)
            team_skill_batch = _flatten(L, N, team_skill_batch)
            indi_skill_batch = _flatten(L, N, indi_skill_batch)
            masks_batch = _flatten(L, N, masks_batch)

            yield share_obs_batch, obs_batch, team_skill_batch, indi_skill_batch, \
                  rnn_team_states_batch, rnn_indi_states_batch, masks_batch

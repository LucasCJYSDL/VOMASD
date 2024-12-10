import torch
import numpy as np
from hmasd.utils.util import update_linear_schedule
from hmasd.utils.util import get_shape_from_obs_space
from hmasd.algorithms.utils.util import check
from hmasd.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer as MAT


class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, num_agents, device=torch.device("cpu")):
        # obs_space: [124, [4, 17], [6, 5], [1, 4], [1, 22]]
        # cent_obs_space: [156, [4, 20], [6, 8], [1, 4], [1, 24]]
        self.device = device
        self.lr = args.h_lr
        self.opti_eps = args.h_opti_eps
        self.weight_decay = args.h_weight_decay
        self._use_policy_active_masks = args.h_use_policy_active_masks

        self.action_type = args.skill_type

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == 'Discrete':
            self.act_dim = max(args.team_skill_dim, args.indi_skill_dim)
            self.act_num = 1
            self.available_actions = np.ones((num_agents + 1, self.act_dim))
            if args.team_skill_dim < args.indi_skill_dim:
                self.available_actions[0, args.team_skill_dim:] = 0.0
            elif args.team_skill_dim > args.indi_skill_dim:
                self.available_actions[1:, args.indi_skill_dim:] = 0.0
        else:
            assert args.team_skill_dim == args.indi_skill_dim
            self.act_dim = args.team_skill_dim
            self.act_num = self.act_dim
            self.available_actions = None

        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.transformer = MAT(self.share_obs_dim, self.obs_dim, self.act_dim, num_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head, 
                               device=device, action_type=self.action_type)

        self.transformer.zero_std()

        # count the volume of parameters of model
        # Total_params = 0
        # Trainable_params = 0
        # NonTrainable_params = 0
        # for param in self.transformer.parameters():
        #     mulValue = np.prod(param.size())
        #     Total_params += mulValue
        #     if param.requires_grad:
        #         Trainable_params += mulValue
        #     else:
        #         NonTrainable_params += mulValue
        # print(f'Total params: {Total_params}')
        # print(f'Trainable params: {Trainable_params}')
        # print(f'Non-trainable params: {NonTrainable_params}')

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        batch_size = obs.shape[0]
        available_actions = self.available_actions
        if available_actions is not None:
            available_actions = np.expand_dims(available_actions, 0).repeat(batch_size, 0)

        actions, action_log_probs, values = self.transformer.get_actions(cent_obs,
                                                                         obs,
                                                                         available_actions,
                                                                         deterministic)

        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        return values, actions, action_log_probs
        # (batch*(n_agent+1), 1), (batch*(n_agent+1), act_num), (batch*(n_agent+1), act_num)

    def get_values(self, cent_obs, obs):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        batch_size = obs.shape[0]
        available_actions = self.available_actions
        if available_actions is not None:
            available_actions = np.expand_dims(available_actions, 0).repeat(batch_size, 0)

        values = self.transformer.get_values(cent_obs, obs, available_actions)

        values = values.view(-1, 1)

        return values # (batch*(n_agent+1), 1)

    def evaluate_actions(self, cent_obs, obs, actions):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents + 1, self.act_num)
        batch_size = obs.shape[0]
        available_actions = self.available_actions
        if available_actions is not None:
            available_actions = np.expand_dims(available_actions, 0).repeat(batch_size, 0)

        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)

        action_log_probs = action_log_probs.view(-1, self.act_num) # (batch*(n_agent+1), act_num)
        values = values.view(-1, 1) # (batch*(n_agent+1), 1)
        entropy = entropy.view(-1, self.act_num) # (batch*(n_agent+1), act_num)

        entropy = entropy.mean()

        return values, action_log_probs, entropy # (batch*(n_agent+1), 1), (batch*(n_agent+1), act_num), (1, )

    def act(self, cent_obs, obs, deterministic=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        _, actions, _ = self.get_actions(cent_obs, obs, deterministic)

        return actions # (batch*(n_agent+1), act_num)

    def save(self, save_dir):
        torch.save(self.transformer.state_dict(), save_dir + "/transformer.pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir + "/transformer.pt")
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()


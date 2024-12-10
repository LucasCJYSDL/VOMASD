import torch
import torch.nn as nn
from torch.nn import functional as F
from hmasd.algorithms.utils.util import init, check
from hmasd.algorithms.utils.cnn import CNNBase
from hmasd.algorithms.utils.mlp import MLPBase
from hmasd.algorithms.utils.rnn import RNNLayer
from hmasd.algorithms.utils.act import ACTLayer
from hmasd.algorithms.utils.popart import PopArt
from hmasd.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        # obs_space: [124, [4, 17], [6, 5], [1, 4], [1, 22]] 
        # action_space: Discrete(12)
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.skill_last_layer = args.skill_last_layer
        self.policy_use_both_skill = args.policy_use_both_skill
        self.skill_type = args.skill_type

        self.indi_skill_dim = args.indi_skill_dim
        self.team_skill_dim = args.team_skill_dim
        if self.policy_use_both_skill:
            self.skill_dim = args.indi_skill_dim + args.team_skill_dim
        else:
            self.skill_dim = args.indi_skill_dim

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.l_use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.skill_last_layer:
            self.input_dim = obs_space[0]
            base = MLPBase
            self.base = base(args, self.input_dim)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.act = ACTLayer(action_space, self.hidden_size + self.skill_dim, self._use_orthogonal, self._gain)
        else:
            self.input_dim = obs_space[0] + self.skill_dim
            base = MLPBase
            self.base = base(args, self.input_dim)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, team_skill, indi_skill, rnn_states, masks, available_actions=None, deterministic=False):
        # obs: (batch, obs_dim) batch=n_roll*n_agent
        # team_skill: (batch, skill_num)
        # indi_skill: (batch, skill_num)
        # rnn_states: (batch, recurrent_N, hidden_size)
        # masks: (batch, 1)
        # available_actions: (batch, n_action)
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        team_skill = check(team_skill).to(**self.tpdv)
        indi_skill = check(indi_skill).to(**self.tpdv)
        if self.skill_type == 'Discrete':
            team_skill = F.one_hot(team_skill.long().squeeze(-1), num_classes=self.team_skill_dim)
            indi_skill = F.one_hot(indi_skill.long().squeeze(-1), num_classes=self.indi_skill_dim)
        if self.policy_use_both_skill:
            input_skill = torch.cat((team_skill, indi_skill), dim=-1)
        else:
            input_skill = indi_skill

        if self.skill_last_layer:
            input = obs
        else:
            input = torch.cat((obs, input_skill), dim=-1)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(input)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.skill_last_layer:
            actor_features = torch.cat((actor_features, input_skill), dim=-1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states # (batch, act_num), (batch, act_num), (batch, recurrent_N, hidden_size)

    def evaluate_actions(self, obs, team_skill, indi_skill, rnn_states, action, masks, available_actions=None, active_masks=None):
        # obs: (data_chunk_length*mini_batch_size, obs_dim) 
        # team_skill: (data_chunk_length*mini_batch_size, skill_num) 
        # indi_skill: (data_chunk_length*mini_batch_size, skill_num) 
        # rnn_states: (mini_batch_size, recurrent_N, hidden_size)
        # action: (data_chunk_length*mini_batch_size, act_num)
        # masks: (data_chunk_length*mini_batch_size, 1)
        # available_actions: (data_chunk_length*mini_batch_size, n_action)
        # active_masks: (data_chunk_length*mini_batch_size, 1)
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        team_skill = check(team_skill).to(**self.tpdv)
        indi_skill = check(indi_skill).to(**self.tpdv)
        if self.skill_type == 'Discrete':
            team_skill = F.one_hot(team_skill.long().squeeze(-1), num_classes=self.team_skill_dim)
            indi_skill = F.one_hot(indi_skill.long().squeeze(-1), num_classes=self.indi_skill_dim)
        if self.policy_use_both_skill:
            input_skill = torch.cat((team_skill, indi_skill), dim=-1)
        else:
            input_skill = indi_skill
        
        if self.skill_last_layer:
            input = obs
        else:
            input = torch.cat((obs, input_skill), dim=-1)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(input)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.skill_last_layer:
            actor_features = torch.cat((actor_features, input_skill), dim=-1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy # (data_chunk_length*mini_batch_size, act_num), (1, )


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.skill_last_layer = args.skill_last_layer
        self.policy_use_both_skill = args.policy_use_both_skill
        self.skill_type = args.skill_type

        self.indi_skill_dim = args.indi_skill_dim
        self.team_skill_dim = args.team_skill_dim
        if self.policy_use_both_skill:
            self.skill_dim = args.indi_skill_dim + args.team_skill_dim
        else:
            self.skill_dim = args.team_skill_dim

        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        if self.skill_last_layer:
            self.input_dim = cent_obs_space[0]
            last_input_size = self.hidden_size + self.skill_dim
        else:
            self.input_dim = cent_obs_space[0] + self.skill_dim
            last_input_size = self.hidden_size
        base = MLPBase
        self.base = base(args, self.input_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(last_input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(last_input_size, 1))

        self.to(device)

    def forward(self, cent_obs, team_skill, indi_skill, rnn_states, masks):
        # roll_out
        # cent_obs: (batch, state_dim) batch=n_roll*n_agent
        # team_skill: (batch, skill_num)
        # indi_skill: (batch, skill_num)
        # rnn_states: (batch, recurrent_N, hidden_size)
        # masks: (batch, 1)
        # train
        # cent_obs: (data_chunk_length*mini_batch_size, state_dim)
        # team_skill: (data_chunk_length*mini_batch_size, skill_num)
        # rnn_states: (mini_batch_size, recurrent_N, hidden_size)
        # masks: (data_chunk_length*mini_batch_size, 1)
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        team_skill = check(team_skill).to(**self.tpdv)
        indi_skill = check(indi_skill).to(**self.tpdv)
        if self.skill_type == 'Discrete':
            team_skill = F.one_hot(team_skill.long().squeeze(-1), num_classes=self.team_skill_dim)
            indi_skill = F.one_hot(indi_skill.long().squeeze(-1), num_classes=self.indi_skill_dim)
        if self.policy_use_both_skill:
            input_skill = torch.cat((team_skill, indi_skill), dim=-1)
        else:
            input_skill = team_skill

        if self.skill_last_layer:
            input = cent_obs
        else:
            input = torch.cat((cent_obs, input_skill), dim=-1)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        if self.skill_last_layer:
            critic_features = torch.cat((critic_features, input_skill), dim=-1)
        values = self.v_out(critic_features)

        return values, rnn_states
        # roll_out (batch, 1), (batch, recurrent_N, hidden_size)
        # train (data_chunk_length*mini_batch_size, 1), (mini_batch_size, recurrent_N, hidden_size)

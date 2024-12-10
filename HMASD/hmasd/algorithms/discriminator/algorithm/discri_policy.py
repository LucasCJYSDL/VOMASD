import torch
from torch.nn import functional as F
import numpy as np
from hmasd.algorithms.discriminator.algorithm.discri_model import R_Discri
from hmasd.utils.util import update_linear_schedule


class DiscriPolicy:

    def __init__(self, args, obs_space, cent_obs_space, device=torch.device("cpu")):
        self.device = device
        self.team_lr = args.d_team_lr
        self.indi_lr = args.d_indi_lr
        self.opti_eps = args.d_opti_eps
        self.weight_decay = args.d_weight_decay

        obs_dim = obs_space[0]
        state_dim = cent_obs_space[0]
        self.team_skill_dim = args.team_skill_dim
        self.indi_skill_dim = args.indi_skill_dim
        self.skill_type = args.skill_type

        self.team_discri = R_Discri(args, state_dim, self.team_skill_dim, self.skill_type, self.device)
        self.indi_discri = R_Discri(args, obs_dim + self.team_skill_dim, self.indi_skill_dim, self.skill_type, self.device)

        self.team_discri_optimizer = torch.optim.Adam(self.team_discri.parameters(),
                                                lr=self.team_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.indi_discri_optimizer = torch.optim.Adam(self.indi_discri.parameters(),
                                                lr=self.indi_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.team_discri_optimizer, episode, episodes, self.team_lr)
        update_linear_schedule(self.indi_discri_optimizer, episode, episodes, self.indi_lr)
    
    def get_logits(self, state, obs, team_skill, rnn_team_states, rnn_indi_states, masks):
        team_discri_input = state
        if self.skill_type == 'Discrete':
            team_skill_onehot = F.one_hot(torch.from_numpy(team_skill).long().squeeze(-1), num_classes=self.team_skill_dim).numpy()
            indi_discri_input = np.concatenate((obs, team_skill_onehot), axis=-1)
        else:
            indi_discri_input = np.concatenate((obs, team_skill), axis=-1)
        team_logits, _ = self.team_discri(team_discri_input, rnn_team_states, masks)
        indi_logits, _ = self.indi_discri(indi_discri_input, rnn_indi_states, masks)

        return team_logits, indi_logits # (batch, team_skill_dim), (batch, indi_skill_dim)


    def get_intrinsic_reward(self, state, obs, team_skill, indi_skill, rnn_team_states, rnn_indi_states, masks):
        # state: (n_roll*n_agent, state_dim)
        # rnn_team_states: (n_roll*n_agent, recurrent_N, hidden_size)
        team_intri_rew, rnn_team_states = self.team_discri.get_intrinsic_reward(state, rnn_team_states, team_skill, masks)
        # (n_roll*n_agent, 1), (n_roll*n_agent, recurrent_N, hidden_size)
    
        if self.skill_type == 'Discrete':
            team_skill_onehot = F.one_hot(torch.from_numpy(team_skill).long().squeeze(-1), num_classes=self.team_skill_dim).numpy()
            indi_discri_input = np.concatenate((obs, team_skill_onehot), axis=-1)
        else:
            indi_discri_input = np.concatenate((obs, team_skill), axis=-1)
        indi_intri_rew, rnn_indi_states = self.indi_discri.get_intrinsic_reward(indi_discri_input, rnn_indi_states, indi_skill, masks)
        # (n_roll*n_agent, 1), (n_roll*n_agent, recurrent_N, hidden_size)

        return team_intri_rew, indi_intri_rew, rnn_team_states, rnn_indi_states
        # (n_roll*n_agent, 1), (n_roll*n_agent, 1), (n_roll*n_agent, recurrent_N, hidden_size), (n_roll*n_agent, recurrent_N, hidden_size)

    def save(self, save_dir):
        torch.save(self.team_discri.state_dict(), save_dir + "/team_discri.pt")
        torch.save(self.indi_discri.state_dict(), save_dir + "/indi_discri.pt")

    def restore(self, model_dir):
        team_state_dict = torch.load(model_dir + "/team_discri.pt")
        self.team_discri.load_state_dict(team_state_dict)
        indi_state_dict = torch.load(model_dir + "/indi_discri.pt")
        self.indi_discri.load_state_dict(indi_state_dict)

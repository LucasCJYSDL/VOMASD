import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from hmasd.utils.util import get_gard_norm
from hmasd.algorithms.utils.util import check

class D_Trainer():

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.team_skill_dim = args.team_skill_dim
        self.indi_skill_dim = args.indi_skill_dim
        self.skill_type = args.skill_type
        self.d_epoch = args.d_epoch
        self.num_mini_batch = args.d_num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.max_grad_norm = args.d_max_grad_norm

        self._use_recurrent_policy = args.use_recurrent_discri
        self._use_max_grad_norm = args.d_use_max_grad_norm

    def discri_update(self, sample):
        share_obs_batch, obs_batch, team_skill_batch, indi_skill_batch, \
        rnn_team_states_batch, rnn_indi_states_batch, masks_batch = sample
        # no rnn
        # share_obs_batch: (mini_batch_size, state_dim)
        # rnn_team_states_batch: (mini_batch_size, recurrent_N, hidden_size)
        # rnn
        # share_obs_batch: (data_chunk_length*mini_batch_size, state_dim)
        # rnn_team_states_batch: (mini_batch_size, recurrent_N, hidden_size)

        # Reshape to do in a single forward pass for all steps
        team_logits, indi_logits = self.policy.get_logits(share_obs_batch,
                                                          obs_batch, 
                                                          team_skill_batch,
                                                          rnn_team_states_batch,
                                                          rnn_indi_states_batch,
                                                          masks_batch)
        # (batch, team_skill_dim), (batch, indi_skill_dim)

        team_skill_batch = check(team_skill_batch).to(**self.tpdv) # (batch, skill_num)
        indi_skill_batch = check(indi_skill_batch).to(**self.tpdv) # (batch, skill_num)

        if self.skill_type == 'Discrete':
            team_log_prob = log_softmax(team_logits, dim=-1)
            team_discri_loss = - torch.gather(team_log_prob, -1, team_skill_batch.long()) # (batch, 1)
            indi_log_prob = log_softmax(indi_logits, dim=-1)
            indi_discri_loss = - torch.gather(indi_log_prob, -1, indi_skill_batch.long()) # (batch, 1)
        else:
            team_discri_loss = ((team_logits - team_skill_batch) ** 2).sum(-1, keepdim=True) # (batch, 1)     
            indi_discri_loss = ((indi_logits - indi_skill_batch) ** 2).sum(-1, keepdim=True) # (batch, 1)  
        
        team_discri_loss = team_discri_loss.mean()
        indi_discri_loss = indi_discri_loss.mean()

        self.policy.team_discri_optimizer.zero_grad()
        team_discri_loss.backward()
        if self._use_max_grad_norm:
            team_grad_norm = nn.utils.clip_grad_norm_(self.policy.team_discri.parameters(), self.max_grad_norm)
        else:
            team_grad_norm = get_gard_norm(self.policy.team_discri.parameters())
        self.policy.team_discri_optimizer.step()

        self.policy.indi_discri_optimizer.zero_grad()
        indi_discri_loss.backward()
        if self._use_max_grad_norm:
            indi_grad_norm = nn.utils.clip_grad_norm_(self.policy.indi_discri.parameters(), self.max_grad_norm)
        else:
            indi_grad_norm = get_gard_norm(self.policy.indi_discri.parameters())
        self.policy.indi_discri_optimizer.step()

        return team_discri_loss, team_grad_norm, indi_discri_loss, indi_grad_norm

    def train(self, buffer):
        train_info = {}

        train_info['team_discri_loss'] = 0
        train_info['indi_discri_loss'] = 0
        train_info['team_grad_norm'] = 0
        train_info['indi_grad_norm'] = 0

        for _ in range(self.d_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.feed_forward_generator(self.num_mini_batch)

            for sample in data_generator:

                team_discri_loss, team_grad_norm, indi_discri_loss, indi_grad_norm = self.discri_update(sample)

                train_info['team_discri_loss'] += team_discri_loss.item()
                train_info['indi_discri_loss'] += indi_discri_loss.item()
                train_info['team_grad_norm'] += team_grad_norm
                train_info['indi_grad_norm'] += indi_grad_norm

        num_updates = self.d_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.team_discri.train()
        self.policy.indi_discri.train()

    def prep_rollout(self):
        self.policy.team_discri.eval()
        self.policy.indi_discri.eval()

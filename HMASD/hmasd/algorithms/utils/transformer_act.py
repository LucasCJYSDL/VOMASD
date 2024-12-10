import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(decoder, obs_rep, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    # obs_rep: (batch, n_agent+1, n_embd)
    # available_actions: (batch, n_agent+1, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent + 1, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent + 1, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent + 1):
        logit = decoder(shifted_action, obs_rep)[:, i, :] # (batch, action_dim)
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample() # (batch, )
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent + 1:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log # (batch_size, n_agent+1, 1), (batch_size, n_agent+1, 1)


def discrete_parallel_act(decoder, obs_rep, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    # obs_rep: (batch, n_agent+1, n_embd)
    # action: (batch, n_agent+1, 1)
    # available_actions: (batch, n_agent+1, action_dim)
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent+1, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent + 1, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep) # (batch, n_agent+1, action_dim)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1) # (batch, n_agent+1, 1)
    entropy = distri.entropy().unsqueeze(-1) # (batch, n_agent+1, 1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    # obs_rep: (batch, n_agent+1, n_embd)
    # available_actions: (batch, n_agent+1, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent + 1, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent + 1, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent + 1):
        act_mean = decoder(shifted_action, obs_rep)[:, i, :] # (batch, action_dim)
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample() # (batch, action_dim)
        action_log = distri.log_prob(action) # (batch, action_dim)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent + 1:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log # (batch_size, n_agent+1, action_dim), (batch_size, n_agent+1, action_dim)


def continuous_parallel_act(decoder, obs_rep, action, batch_size, n_agent, action_dim, tpdv):
    # obs_rep: (batch, n_agent+1, n_embd)
    # action: (batch, n_agent+1, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent + 1, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action) # (batch, n_agent+1, action_dim)
    entropy = distri.entropy() # (batch, n_agent+1, action_dim)
    return action_log, entropy

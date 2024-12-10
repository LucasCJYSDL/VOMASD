import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.distributions import Categorical


def discrete_autoregreesive_act(decoder, obs_rep, batch_size, n_agent, action_dim, tpdv,
                                available_actions, deterministic, second_stage, max_skill_size):
    # obs_rep: (batch, n_agent, n_embd)
    # available_actions: (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv) # there is an extra action dimension for z_0
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long, device=shifted_action.device)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32, device=shifted_action.device)

    if second_stage:
        recording = [[0 for i in range(n_agent)] for j in range(batch_size)]

    output_ls = []
    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep)[:, i, :] # (batch, action_dim)
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10
        output_ls.append(logit)

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample() # (batch, )
        action_log = distri.log_prob(action)

        if second_stage:
            for b in range(batch_size):
                recording[b][action[b].item()] += 1
                if recording[b][action[b].item()] >= max_skill_size: # action_dim is max_skill_size
                    for agent_id in range(i+1, n_agent):
                        available_actions[b][agent_id][action[b].item()] = 0

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
        
        # print("2: ", logit[0], action[0], action_log[0], output_action[0], shifted_action[0])
    if not second_stage:
        return output_action, output_action_log, torch.stack(output_ls, dim=1) # (batch_size, n_agent, 1), (batch_size, n_agent, 1)
    else:
        return output_action, output_action_log, torch.stack(output_ls, dim=1), available_actions

def discrete_parallel_act(decoder, obs_rep, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    # obs_rep: (batch, n_agent, n_embd)
    # action: (batch, n_agent, 1)
    # available_actions: (batch, n_agent, action_dim)
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :] 
    logit = decoder(shifted_action, obs_rep) # (batch, n_agent, action_dim)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1) # (batch, n_agent, 1)
    entropy = distri.entropy().unsqueeze(-1) # (batch, n_agent, 1)
    return action_log, entropy

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, max_n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, max_n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, x, x, mask))
        x = self.ln2(x + self.mlp(x))
        return x

class LightEncodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, max_n_agent):
        super(LightEncodeBlock, self).__init__()
        self.attn = SelfAttention(n_embd, n_head, max_n_agent, masked=False)

    def forward(self, x, mask=None):
        x = self.attn(x, x, x, mask)
        return x

class Encoder(nn.Module):
    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, max_n_agent):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.max_n_agent = max_n_agent

        # TODO: comment this. we have encoded them outside of MAT, so that it can be used in multi-task scenarios
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU()) 

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, max_n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1))) # critic head

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        state_embeddings = self.state_encoder(state) # (batch, n_agent, n_embd)
        state_embeddings = state_embeddings.mean(dim=1, keepdim=True) # (batch, 1, n_embd)
        
        obs_embeddings = self.obs_encoder(obs) # (batch, n_agent, n_embd)
        x = torch.cat((state_embeddings, obs_embeddings), dim=1) # (batch, n_agent+1, n_embd)

        rep = self.blocks(self.ln(x)) # (batch, n_agent+1, n_embd)
        v_loc = self.head(rep) # (batch, n_agent+1, 1)

        return v_loc, rep
    
class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, max_n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(max_n_agent + 1, max_n_agent + 1))
                             .view(1, 1, max_n_agent + 1, max_n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query, mask=None):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)
        if self.masked and mask is None:
            # print("3: ", self.mask, self.mask.shape)
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        if mask is not None:
            # print("2: ", att, mask)
            att = att.masked_fill(mask == 0, float('-inf'))
            # print("3: ", att)

        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, max_n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, max_n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, max_n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):

    def __init__(self, action_dim, n_block, n_embd, n_head, max_n_agent, action_type='Discrete'):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.max_n_agent = max_n_agent

        # self.team_action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
        #                                     nn.GELU())
        self.indi_action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, max_n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                    init_(nn.Linear(n_embd, action_dim)))

    # state, action, and return
    def forward(self, action, obs_rep):
        # action: (batch, n_agent, n_action+1), one-hot/logits?
        # obs_rep: (batch, real_n_agent, n_embd)

        action_embeddings = self.indi_action_encoder(action) # (batch, n_agent, n_embd)
        x = self.ln(action_embeddings)
        for block in self.blocks:
            x = block(x, obs_rep) # (batch, n_agent, n_embd)
        logit = self.head(x) # (batch, n_agent, action_dim)

        return logit

class MultiAgentTransformer(nn.Module):

    def __init__(self, action_dim, max_n_agent, max_skill_size, n_block, n_embd, 
                 n_head, device, action_type='Discrete'):
        super(MultiAgentTransformer, self).__init__()

        self.max_n_agent = max_n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.max_skill_size = max_skill_size

        self.decoder = Decoder(action_dim, n_block, n_embd, n_head, max_n_agent, self.action_type)
        
        self.to(device)

    def forward(self, state_rep, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, action_dim)
        # only for training

        batch_size, n_agent = state_rep.shape[0], state_rep.shape[1]

        action = action.long()
        action_log, entropy = discrete_parallel_act(self.decoder, state_rep, action, batch_size,
                                                    n_agent, self.action_dim, self.tpdv, available_actions)
        # (batch, n_agent, 1), (batch, n_agent, 1)

        return action_log, entropy
    
    def get_actions(self, state_rep, available_actions=None, deterministic=False, second_stage=False):
        # state: (batch, n_agent, state_dim)
        batch_size, n_agent = state_rep.shape[0], state_rep.shape[1]

        # TODO: use self.discrete_autoregreesive_act or discrete_autoregreesive_act
        outcomes = discrete_autoregreesive_act(self.decoder, state_rep, batch_size, n_agent, self.action_dim, self.tpdv,
                                               available_actions, deterministic, second_stage, self.max_skill_size)
        # outcomes = self.discrete_autoregreesive_act(self.decoder, state_rep, batch_size, n_agent, self.action_dim, self.tpdv,
        #                                             available_actions, deterministic, second_stage, self.max_skill_size)
        if not second_stage:
            output_action, output_action_log, output_action_logits = outcomes
        else:
            output_action, output_action_log, output_action_logits, avai_groups = outcomes
        # print("1: ", output_action.shape, output_action_log.shape) # torch.Size([128, 3, 1]) torch.Size([128, 3, 1])
        if not second_stage:
            return output_action, output_action_log, output_action_logits
        else:
            return output_action, output_action_log, output_action_logits, avai_groups
    
    def discrete_autoregreesive_act(self, decoder, obs_rep, batch_size, n_agent, action_dim, tpdv, available_actions, deterministic, second_stage, max_skill_size):
        # obs_rep: (batch, n_agent, n_embd)
        # available_actions: (batch, n_agent, action_dim)
        self.shifted_action = [torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv) for _ in range(n_agent)] # there is an extra action dimension for z_0
        for i in range(n_agent):
            self.shifted_action[i][:, 0, 0] = 1
        output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long, device=self.shifted_action[0].device)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32, device=self.shifted_action[0].device)

        output_ls = []

        if second_stage:
            recording = [[0 for i in range(n_agent)] for j in range(batch_size)]

        for i in range(n_agent):
            logit = decoder(self.shifted_action[i], obs_rep)[:, i, :] # (batch, action_dim)
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10
            output_ls.append(logit)

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample() # (batch, )
            action_log = distri.log_prob(action)

            if second_stage:
                for b in range(batch_size):
                    recording[b][action[b].item()] += 1
                    if recording[b][action[b].item()] >= max_skill_size: # action_dim is max_skill_size
                        for agent_id in range(i+1, n_agent):
                            available_actions[b][agent_id][action[b].item()] = 0

            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i, :] = action_log.unsqueeze(-1)
            if i + 1 < n_agent:
                self.shifted_action[i+1] = self.shifted_action[i].clone()
                self.shifted_action[i+1][:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)

        # return output_action, output_action_log, torch.stack(output_ls, dim=1) # (batch_size, n_agent, 1), (batch_size, n_agent, 1)
        if not second_stage:
            return output_action, output_action_log, torch.stack(output_ls, dim=1) # (batch_size, n_agent, 1), (batch_size, n_agent, 1)
        else:
            return output_action, output_action_log, torch.stack(output_ls, dim=1), available_actions

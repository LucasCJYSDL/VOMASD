import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .util import init, get_clones, binary_embed

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x


class MultiTaskBase(nn.Module):
    def __init__(self, args, state_shape):
        super(MultiTaskBase, self).__init__()
        self.args = args

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # example: for 3m state, [81, [2, 8, 9], [3, 8], [1, 4], [1, 7, 9, 3, 0]]
        # example: for 3m obs, [64, [2, 5, 9], [3, 5], [1, 4], [1, 5, 9, 3, 0]]
        # ally info
        self.n_ally = state_shape[1][0]
        self.ally_shape = state_shape[1][1]
        # enemy info
        self.n_enemy = state_shape[2][0]
        self.enemy_shape = state_shape[2][1]
        # move info
        self.move_shape = state_shape[3][1]
        # own info
        self.own_shape = state_shape[4][1]
        self.id_shape = state_shape[4][3]
        self.time_shape = state_shape[4][4]
        # last action info
        self.last_action_shape = state_shape[1][2]
        self.n_agent = self.n_ally + 1
        self.n_no_attack_action = self.last_action_shape - args.max_agent_id 

        self._build_mlp_layers()

        self.attn_embed_dim = 8
        self.state_query = nn.Linear(self.hidden_size, self.attn_embed_dim)
        self.state_key = nn.Linear(self.hidden_size, self.attn_embed_dim) # TODO: adjust this parameter
    
    def _build_mlp_layers(self):

        w_ally_shape = self.ally_shape + self.n_no_attack_action + 1
        w_enemy_shape = self.enemy_shape + 1
        w_own_shape = self.own_shape + self.move_shape + self.n_no_attack_action + 1 + self.args.id_length + self.time_shape

        if self._use_feature_normalization:
            self.own_feature_norm = nn.LayerNorm(w_own_shape)
            self.ally_feature_norm = nn.LayerNorm(w_ally_shape)
            self.enemy_feature_norm = nn.LayerNorm(w_enemy_shape)

        self.own_mlp = MLPLayer(w_own_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.ally_mlp = MLPLayer(w_ally_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.enemy_mlp = MLPLayer(w_enemy_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        
    
    def decompose_action(self, action_info):
        """
        action_info: shape [bs * n_agent, n_action]
        """
        shape = action_info.shape
        if len(shape) > 2:
            action_info = action_info.reshape(-1, shape[-1])
        no_attack_action_info = action_info[:, :self.n_no_attack_action]
        
        attack_action_info = action_info[:, self.n_no_attack_action: self.n_no_attack_action + self.n_enemy]
        # recover shape
        no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], self.n_no_attack_action)    
        attack_action_info = attack_action_info.reshape(*shape[:-1], self.n_enemy)
        
        # get compact action # TODO: can be improved
        bin_attack_info = th.sum(attack_action_info, dim=-1).unsqueeze(-1)
        compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1)
        
        return no_attack_action_info, attack_action_info, compact_action_info
    
    def decompose(self, state):
        # print(state.shape) # torch.Size([3, 81]) for batch state, torch.Size([24, 64]) for batch obs
        
        bs = state.shape[0] // self.n_agent

        ally_states, ally_states_la = [], []
        ally_step = self.ally_shape + self.last_action_shape
        for i in range(self.n_agent - 1):
            ally_states.append(state[:, i*ally_step: i*ally_step + self.ally_shape])
            ally_states_la.append(state[:, i*ally_step + self.ally_shape: (i+1)*ally_step])
        base = (self.n_agent - 1) * ally_step

        enemy_states = [state[:, base + i*self.enemy_shape: base + (i + 1)*self.enemy_shape] for i in range(self.n_enemy)]
        base += self.n_enemy * self.enemy_shape
        
        move_states = state[:, base: base + self.move_shape]
        base += self.move_shape

        own_states = state[:, base: base + self.own_shape]
        base += self.own_shape

        own_states_la = state[:, base: base + self.last_action_shape]
        base += self.last_action_shape

        agent_id_states = state[:, base: base + self.n_agent]
        base += self.n_agent

        if self.time_shape > 0:
            time_states = state[:, base: base + 1]

        # print(ally_states, ally_states_la, own_states, own_states_la, agent_id_states)
        u_agent_id_states = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent_id), dtype=state.dtype) for i in range(self.n_agent)]
        u_agent_id_states = th.stack(u_agent_id_states, dim=0).repeat(bs, 1).to(state.device)
        # print(u_agent_id_states.shape, u_agent_id_states)

        _, _, ally_compact_la = self.decompose_action(th.stack(ally_states_la, dim=0))
        # print(ally_compact_la.shape, th.stack(ally_states, dim=0).shape), torch.Size([2, 3, 7]) torch.Size([2, 3, 8])
        w_ally_states = th.cat([th.stack(ally_states, dim=0), ally_compact_la], dim=-1)

        _, own_attack_la, own_compact_ls = self.decompose_action(own_states_la)
        own_attack_la = own_attack_la.transpose(0, 1).unsqueeze(-1)
        # print(own_attack_la.shape) torch.Size([3, 3, 1])
        w_enemy_states = th.cat([th.stack(enemy_states, dim=0), own_attack_la], dim=-1)

        w_own_states = th.cat([own_states, move_states, own_compact_ls, u_agent_id_states], dim=-1)
        if self.time_shape > 0:
            w_own_states = th.cat([w_own_states, time_states], dim=-1)
        w_own_states = w_own_states.unsqueeze(0)
        # torch.Size([2, 3, 15]) torch.Size([3, 3, 9]) torch.Size([1, 3, 22])
        # torch.Size([4, 5, 15]) torch.Size([5, 5, 9]) torch.Size([1, 5, 22])
        # print(w_ally_states.shape, w_enemy_states.shape, w_own_states.shape)

        return w_own_states, w_ally_states, w_enemy_states
    
    def forward(self, x):
        own_x, ally_x, enemy_x = self.decompose(x)
        
        if self._use_feature_normalization:
            own_x = self.own_feature_norm(own_x)
            ally_x = self.ally_feature_norm(ally_x)
            enemy_x = self.enemy_feature_norm(enemy_x)
        
        own_x = self.own_mlp(own_x)
        ally_x = self.ally_mlp(ally_x)
        enemy_x = self.enemy_mlp(enemy_x)

        # use self attention to aggregate
        entity_s_embed = th.cat([own_x, ally_x, enemy_x], dim=0) # (n_entity, bs*seq_len, -1), torch.Size([6, 3, 64])
        s_proj_query = self.state_query(entity_s_embed).permute(1, 0, 2) # torch.Size([3, 6, 8])
        s_proj_key = self.state_key(entity_s_embed).permute(1, 2, 0) # torch.Size([3, 8, 6])
        s_energy = th.bmm(s_proj_query / (self.attn_embed_dim ** (1 / 2)), s_proj_key) # torch.Size([3, 6, 6])
        s_attn_score = F.softmax(s_energy, dim=2) # torch.Size([3, 6, 6])
        s_proj_value = entity_s_embed.permute(1, 0, 2) # torch.Size([3, 6, 64])
        s_attn_out = th.bmm(s_attn_score, s_proj_value) # torch.Size([3, 64])
        # print(s_attn_score)

        return s_attn_out[:, 0, :]
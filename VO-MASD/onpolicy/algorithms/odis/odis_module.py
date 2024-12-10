import torch as th
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.mlp import MLPLayer
from algorithms.utils.util import binary_embed
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.util import init, check


class EnvInfo(object):
    def __init__(self):
        # ally info
        self.n_ally = None
        self.ally_shape = None
        # enemy info
        self.n_enemy = None
        self.enemy_shape = None
        # move info
        self.move_shape = None
        # own info
        self.own_shape = None
        self.id_shape = None
        self.time_shape = None
        # last action info
        self.last_action_shape = None
        self.n_agent = None
        self.n_no_attack_action = None
    
    def get_shape(self):
        return self.n_agent, self.ally_shape, self.last_action_shape, self.enemy_shape,\
              self.n_enemy, self.move_shape, self.own_shape, self.time_shape

class MultiTaskDecomposer(object):
    def __init__(self, args, state_shape_list):
        self.args = args

        self.task_info = {}
        for task in state_shape_list:
            # example: for 3m state, [81, [2, 8, 9], [3, 8], [1, 4], [1, 7, 9, 3, 0]]
            # example: for 3m obs, [64, [2, 5, 9], [3, 5], [1, 4], [1, 5, 9, 3, 0]]
            state_shape = state_shape_list[task]
            env_info = EnvInfo()
            # ally info
            env_info.n_ally = state_shape[1][0]
            env_info.ally_shape = state_shape[1][1]
            # enemy info
            env_info.n_enemy = state_shape[2][0]
            env_info.enemy_shape = state_shape[2][1]
            # move info
            env_info.move_shape = state_shape[3][1]
            # own info
            env_info.own_shape = state_shape[4][1]
            env_info.id_shape = state_shape[4][3]
            env_info.time_shape = state_shape[4][4]
            # last action info
            env_info.last_action_shape = state_shape[1][2]
            env_info.n_agent = env_info.n_ally + 1
            env_info.n_no_attack_action = env_info.last_action_shape - args.max_agent_id

            self.task_info[task] = env_info
        
        self.w_ally_shape = self.task_info[task].ally_shape + self.task_info[task].n_no_attack_action + 1
        self.w_enemy_shape = self.task_info[task].enemy_shape + 1
        self.w_own_shape = self.task_info[task].own_shape + self.task_info[task].move_shape + self.task_info[task].n_no_attack_action + 1 +\
                           self.args.id_length + self.task_info[task].time_shape
        self.n_no_attack_action = self.task_info[task].n_no_attack_action
    
    def decompose_action(self, action_info, task):
        """
        action_info: shape [bs * n_agent, n_action]
        """
        shape = action_info.shape
        if len(shape) > 2:
            action_info = action_info.reshape(-1, shape[-1])
        
        n_no_attack_action = self.task_info[task].n_no_attack_action
        n_enemy = self.task_info[task].n_enemy

        no_attack_action_info = action_info[:, :n_no_attack_action]
        attack_action_info = action_info[:, n_no_attack_action: n_no_attack_action + n_enemy]
        # recover shape
        no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], n_no_attack_action)    
        attack_action_info = attack_action_info.reshape(*shape[:-1], n_enemy)
        
        # get compact action # TODO: can be improved
        bin_attack_info = th.sum(attack_action_info, dim=-1).unsqueeze(-1)
        compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1)
        
        return no_attack_action_info, attack_action_info, compact_action_info

    def decompose(self, state, task):
        # print(state.shape) # torch.Size([3, 81]) for batch state, torch.Size([24, 64]) for batch obs
        n_agent, ally_shape, last_action_shape, enemy_shape,\
              n_enemy, move_shape, own_shape, time_shape = self.task_info[task].get_shape()

        bs = state.shape[0] // n_agent

        ally_states, ally_states_la = [], []
        ally_step = ally_shape + last_action_shape
        for i in range(n_agent - 1):
            ally_states.append(state[:, i*ally_step: i*ally_step + ally_shape])
            ally_states_la.append(state[:, i*ally_step + ally_shape: (i+1)*ally_step])
        base = (n_agent - 1) * ally_step

        enemy_states = [state[:, base + i*enemy_shape: base + (i + 1)*enemy_shape] for i in range(n_enemy)]
        base += n_enemy * enemy_shape
        
        move_states = state[:, base: base + move_shape]
        base += move_shape

        own_states = state[:, base: base + own_shape]
        base += own_shape

        own_states_la = state[:, base: base + last_action_shape]
        base += last_action_shape

        agent_id_states = state[:, base: base + n_agent]
        base += n_agent

        if time_shape > 0:
            time_states = state[:, base: base + 1]

        # print(ally_states, ally_states_la, own_states, own_states_la, agent_id_states)
        u_agent_id_states = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent_id), dtype=state.dtype) for i in range(n_agent)]
        u_agent_id_states = th.stack(u_agent_id_states, dim=0).repeat(bs, 1).to(state.device)
        # print(u_agent_id_states.shape, u_agent_id_states)

        _, _, ally_compact_la = self.decompose_action(th.stack(ally_states_la, dim=0), task)
        # print(ally_compact_la.shape, th.stack(ally_states, dim=0).shape), torch.Size([2, 3, 7]) torch.Size([2, 3, 8])
        w_ally_states = th.cat([th.stack(ally_states, dim=0), ally_compact_la], dim=-1)

        _, own_attack_la, own_compact_ls = self.decompose_action(own_states_la, task)
        own_attack_la = own_attack_la.transpose(0, 1).unsqueeze(-1)
        # print(own_attack_la.shape) torch.Size([3, 3, 1])
        w_enemy_states = th.cat([th.stack(enemy_states, dim=0), own_attack_la], dim=-1)

        w_own_states = th.cat([own_states, move_states, own_compact_ls, u_agent_id_states], dim=-1)
        if time_shape > 0:
            w_own_states = th.cat([w_own_states, time_states], dim=-1)
        w_own_states = w_own_states.unsqueeze(0)
        # torch.Size([2, 3, 15]) torch.Size([3, 3, 9]) torch.Size([1, 3, 22])
        # torch.Size([4, 5, 15]) torch.Size([5, 5, 9]) torch.Size([1, 5, 22])
        # print(w_ally_states.shape, w_enemy_states.shape, w_own_states.shape)

        return w_own_states, w_ally_states, w_enemy_states

    def get_task_independent_shape(self):

        return self.w_own_shape, self.w_ally_shape, self.w_enemy_shape, self.n_no_attack_action

    def get_n_enemy(self, task):

        return self.task_info[task].n_enemy


class MultiTaskModuleBase(nn.Module):
    def __init__(self, args, decomposer, device):
        super(MultiTaskModuleBase, self).__init__()

        self.args = args
        self.decomposer = decomposer

        self.attn_embed_dim = 8
        self.hidden_size = args.hidden_size # 64

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.tpdv = dict(dtype=th.float32, device=device)

        self._build_layers()

        self.state_query = nn.Linear(self.hidden_size, self.attn_embed_dim)
        self.state_key = nn.Linear(self.hidden_size, self.attn_embed_dim)

        self.to(device)
    
    def _build_layers(self):
        pass


class StateEncoder(MultiTaskModuleBase): 
    def __init__(self, args, decomposer, device):
        super(StateEncoder, self).__init__(args, decomposer, device)
    
    def _build_layers(self):
        w_own_shape, w_ally_shape, w_enemy_shape, self.n_no_attack_action = self.decomposer.get_task_independent_shape()
        w_own_shape += (self.n_no_attack_action + 1)
        w_enemy_shape += 1

        if self._use_feature_normalization:
            self.own_feature_norm = nn.LayerNorm(w_own_shape)
            self.ally_feature_norm = nn.LayerNorm(w_ally_shape)
            self.enemy_feature_norm = nn.LayerNorm(w_enemy_shape)

        self.own_mlp = MLPLayer(w_own_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.ally_mlp = MLPLayer(w_ally_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.enemy_mlp = MLPLayer(w_enemy_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)

        self.skill_dim = self.args.skill_dim
        self.skill_head = nn.Linear(self.hidden_size, self.skill_dim)

    def forward(self, states, actions, task):
        # states and actions: (bs*n_agent, shape)

        x = states
        own_x, ally_x, enemy_x = self.decomposer.decompose(x, task) # torch.Size([1, 2592, 34]) torch.Size([2, 2592, 27]) torch.Size([3, 2592, 9])

        n_enemy = self.decomposer.get_n_enemy(task)
        onehot_action = F.one_hot(actions.reshape(-1), num_classes=self.n_no_attack_action+n_enemy)
        # print("0: ", n_enemy, self.n_no_attack_action, onehot_action, onehot_action.shape) # 3, 6, torch.Size([2592, 9])
        no_attack_action, attack_action, compact_action = self.decomposer.decompose_action(onehot_action, task)
        # print("2: ", attack_action.shape, compact_action.shape) # torch.Size([2592, 3]) torch.Size([2592, 7])
        attack_action = attack_action.transpose(0, 1).unsqueeze(-1)
        enemy_x = th.cat([enemy_x, attack_action], dim=-1)
        own_x = th.cat([own_x, compact_action.unsqueeze(0)], dim=-1)
        
        if self._use_feature_normalization:
            own_x = self.own_feature_norm(own_x)
            ally_x = self.ally_feature_norm(ally_x)
            enemy_x = self.enemy_feature_norm(enemy_x)
        
        own_x = self.own_mlp(own_x)
        ally_x = self.ally_mlp(ally_x)
        enemy_x = self.enemy_mlp(enemy_x)

        # use self attention to aggregate
        entity_s_embed = th.cat([own_x, ally_x, enemy_x], dim=0) # torch.Size([6, 3, 64])
        s_proj_query = self.state_query(entity_s_embed).permute(1, 0, 2) # torch.Size([3, 6, 8])
        s_proj_key = self.state_key(entity_s_embed).permute(1, 2, 0) # torch.Size([3, 8, 6])
        s_energy = th.bmm(s_proj_query / (self.attn_embed_dim ** (1 / 2)), s_proj_key) # torch.Size([3, 6, 6])
        s_attn_score = F.softmax(s_energy, dim=2) # torch.Size([3, 6, 6])
        s_proj_value = entity_s_embed.permute(1, 0, 2) # torch.Size([3, 6, 64])
        s_attn_out = th.bmm(s_attn_score, s_proj_value)[:, 0] # torch.Size([3, 64])
        # print(entity_s_embed.shape, s_proj_query.shape, s_proj_key.shape, s_energy.shape, s_attn_score.shape, s_proj_value.shape, s_attn_out.shape)
        # TODO; add relu
        return self.skill_head(s_attn_out)


class Decoder(MultiTaskModuleBase): 
    def __init__(self, args, decomposer, action_space, device, is_hier=False):
        self.action_space = action_space
        self.is_hier = is_hier
        super(Decoder, self).__init__(args, decomposer, device)
    
    def _build_layers(self):
        w_own_shape, w_ally_shape, w_enemy_shape, self.n_no_attack_action = self.decomposer.get_task_independent_shape()

        if self._use_feature_normalization:
            self.own_feature_norm = nn.LayerNorm(w_own_shape)
            self.ally_feature_norm = nn.LayerNorm(w_ally_shape)
            self.enemy_feature_norm = nn.LayerNorm(w_enemy_shape)

        self.own_mlp = MLPLayer(w_own_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.ally_mlp = MLPLayer(w_ally_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.enemy_mlp = MLPLayer(w_enemy_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)

        self.skill_dim = self.args.skill_dim
        if self.is_hier:
            self.skill_emb = nn.Linear(2 * self.skill_dim, self.hidden_size)
        else:
            self.skill_emb = nn.Linear(self.skill_dim, self.hidden_size)

        self._recurrent_N = self.args.recurrent_N
        self._gain = self.args.gain
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.act = ACTLayer(self.action_space, self.hidden_size * 2, self._use_orthogonal, self._gain)
    
    def forward(self, obs, rnn_states, masks, task, skill, available_actions=None, deterministic=False, actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        skill = check(skill).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        x = obs
        own_x, ally_x, enemy_x = self.decomposer.decompose(x, task)
        
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
        actor_features = th.bmm(s_attn_score, s_proj_value)[:, 0] # torch.Size([3, 64])

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        skill_out = self.skill_emb(skill)
        actor_features = th.cat([actor_features, skill_out], dim=-1)

        if actions is not None:
            actions_log_probs, _ = self.act.evaluate_actions(actor_features, actions, available_actions)
            return actions_log_probs, rnn_states
        
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states




        
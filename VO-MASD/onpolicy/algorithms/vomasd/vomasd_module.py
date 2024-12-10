import torch as th
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.mlp import MLPLayer
from algorithms.utils.transformer import init_, LightEncodeBlock, MultiAgentTransformer
from algorithms.odis.odis_module import MultiTaskModuleBase
from typing import DefaultDict
from algorithms.utils.util import init
from algorithms.utils.popart import PopArt


class TrajEncoder(MultiTaskModuleBase): 

    def __init__(self, args, decomposer, device):
        self.skill_dim = args.skill_dim
        self.dropout = args.dropout
        super(TrajEncoder, self).__init__(args, decomposer, device)
    
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

        self.traj_agg = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.skill_head = nn.Linear(2 * self.hidden_size, self.skill_dim)
        
    
    def forward(self, obses, actions, task): 
        # print(obses.shape) # [32, 10, 3, 30]
        bs, skill_len, n_agent = obses.shape[0], obses.shape[1], obses.shape[2]

        tot_bs = bs * skill_len * n_agent
        obses = obses.reshape(tot_bs, -1)
        actions = actions.reshape(tot_bs, -1)

        x = obses
        own_x, ally_x, enemy_x = self.decomposer.decompose(x, task) 

        n_enemy = self.decomposer.get_n_enemy(task)
        onehot_action = F.one_hot(actions.reshape(-1), num_classes=self.n_no_attack_action+n_enemy)

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
        entity_s_embed = th.cat([own_x, ally_x, enemy_x], dim=0) # torch.Size([6, tot_bs, 64])
        s_proj_query = self.state_query(entity_s_embed).permute(1, 0, 2) # torch.Size([tot_bs, 6, 8])
        s_proj_key = self.state_key(entity_s_embed).permute(1, 2, 0) # torch.Size([tot_bs, 8, 6])
        s_energy = th.bmm(s_proj_query / (self.attn_embed_dim ** (1 / 2)), s_proj_key) # torch.Size([tot_bs, 6, 6])
        s_attn_score = F.softmax(s_energy, dim=2) # torch.Size([tot_bs, 6, 6])
        s_proj_value = entity_s_embed.permute(1, 0, 2) # torch.Size([tot_bs, 6, 64])
        outputs = th.bmm(s_attn_score, s_proj_value)[:, 0] # torch.Size([tot_bs, 64])

        skill_inputs = outputs.reshape(bs, skill_len, n_agent, self.hidden_size)
        skill_inputs = skill_inputs.permute(0, 2, 1, 3)
        skill_inputs = skill_inputs.reshape(bs*n_agent, skill_len, self.hidden_size)

        gru_outs, _ = self.traj_agg(skill_inputs)
        skill_embs = self.skill_head(gru_outs[:, -1, :])
        # print("4: ", skill_inputs.shape, gru_outs.shape, skill_embs.shape) # torch.Size([96, 10, 64]) torch.Size([96, 10, 128]) torch.Size([96, 8])
        
        return skill_embs

class TopSkillEncoder(nn.Module):
    def __init__(self, args, max_group_size, device):
        super(TopSkillEncoder, self).__init__()
        self.skill_dim = args.skill_dim
        self.hidden_size = args.hidden_size
        self.n_head = 1

        self.skill_encoder = nn.Sequential(nn.LayerNorm(self.skill_dim),
                                         init_(nn.Linear(self.skill_dim, self.hidden_size), activate=True), nn.GELU()) 

        self.ln = nn.LayerNorm(self.hidden_size)
        self.blocks = LightEncodeBlock(self.hidden_size, self.n_head, max_group_size) # TODO: do a lightweight version
        self.head = init_(nn.Linear(self.hidden_size, self.skill_dim)) 

        self.to(device)
    
    def forward(self, skill, mask):
        skill_embeddings = self.skill_encoder(skill) 
        # print("0: ", skill_embeddings.shape, mask.unsqueeze(1).repeat(1, self.n_head, 1, 1).shape) # torch.Size([191, 3, 64]) torch.Size([191, 1, 3, 3])
        rep = self.blocks.forward(self.ln(skill_embeddings), mask.unsqueeze(1).repeat(1, self.n_head, 1, 1))  # torch.Size([1404, 3, 64])
        out = self.head(rep) 
        out_mask = mask[:, 0:1].float()
        out_mask = out_mask / th.sum(out_mask, dim=-1, keepdim=True).repeat(1, 1, out_mask.shape[-1])
        masked_out = out_mask @ out
        # print("1: ", rep.shape, out.shape, out_mask, out_mask.shape, masked_out.shape, masked_out[:, 0]) 
        # torch.Size([191, 3, 64]) torch.Size([191, 3, 8]) torch.Size([191, 1, 3]) torch.Size([191, 1, 8])

        return masked_out


class BottomSkillEncoder(nn.Module):
    def __init__(self, args, device):
        super(BottomSkillEncoder, self).__init__()
        skill_dim = args.skill_dim
        n_hidden = args.hidden_size
        # top_z_q_decoder = nn.Linear(skill_dim, skill_dim)
        self.top_z_q_decoder = nn.Sequential(nn.Linear(skill_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, skill_dim))
        self.syn_layer = nn.Sequential(nn.Linear(2 * skill_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, skill_dim))
        self.to(device)
    
    def forward(self, btm_z_e, top_z_q):
        top_z_q = self.top_z_q_decoder(top_z_q)
        last_input = th.cat([btm_z_e, top_z_q], dim=-1)
        output = self.syn_layer(last_input)
        
        return output


class Grouper(MultiTaskModuleBase): # TODO: only use state or obses as input

    def __init__(self, args, decomposer, max_group_size, max_skill_size, device):
        self.max_group_size = max_group_size
        self.max_skill_size = max_skill_size
        self.device = device
        super(Grouper, self).__init__(args, decomposer, device)
        

    def _build_layers(self):
        w_own_shape, w_ally_shape, w_enemy_shape, self.n_no_attack_action = self.decomposer.get_task_independent_shape()

        if self._use_feature_normalization:
            self.own_feature_norm = nn.LayerNorm(w_own_shape)
            self.ally_feature_norm = nn.LayerNorm(w_ally_shape)
            self.enemy_feature_norm = nn.LayerNorm(w_enemy_shape)

        self.own_mlp = MLPLayer(w_own_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.ally_mlp = MLPLayer(w_ally_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.enemy_mlp = MLPLayer(w_enemy_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_hidden = init_(nn.Linear(self.hidden_size, self.hidden_size))
        if self.args.use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=self.device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        # MAT (multi-agent transformer)
        self.mat_ln = nn.LayerNorm(self.hidden_size)
        self.mat = MultiAgentTransformer(action_dim=self.max_group_size, max_n_agent=self.max_group_size, max_skill_size=self.max_skill_size,
                                         n_block=1, n_embd=64, n_head=1, device=self.device) # so far, decoder-only
        self.idt = nn.Identity()

    def _preprocess_state(self, states, task):
        x = states.reshape(-1, states.shape[-1])
        own_x, ally_x, enemy_x = self.decomposer.decompose(x, task) # torch.Size([1, 2592, 34]) torch.Size([2, 2592, 27]) torch.Size([3, 2592, 9])
        
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

        v_hidden = F.relu(self.v_hidden(s_attn_out))
        v_out = self.v_out(v_hidden)
        
        s_attn_out = self.mat_ln(s_attn_out)

        return s_attn_out, v_out

    
    def forward(self, states, pre_skill_emb, task, test_mode, second_stage=False): # TODO: contain agent_id or 
        # print("0: ", states.shape, pre_skill_embs.shape) 

        ## process obs
        bs, seq_len, n_agent = states.shape[0], states.shape[1], states.shape[2]
        s_attn_out, v_out = self._preprocess_state(states, task) 
        s_attn_out = s_attn_out.reshape(bs*seq_len, n_agent, -1) # torch.Size([800, 3, 64])
        ## use a transformer as the final aggregator
        avai_actions = th.zeros(size=(bs*seq_len, n_agent, self.max_group_size), device=self.device, dtype=th.float32)
        avai_actions[:, :, :n_agent] = 1.0

        # deterministic is set as True, as this step is not mainly for training the grouper
        if not second_stage:
            grouping, log_group, logits = self.mat.get_actions(state_rep=s_attn_out, available_actions=avai_actions,\
                                                                        deterministic=test_mode, second_stage=second_stage)
        else:
            grouping, log_group, logits, avai_groups = self.mat.get_actions(state_rep=s_attn_out, available_actions=avai_actions,\
                                                                                     deterministic=test_mode, second_stage=second_stage)
        
        probs = F.softmax(logits, dim=-1)[:, :, :n_agent]
        # print("7: ", grouping.shape, logits.shape, probs.shape, probs) #  torch.Size([800, 3, 1]) torch.Size([800, 3, 10]) torch.Size([800, 3, 3])

        # multiply pre_skill_emb with probs
        pre_skill_emb = pre_skill_emb.reshape(bs*seq_len, n_agent, pre_skill_emb.shape[-1])
        # temp_ls = []
        # for i in range(n_agent):
        #     temp_ls.append(self.idt(pre_skill_emb))
        # temp_skill_emb = th.stack(temp_ls, dim=2)
        # skill_emb = (probs.unsqueeze(-2) @ temp_skill_emb).squeeze(-2)
        skill_emb = pre_skill_emb
        # print("8: ", pre_skill_emb.shape, temp_skill_emb.shape, probs.unsqueeze(-2).shape, (probs.unsqueeze(-2) @ temp_skill_emb).shape, skill_emb.shape)
        # # torch.Size([128, 3, 8]) torch.Size([128, 3, 3, 8]) torch.Size([128, 3, 1, 3]) torch.Size([128, 3, 1, 8]) torch.Size([128, 3, 8])

        grouping_rlt = []
        for i in range(bs * seq_len):
            temp_dict_idx = DefaultDict(list)
            temp_dict_skill = DefaultDict(list)
            for j in range(n_agent):
                temp_dict_idx[grouping[i][j].item()].append(j)
                temp_dict_skill[grouping[i][j].item()].append(skill_emb[i][j])
            i_dict = {}
            for k in temp_dict_idx:
                i_dict[tuple(temp_dict_idx[k])] = th.stack(temp_dict_skill[k], dim=0).reshape(1, -1)
            grouping_rlt.append(i_dict)

        # print("9: ", grouping_rlt, grouping.device)

        if not second_stage:
            return grouping_rlt, skill_emb, grouping, log_group, v_out
        return grouping_rlt, skill_emb, grouping, log_group, v_out, avai_groups
    
    
    def evaluate_actions(self, state_batch, act_batch, task):
        bs, n_agent = state_batch.shape[0], state_batch.shape[1]
        s_attn_out, v_out = self._preprocess_state(state_batch, task)
        s_attn_out = s_attn_out.reshape(bs, n_agent, -1)
        v_out = v_out.reshape(bs, n_agent, -1)

        avai_actions = th.zeros(size=(bs, n_agent, self.max_group_size), device=self.device, dtype=th.float32)
        avai_actions[:, :, :n_agent] = 1.0

        action_log_probs, entropy = self.mat(s_attn_out, act_batch, avai_actions) # torch.Size([3200, 3, 1]) torch.Size([3200, 3, 1])
    
        # act_num = action_log_probs.shape[-1]
        # action_log_probs = action_log_probs.reshape(-1, act_num) 
        # v_out = v_out
        # entropy = entropy.reshape(-1, act_num) 

        return v_out, action_log_probs, entropy
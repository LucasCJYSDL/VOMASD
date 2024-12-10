import torch

from algorithms.odis.odis_module import MultiTaskDecomposer, Decoder
from algorithms.vomasd.vomasd_module import TrajEncoder, Grouper
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic

from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from algorithms.utils.util import check, Item
from typing import DefaultDict
import itertools
import heapq


class VOMASD3dPolicy(R_MAPPOPolicy):

    def __init__(self, args, obs_space, cent_obs_space, act_space, obs_shapes, state_shapes, \
                 skill_space, max_group_size, max_skill_size, device=torch.device("cpu")):
        # super(VOMASD3dPolicy, self).__init__()
        self.online_task = args.map_name
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.skill_dim = args.skill_dim
        self.skill_num = args.skill_num
        self.max_skill_size = max_skill_size
        self.beta = args.beta

        self.rule_based = args.rule_based
        self.mixed = (args.algorithm_name == "vomasd-mixed")
        self.collected = False

        self.obs_decomposer = MultiTaskDecomposer(args, obs_shapes)
        self.state_decomposer = MultiTaskDecomposer(args, state_shapes)

        self.traj_encoder = TrajEncoder(args, self.obs_decomposer, device)
        self.decoder = Decoder(args, self.obs_decomposer, act_space, device)

        self.actor = R_Actor(args, obs_space, action_space=skill_space, device=device)
        self.critic = R_Critic(args, cent_obs_space, device)

        # new member
        self.grouper = Grouper(args, self.state_decomposer, max_group_size, max_skill_size, device)

        self.codebooks = []
        self.codebooks_freq = {}
        for i in range(max_skill_size):
            temp_emb = torch.nn.Embedding(self.skill_num, (i + 1) * self.skill_dim) 
            temp_emb.weight.data.uniform_(-1.0 / self.skill_num, 1.0 / self.skill_num)
            self.codebooks.append(temp_emb.to(self.device))
            self.codebooks_freq[i+1] = [0 for _ in range(self.skill_num)]
            # print(temp_emb.weight, temp_emb.weight.shape)
        self.codebooks = torch.nn.ModuleList(self.codebooks)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        self.pretrain_parameters = list(self.traj_encoder.parameters()) + list(self.decoder.parameters()) +\
                                   list(self.grouper.parameters()) + list(self.codebooks.parameters())
        
        self.pretrain_optimizer = torch.optim.Adam(self.pretrain_parameters,
                                                   lr=self.lr,
                                                   eps=self.opti_eps,
                                                   weight_decay=self.weight_decay)
        
        self.model_list = [self.traj_encoder, self.decoder, self.codebooks, self.grouper]
        self.name_list = ["traj_encoder", "decoder", "codebooks", "grouper"]
    
    def collect_codes(self):
        # threshold = 0.0
        tot_num = 0
        for i in range(self.max_skill_size):
            tot_num += sum(self.codebooks_freq[i+1])
        threshold = float(tot_num) / float(self.max_skill_size * self.skill_num * 2) # TODO: important threshold
        if self.mixed:
            self.codes = []
            for i in range(self.max_skill_size):
                for j in range(self.skill_num):
                    if self.codebooks_freq[i+1][j] >= threshold:
                        self.codes.append(self.codebooks[i].weight[j].reshape(-1, self.skill_dim))
        
            self.codes = torch.cat(self.codes, dim=0)
            # print("here: ", self.codes.shape) # torch.Size([30, 8])
        elif self.rule_based:
            self.codes = DefaultDict(list)
            for i in range(self.max_skill_size):
                for j in range(self.skill_num):
                    if self.codebooks_freq[i+1][j] >= threshold:
                        self.codes[i+1].append(self.codebooks[i].weight[j])
            
            for k in self.codes:
                self.codes[k] = torch.stack(self.codes[k], dim=0) # contains gradient info
            # print("0: ", self.codes)

    def _rule_based_match(self, z_e):
        ori_z_e = z_e
        n_agent = z_e.shape[-2]
        z_e = z_e.reshape(-1, n_agent, self.skill_dim)
        bs = z_e.shape[0]
        z_q = []
        for b in range(bs):
            b_code = {}
            min_heap = []
            for i in self.codes:
                # print("3: ", i)
                i_groups = list(itertools.combinations(range(n_agent), i))
                for g in i_groups:
                    temp_code = z_e[b][list(g)].reshape(-1)
                    for j in range(self.codes[i].shape[0]):
                        # TODO: store the id rather than code
                        heapq.heappush(min_heap, Item(group=list(g), code=self.codes[i][j], distance=((temp_code - self.codes[i][j])**2.0).sum()/float(i)))
            # print("0: ", min_heap)
            assigned = torch.tensor([False for _ in range(n_agent)])
            while not assigned.all().item():
                temp_item = heapq.heappop(min_heap)
                # print("1: ", temp_item, assigned)
                if assigned[temp_item.group].any().item():
                    continue
                assigned[temp_item.group] = True
                temp_code = temp_item.code
                for i in range(len(temp_item.group)):
                    b_code[temp_item.group[i]] = temp_code[i*self.skill_dim:(i+1)*self.skill_dim]
            b_z_q = []
            for i in range(n_agent):
                b_z_q.append(b_code[i])
            z_q.append(torch.stack(b_z_q, dim=0))
        z_q = torch.stack(z_q, dim=0)
        # print("2: ", z_q.shape) # torch.Size([1, 3, 5])
        return z_q.view(ori_z_e.shape)
    
    def _single_match(self, z_e): # for fair comparisons with other methods, we still use continuous skill embeddings and macthing-based skill selection
        z_e_flattened = z_e.reshape(-1, self.skill_dim)
        d = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codes ** 2, dim=1) - 2 * \
            torch.matmul(z_e_flattened, self.codes.t())
        # print("0: ", z_e_flattened.shape, self.codes.shape, d.shape) # torch.Size([3, 8]) torch.Size([30, 8]) torch.Size([3, 30])
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codes.shape[0]).to(z_e.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.codes).view(z_e.shape)
        # print("1: ", min_encoding_indices, min_encodings.shape, torch.matmul(min_encodings, self.codes).shape) 
        # torch.Size([3, 75]) torch.Size([3, 5])
        return z_q
    
    def forward_code(self, states, z_e, task, test_mode, second_stage=False):
        # torch.Size([32, 25, 3, 117]) torch.Size([32, 25, 3, 5])
        states = check(states).to(**self.tpdv)
        z_e = check(z_e).to(**self.tpdv)
        bs, seq_len, n_agent = states.shape[0], states.shape[1], states.shape[2]

        if second_stage:
            if not self.collected:
                self.collected = True
                self.collect_codes()
            if self.mixed:
                return self._single_match(z_e)
            if self.rule_based:
                return self._rule_based_match(z_e)

        if second_stage:
            grouping_rlt, z_e, group, log_group, v_out, _ = self.grouper(states, z_e, task, test_mode=test_mode,\
                                                                         second_stage=second_stage) # specific to vomasd
        else:
            grouping_rlt, z_e, group, log_group, v_out = self.grouper(states, z_e, task, test_mode=test_mode)
        # print(grouping_rlt[0], btm_z_e.shape, group.shape, log_group.shape) 
        # torch.Size([800, 3, 5]) torch.Size([800, 3, 1]) torch.Size([800, 3, 1])
        
        z_q = []
        for i in range(len(grouping_rlt)):
            code_dict = {}
            for k in grouping_rlt[i]:
                skill_size = len(k)
                z_flattened = grouping_rlt[i][k]
                d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.codebooks[skill_size-1].weight**2, dim=1) -\
                    2 * torch.matmul(z_flattened, self.codebooks[skill_size-1].weight.t())
                idx = torch.argmin(d, dim=1)[0].item()
                # print("0: ", d, d.shape, idx) # torch.Size([1, 5]) 4
                # keep track of the frequency for each code
                if not second_stage: # TODO: start to track the frequency in a later stage
                    self.codebooks_freq[skill_size][idx] += 1
                temp_code = self.codebooks[skill_size-1].weight[idx].reshape(skill_size, -1)
                # print("1: ", temp_code.shape)
                code_id = 0
                for agent_id in k:
                    code_dict[agent_id] = temp_code[code_id]
                    code_id += 1
            code_ls = []
            for agent_id in range(n_agent):
                code_ls.append(code_dict[agent_id])
            z_q.append(torch.stack(code_ls, dim=0))
            # print("2: ", th.stack(code_ls, dim=0).shape)
        z_q = torch.stack(z_q, dim=0)
        # print("3: ", z_q.shape) # torch.Size([800, 3, 5])
    
        code_diff = torch.mean((z_q.detach() - z_e) ** 2, dim=-1, keepdim=True) +\
                   self.beta * torch.mean((z_q - z_e.detach()) ** 2, dim=-1, keepdim=True)
        emb_rwd = - torch.mean((z_q.detach() - z_e) ** 2, dim=-1, keepdim=True).detach().clone()
        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.reshape(bs, seq_len, n_agent, self.skill_dim)

        if second_stage:
            return z_q
        return z_q, code_diff, group, log_group, v_out, emb_rwd
    
    def save_pretrain(self, save_dir):
        for i in range(len(self.model_list)):
            torch.save(self.model_list[i].state_dict(), str(save_dir) + "/{}.pt".format(self.name_list[i]))
    
    def save(self, save_dir):
        pass
    
    def restore_pretrain(self, model_dir):
        for i in range(len(self.model_list)):
            self.model_list[i].load_state_dict(torch.load(str(model_dir) + "/{}.pt".format(self.name_list[i])))
    
    def get_real_actions(self, skills, obses, rnn_states, masks, avail_actions):

        actions, _, rnn_states = self.decoder(obses, rnn_states, masks, self.online_task, skills, \
                                              avail_actions, deterministic=True)
        
        return actions, rnn_states
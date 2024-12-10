import torch

from algorithms.odis.odis_module import MultiTaskDecomposer, Decoder
from algorithms.vomasd.vomasd_module import TrajEncoder, TopSkillEncoder, BottomSkillEncoder, Grouper
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic

from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from algorithms.utils.util import check


class VOMASDHierPolicy(R_MAPPOPolicy):

    def __init__(self, args, obs_space, cent_obs_space, act_space, obs_shapes, state_shapes, skill_space, max_group_size, max_skill_size, device=torch.device("cpu")):
        # super(VOMASDHierPolicy, self).__init__()
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
        self.top_skill_num = args.top_skill_num
        self.beta = args.beta

        self.obs_decomposer = MultiTaskDecomposer(args, obs_shapes)
        self.state_decomposer = MultiTaskDecomposer(args, state_shapes)

        # hierarchical codebooks
        self.top_codebook = torch.nn.Embedding(self.top_skill_num, self.skill_dim) 
        self.top_codebook.weight.data.uniform_(-1.0 / self.top_skill_num, 1.0 / self.top_skill_num)
        self.top_codebook.to(device)

        self.btm_codebook = torch.nn.Embedding(self.skill_num, self.skill_dim)
        self.btm_codebook.weight.data.uniform_(-1.0 / self.skill_num, 1.0 / self.skill_num)
        self.btm_codebook.to(device)

        self.traj_encoder = TrajEncoder(args, self.obs_decomposer, device)
        self.decoder = Decoder(args, self.obs_decomposer, act_space, device, is_hier=True)

        self.actor = R_Actor(args, obs_space, action_space=skill_space, device=device)
        self.critic = R_Critic(args, cent_obs_space, device)

        # new member
        self.grouper = Grouper(args, self.state_decomposer, max_group_size, max_skill_size, device)

        self.top_skill_encoder = TopSkillEncoder(args, max_group_size, device)
        self.btm_skill_encoder = BottomSkillEncoder(args, device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        self.pretrain_parameters = list(self.traj_encoder.parameters()) + list(self.decoder.parameters()) +\
                                   list(self.top_codebook.parameters()) + list(self.btm_codebook.parameters()) +\
                                   list(self.grouper.parameters()) + list(self.top_skill_encoder.parameters()) +\
                                   list(self.btm_skill_encoder.parameters())
        
        self.pretrain_optimizer = torch.optim.Adam(self.pretrain_parameters,
                                                   lr=self.lr,
                                                   eps=self.opti_eps,
                                                   weight_decay=self.weight_decay)
        
        self.model_list = [self.traj_encoder, self.decoder, self.top_codebook, \
                      self.btm_codebook, self.top_skill_encoder, self.btm_skill_encoder,\
                      self.grouper]
        self.name_list = ["traj_encoder", "decoder", "top_codebook", "btm_codebook", "top_skill_encoder",\
                     "btm_skill_encoder", "grouper"]
    
    def forward_codebook(self, z_e, is_top):
        def _code_match(z_e, codebook, skill_num):
            z_e_flattened = z_e.reshape(-1, self.skill_dim)
            d = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook.weight**2, dim=1) - 2 * \
                torch.matmul(z_e_flattened, codebook.weight.t())
            # print("4: ", d.shape) # torch.Size([384, 5])
            # find closest encodings
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(min_encoding_indices.shape[0], skill_num).to(z_e.device)
            # print("5: ", min_encodings.shape, min_encoding_indices.shape, min_encoding_indices) # torch.Size([384, 5]) torch.Size([384, 1])
            min_encodings.scatter_(1, min_encoding_indices, 1)
            # print("6 ", min_encodings.shape, min_encodings) # torch.Size([384, 5])
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings, codebook.weight).view(z_e.shape)
            return z_q
        
        if is_top:
            return _code_match(z_e, self.top_codebook, self.top_skill_num)
        return _code_match(z_e, self.btm_codebook, self.skill_num)
    
    
    def forward_code(self, states, btm_z_e, task, test_mode, second_stage=False):
        # torch.Size([32, 25, 3, 117]) torch.Size([32, 25, 3, 5])
        states = check(states).to(**self.tpdv)
        btm_z_e = check(btm_z_e).to(**self.tpdv)
        bs, seq_len, n_agent = states.shape[0], states.shape[1], states.shape[2]

        if second_stage:
            grouping_rlt, btm_z_e, group, log_group, v_out, _ = self.grouper(states, btm_z_e, task, test_mode=test_mode, second_stage=second_stage) # specific to vomasd
        else:
            grouping_rlt, btm_z_e, group, log_group, v_out = self.grouper(states, btm_z_e, task, test_mode=test_mode)
        # print(grouping_rlt[0], btm_z_e.shape, group.shape, log_group.shape) # torch.Size([800, 3, 5]) torch.Size([800, 3, 1]) torch.Size([800, 3, 1])
        
        # if you are using Python 3.7 or later, the order of keys when you enumerate a dictionary will be consistent,\
        # reflecting the order in which elements were added to the dictionary.
        # get the top-level skills

        btm_skills, btm_masks = [], []
        for g in grouping_rlt:
            for sg in g:
                sg_skills = g[sg].reshape(-1, self.skill_dim)
                sg_size = sg_skills.shape[0]
                padding_skills = torch.zeros(size=(n_agent, self.skill_dim), device=sg_skills.device, dtype=sg_skills.dtype)
                padding_skills[:sg_size] = padding_skills[:sg_size] + sg_skills
                # print("1: ", padding_skills) # contains gradient TODO: detach the gradient for bottom skill codes
                padding_masks = torch.zeros(size=(n_agent, n_agent), device=sg_skills.device, dtype=torch.int)
                padding_masks[:, :sg_size] = 1
                btm_skills.append(padding_skills)
                btm_masks.append(padding_masks)
        btm_skills = torch.stack(btm_skills, dim=0)
        btm_masks = torch.stack(btm_masks, dim=0)
        # print("2: ", btm_skills.shape, btm_masks.shape) # torch.Size([191, 3, 8]) torch.Size([191, 3, 3])
        pre_top_skills = self.top_skill_encoder(btm_skills, btm_masks)
        
        top_skill_ls = []
        b_id = 0
        for g in grouping_rlt:
            top_skill_dict = {}
            for sg in g:
                for agent_id in list(sg):
                    top_skill_dict[agent_id] = pre_top_skills[b_id]
                b_id += 1
            top_skill = []
            for i in range(n_agent):
                top_skill.append(top_skill_dict[i])
            top_skill_ls.append(torch.cat(top_skill, dim=0))
        top_skills = torch.stack(top_skill_ls, dim=0)
        top_z_e = top_skills # torch.Size([800, 3, 5])
        # top_z_q, btm_z_q = self.mac.agent.forward_code(top_z_e, btm_z_e)

        top_z_q = self.forward_codebook(top_z_e, is_top=True)
        top_diff = torch.mean((top_z_q.detach() - top_z_e) ** 2, dim=-1, keepdim=True) +\
                   self.beta * torch.mean((top_z_q - top_z_e.detach()) ** 2, dim=-1, keepdim=True)
        top_z_q = top_z_e + (top_z_q - top_z_e).detach()


        # btm_z_e = self.btm_skill_encoder(btm_z_e, top_z_q) # TODO: comment this line, torch.Size([800, 3, 5])
        btm_z_q = self.forward_codebook(btm_z_e, is_top=False) 
        btm_diff = torch.mean((btm_z_q.detach() - btm_z_e) ** 2, dim=-1, keepdim=True) +\
                   self.beta * torch.mean((btm_z_q - btm_z_e.detach()) ** 2, dim=-1, keepdim=True)
        btm_z_q = btm_z_e + (btm_z_q - btm_z_e).detach()

        emb_rwd = - (torch.mean((btm_z_q.detach() - btm_z_e) ** 2, dim=-1, keepdim=True) +\
                     torch.mean((top_z_q.detach() - top_z_e) ** 2, dim=-1, keepdim=True)).detach().clone()

        top_z_q = top_z_q.reshape(bs, seq_len, n_agent, self.skill_dim)
        btm_z_q = btm_z_q.reshape(bs, seq_len, n_agent, self.skill_dim)
        z_q = torch.cat([top_z_q, btm_z_q], dim=-1)

        if second_stage:
            return z_q
        return z_q, top_diff, btm_diff, group, log_group, v_out, emb_rwd
    
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
        

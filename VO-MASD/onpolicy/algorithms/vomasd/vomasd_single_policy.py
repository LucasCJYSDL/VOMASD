import torch
import numpy as np
from algorithms.odis.odis_module import MultiTaskDecomposer, Decoder
from algorithms.vomasd.vomasd_module import TrajEncoder
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic

from gym import spaces
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from algorithms.utils.util import check


class VOMASDSinglePolicy(R_MAPPOPolicy):

    def __init__(self, args, obs_space, cent_obs_space, act_space, obs_shapes, state_shapes, skill_space, device=torch.device("cpu")):

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

        self.obs_decomposer = MultiTaskDecomposer(args, obs_shapes)

        self.codebook = torch.nn.Embedding(args.skill_num, args.skill_dim) 
        self.codebook.weight.data.uniform_(-1.0 / args.skill_num, 1.0 / args.skill_num)
        self.codebook.to(device)

        self.traj_encoder = TrajEncoder(args, self.obs_decomposer, device)
        self.decoder = Decoder(args, self.obs_decomposer, act_space, device)

        self.actor = R_Actor(args, obs_space, action_space=skill_space, device=device)
        self.critic = R_Critic(args, cent_obs_space, device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        self.pretrain_parameters = list(self.traj_encoder.parameters()) + list(self.decoder.parameters()) + list(self.codebook.parameters())
        
        self.pretrain_optimizer = torch.optim.Adam(self.pretrain_parameters,
                                                   lr=self.lr,
                                                   eps=self.opti_eps,
                                                   weight_decay=self.weight_decay)

    def forward_code(self, z_e):
        z_e = check(z_e).to(**self.tpdv)
        z_e_flattened = z_e.reshape(-1, self.skill_dim)
        d = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - 2 * \
            torch.matmul(z_e_flattened, self.codebook.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.skill_num).to(z_e.device)
        # print("5: ", min_encodings.shape, min_encoding_indices.shape, min_encoding_indices) # torch.Size([384, 5]) torch.Size([384, 1])
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # print("6 ", min_encodings.shape, min_encodings) # torch.Size([384, 5])
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.codebook.weight).view(z_e.shape)
        return z_q
    
    def save_pretrain(self, save_dir):
        torch.save(self.codebook.state_dict(), str(save_dir) + "/codebook.pt")
        torch.save(self.traj_encoder.state_dict(), str(save_dir) + "/traj_encoder.pt")
        torch.save(self.decoder.state_dict(), str(save_dir) + "/decoder.pt")
    
    def save(self, save_dir):
        pass
    
    def restore_pretrain(self, model_dir):
        self.codebook.load_state_dict(torch.load(str(model_dir) + "/codebook.pt"))
        self.traj_encoder.load_state_dict(torch.load(str(model_dir) + "/traj_encoder.pt"))
        self.decoder.load_state_dict(torch.load(str(model_dir) + "/decoder.pt"))
    
    def get_real_actions(self, skills, obses, rnn_states, masks, avail_actions):

        actions, _, rnn_states = self.decoder(obses, rnn_states, masks, self.online_task, skills, \
                                              avail_actions, deterministic=True)
        
        return actions, rnn_states
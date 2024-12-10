import torch
from algorithms.odis.odis_module import StateEncoder, MultiTaskDecomposer, Decoder
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic

from gym import spaces
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


class ODISPolicy(R_MAPPOPolicy):

    def __init__(self, args, obs_space, cent_obs_space, act_space, obs_shapes, state_shapes, skill_space, device=torch.device("cpu")):

        self.online_task = args.map_name

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.skill_dim = args.skill_dim

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.obs_decomposer = MultiTaskDecomposer(args, obs_shapes)
        self.state_decomposer = MultiTaskDecomposer(args, state_shapes)

        self.state_encoder = StateEncoder(args, self.state_decomposer, device)
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
        
        self.pretrain_parameters = list(self.state_encoder.parameters()) + list(self.decoder.parameters())
        
        self.pretrain_optimizer = torch.optim.Adam(self.pretrain_parameters,
                                                   lr=self.lr,
                                                   eps=self.opti_eps,
                                                   weight_decay=self.weight_decay)
    
    
    def get_real_actions(self, skills, obses, rnn_states, masks, avail_actions):

        actions, _, rnn_states = self.decoder(obses, rnn_states, masks, self.online_task, skills, \
                                              avail_actions, deterministic=True)
        
        return actions, rnn_states

    def save_pretrain(self, save_dir):
        torch.save(self.state_encoder.state_dict(), str(save_dir) + "/state_encoder.pt")
        torch.save(self.decoder.state_dict(), str(save_dir) + "/decoder.pt")

    def save(self, save_dir):
        pass
    
    def restore_pretrain(self, model_dir):
        state_encoder_state_dict = torch.load(str(model_dir) + '/state_encoder.pt')
        self.state_encoder.load_state_dict(state_encoder_state_dict)

        decoder_state_dict = torch.load(str(model_dir) + "/decoder.pt")
        self.decoder.load_state_dict(decoder_state_dict)
    
    def forward_code(self, skills):
        skills = skills.reshape(-1, 1)
        skills = torch.eye(self.skill_dim, device=self.device)[skills.squeeze(-1)] 
        
        return skills




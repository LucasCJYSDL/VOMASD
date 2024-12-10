import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from hmasd.utils.h_shared_buffer import H_SharedReplayBuffer
from hmasd.utils.l_shared_buffer import L_SharedReplayBuffer
from hmasd.utils.state_skill_dataset import StateSkillDataset
from hmasd.algorithms.mat.mat_trainer import MATTrainer as H_TrainAlgo
from hmasd.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as H_Policy
from hmasd.algorithms.r_mappo.r_mappo import R_MAPPO as L_TrainAlgo
from hmasd.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as L_Policy
from hmasd.algorithms.discriminator.d_trainer import D_Trainer
from hmasd.algorithms.discriminator.algorithm.discri_policy import DiscriPolicy as D_policy


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.use_sparse_reward = self.all_args.use_sparse_reward
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        self.use_linear_lambda_decay = self.all_args.use_linear_lambda_decay
        self.l_reward_mix_type = self.all_args.l_reward_mix_type

        self.h_entropy_coef_start = self.all_args.h_entropy_coef_start
        self.h_entropy_coef_end = self.all_args.h_entropy_coef_end
        self.h_entropy_coef_decay = self.all_args.h_entropy_coef_decay

        self.low_train_ratio = self.all_args.low_train_ratio

        # interval
        self.skill_interval = self.all_args.skill_interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        # policy network
        self.h_policy = H_Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 share_observation_space,
                                 self.num_agents,
                                 device=self.device)
        self.l_policy = L_Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 share_observation_space,
                                 self.envs.action_space[0],
                                 device=self.device)
        self.discri = D_policy(self.all_args,
                               self.envs.observation_space[0],
                               share_observation_space,
                               device=self.device)

        # algorithm
        self.h_trainer = H_TrainAlgo(self.all_args, self.h_policy, self.num_agents, device=self.device)
        self.l_trainer = L_TrainAlgo(self.all_args, self.l_policy, device = self.device)
        self.d_trainer = D_Trainer(self.all_args, self.discri, device = self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)
        
        # buffer
        self.h_buffer = H_SharedReplayBuffer(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space[0],
                                             share_observation_space,
                                             self.all_args.env_name)
        self.l_buffer = L_SharedReplayBuffer(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space[0],
                                             share_observation_space,
                                             self.envs.action_space[0])
        self.state_skill = StateSkillDataset(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space[0],
                                             share_observation_space)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def h_collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError
    
    def l_collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError
    
    def d_collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def h_insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    def l_insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    def d_insert(self, data, step):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.h_trainer.prep_rollout()
        h_next_values = self.h_trainer.policy.get_values(np.concatenate(self.h_buffer.share_obs[-1]),
                                                         np.concatenate(self.h_buffer.obs[-1])) # (n_roll*(n_agent+1), 1)
        h_next_values = np.array(np.split(_t2n(h_next_values), self.n_rollout_threads)) # (n_roll, n_agent+1, 1)
        self.h_buffer.compute_returns(h_next_values, self.h_trainer.value_normalizer)

        self.l_trainer.prep_rollout()
        # caculate the team skill of episode_len + 1 steps 
        _, h_action, _ = self.h_trainer.policy.get_actions(np.concatenate(self.h_buffer.share_obs[-1]),
                                                           np.concatenate(self.h_buffer.obs[-1])) # (n_roll*(n_agent+1), act_num)
        h_actions = np.array(np.split(_t2n(h_action), self.n_rollout_threads)) # (n_roll, n_agent+1, act_num)
        team_skill = h_actions[:, 0] 
        team_skill = np.expand_dims(team_skill, 1).repeat(self.num_agents, 1)  # (n_roll, n_agent, skill_num)
        indi_skill = h_actions[:, 1:] # (n_roll, n_agent, skill_num)

        l_next_values = self.l_trainer.policy.get_values(np.concatenate(self.l_buffer.share_obs[-1]),
                                                         np.concatenate(team_skill),
                                                         np.concatenate(indi_skill),
                                                         np.concatenate(self.l_buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.l_buffer.masks[-1])) # (n_roll*n_agent, 1) 
        l_next_values = np.array(np.split(_t2n(l_next_values), self.n_rollout_threads)) # (n_roll, n_agent, 1) 
        self.l_buffer.compute_returns(l_next_values, self.l_trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.h_trainer.prep_training()
        h_train_infos = self.h_trainer.train(self.h_buffer)      
        self.h_buffer.after_update()

        self.l_trainer.prep_training()
        l_train_infos = self.l_trainer.train(self.l_buffer)      
        self.l_buffer.after_update()

        self.d_trainer.prep_training()
        d_train_infos = self.d_trainer.train(self.state_skill)  

        train_infos = dict(**h_train_infos, **l_train_infos, **d_train_infos)

        return train_infos

    def save(self, episode):
        save_dir_episode = str(self.save_dir) + '/' + str(episode)
        if not os.path.exists(save_dir_episode):
            os.makedirs(save_dir_episode)

        # high policy
        self.h_policy.save(save_dir_episode)
        # low policy
        policy_actor = self.l_trainer.policy.actor
        torch.save(policy_actor.state_dict(), save_dir_episode + "/actor.pt")
        policy_critic = self.l_trainer.policy.critic
        torch.save(policy_critic.state_dict(), save_dir_episode + "/critic.pt")
        if self.l_trainer._use_valuenorm:
            policy_vnorm = self.l_trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), save_dir_episode + "/vnorm.pt")
        # discriminator
        self.discri.save(save_dir_episode)

    def restore(self, model_dir):
        # high policy
        self.h_policy.restore(model_dir)
        # low policy
        policy_actor_state_dict = torch.load(model_dir + '/actor.pt')
        self.l_policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(model_dir + '/critic.pt')
        self.l_policy.critic.load_state_dict(policy_critic_state_dict)
        if self.l_trainer._use_valuenorm:
            policy_vnorm_state_dict = torch.load(model_dir + '/vnorm.pt')
            self.l_trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
        # discriminator
        self.discri.restore(model_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym
from gym.spaces import Discrete
import numpy as np

class Overcooked():
    def __init__(self, args):
        self.map_name = args.map_name
        self.episode_length = args.episode_length
        mdp = OvercookedGridworld.from_layout_name(self.map_name)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=self.episode_length)
        self.env = gym.make("Overcooked-v0", base_env = base_env, featurize_fn =base_env.featurize_state_mdp)

        self.n_agents = 2
        self.n_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.state_dim = self.n_agents * self.obs_dim
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for _ in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append([self.obs_dim])
            self.share_observation_space.append([self.state_dim])
    
    def reset(self):
        obs = self.env.reset()
        obs = obs['both_agent_obs']
        local_obs = [o for o in obs]
        state = [np.concatenate(obs) for _ in range(self.n_agents)]
        return local_obs, state, self.get_avail_actions()

    def step(self, actions):
        actions = [int(a) for a in actions]
        actions = tuple(actions)
        obs, reward, done, env_info = self.env.step(actions)
        obs = obs['both_agent_obs']
        local_obs = [o for o in obs]
        state = [np.concatenate(obs) for _ in range(self.n_agents)]
        rewards = [[reward]]*self.n_agents
        dones = np.array([done] * self.n_agents)
        infos = [env_info for _ in range(self.n_agents)]

        return local_obs, state, rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)
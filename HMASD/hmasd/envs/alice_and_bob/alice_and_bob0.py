import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from gym.spaces import Discrete

class Alice_and_Bob(object):
    def __init__(self, map_size=(10, 10), n=2):
        self.length, self.width = map_size
        self.agent_num = n
        self.n_agents = n
        self.n_actions = 5
        self.generate_map()
        self.map = {'agent':{0: 2, 1: 3}, 'key':{0: 4, 1: 5}, 'goal':{0: 6, 1: 7}}
        self.episode_limit = 100
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for _ in range(self.agent_num):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append([self.get_obs_size()])
            self.share_observation_space.append([self.get_state_size()])
    
    def generate_map_obs(self):
        occupancy_obs = np.zeros((self.length, self.width, 3))
        occupancy_obs[self.occupancy==0] = [1,1,1]
        occupancy_obs[self.occupancy==2] = [1,0,0]
        occupancy_obs[self.occupancy==3] = [0,0,1]
        occupancy_obs[self.occupancy==4] = [0,1,0]
        occupancy_obs[self.occupancy==5] = [1,1,0]
        occupancy_obs[self.occupancy==6] = [0,1,1]
        occupancy_obs[self.occupancy==7] = [1,0,1]
        return occupancy_obs

    def generate_map(self):
        self.occupancy = np.zeros((self.length, self.width))
        
        # enclose the surroundings
        for i in range(self.length):
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.width - 1] = 1
        for i in range(self.width):
            self.occupancy[0, i] = 1
            self.occupancy[self.length -1, i] = 1

        # generate keys and goals
        self.keys_pos = [[1, self.width-2], [1, 1]]
        self.goals_pos = [[self.length-2, 1], [self.length-2, self.width-2]]
        self.keys_in_use = [False, False]
        self.goals_reach = [False, False]
        self.occupancy[self.keys_pos[0][0]][self.keys_pos[0][1]] = 4
        self.occupancy[self.keys_pos[1][0]][self.keys_pos[1][1]] = 5
        self.occupancy[self.goals_pos[0][0]][self.goals_pos[0][1]] = 6
        self.occupancy[self.goals_pos[1][0]][self.goals_pos[1][1]] = 7

        # initialize agents
        self.agt_pos = []
        ll, ww = self.length - 2, self.width - 2
        init_pos = np.random.choice(ll * ww, 2, replace=False)
        self.agt_pos.append([init_pos[0] // ww + 1, init_pos[0] % ww + 1])
        self.agt_pos.append([init_pos[1] // ww + 1, init_pos[1] % ww + 1])
        self.occupancy[self.agt_pos[0][0]][self.agt_pos[0][1]] = 2
        self.occupancy[self.agt_pos[1][0]][self.agt_pos[1][1]] = 3

        self.occupancy_obs = self.generate_map_obs()

    def reset(self):
        self._episode_steps = 0
        self.generate_map()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def step(self, action_list):
        action_list = [int(a) for a in action_list]
        self._episode_steps += 1
        reward = 0.0
        # agent move
        for i in range(self.agent_num):
            if action_list[i] == 0:  # move up
                if self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]+1] != 1:  # if can move
                    self.agt_pos[i][1] = self.agt_pos[i][1] + 1
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]-1] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]
            elif action_list[i] == 1:  # move down
                if self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]-1] != 1:  # if can move
                    self.agt_pos[i][1] = self.agt_pos[i][1] - 1
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]+1] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]
            elif action_list[i] == 2:  # move left
                if self.occupancy[self.agt_pos[i][0]-1][self.agt_pos[i][1]] != 1:  # if can move
                    self.agt_pos[i][0] = self.agt_pos[i][0] - 1
                    self.occupancy[self.agt_pos[i][0]+1][self.agt_pos[i][1]] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]
            elif action_list[i] == 3:  # move right
                if self.occupancy[self.agt_pos[i][0]+1][self.agt_pos[i][1]] != 1:  # if can move
                    self.agt_pos[i][0] = self.agt_pos[i][0] + 1
                    self.occupancy[self.agt_pos[i][0]-1][self.agt_pos[i][1]] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]

        # check keys
        for i in range(len(self.keys_pos)):
            if self.keys_pos[i] in self.agt_pos:
                self.keys_in_use[i] = True
            else:
                self.keys_in_use[i] = False
        
        # check goals
        for i in range(len(self.keys_pos)):
            if self.keys_in_use[i] and self.goals_pos[i] in self.agt_pos:
                self.goals_reach[i] = True
        
        for i in range(len(self.goals_pos)):
            if self.goals_pos[i] not in self.agt_pos:
                if self.goals_reach[i]:
                    self.occupancy[self.goals_pos[i][0]][self.goals_pos[i][1]] = 0
                else:
                    self.occupancy[self.goals_pos[i][0]][self.goals_pos[i][1]] = self.map['goal'][i]
        
        for i in range(len(self.keys_pos)):
            if not self.keys_in_use[i]:
                self.occupancy[self.keys_pos[i][0]][self.keys_pos[i][1]] = self.map['key'][i]

        done = False
        info_ = {}
        info_['battle_won'] = False
        # check treasure
        if np.all(self.goals_reach):
            reward = 1.0
            done = True
            info_['battle_won'] = True

        if self._episode_steps >= self.episode_limit:
            done = True

        if done:
            info_['key0'] = self.goals_reach[0]
            info_['key1'] = self.goals_reach[1]
        
        rewards = [[reward]]*self.agent_num
        dones = np.array([done] * self.agent_num)
        infos = [info_ for _ in range(self.agent_num)]

        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def get_global_obs(self):
        return self.generate_map_obs()

    def get_agt_obs(self, i):
        obs = self.generate_map_obs()[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2]
        return obs

    def get_all_agt_obs(self):
        return [self.get_agt_obs(i) for i in range(self.agent_num)]

    def get_partial_obs(self, i):
        # 3x3 surrounding env
        partial_obs = self.occupancy[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2].reshape(1,-1)

        # dis from agent to keys
        rel_agt_land_dis = np.zeros((2,2))
        for j in range(len(self.keys_pos)):
            rel_agt_land_dis[j] = np.array(self.agt_pos[i]) - np.array(self.keys_pos[j])
        
        rel_dis = np.zeros((2,2))
        for j in range(len(self.goals_pos)):
            if not self.goals_reach[j]:
                rel_dis[j] = np.array(self.agt_pos[i]) - np.array(self.goals_pos[j])

        # relative distance from agent to the ohter
        other_pos = np.array(self.agt_pos[i])-np.array(self.agt_pos[1-i])

        # return np.concatenate([np.squeeze(partial_obs), self.agt_pos[i], other_pos, rel_agt_land_dis.flatten(), rel_dis.flatten()])
        return np.concatenate([np.squeeze(partial_obs), self.agt_pos[i]])

    def get_obs(self):
        return [self.get_partial_obs(i) for i in range(self.agent_num)]
    
    def get_obs_size(self):
        return self.get_partial_obs(0).shape[0]

    def get_state_agent(self, agent_id):
        return np.concatenate([line_state for line_state in self.occupancy], axis = 0)

    def get_state(self):
        return [self.get_state_agent(i) for i in range(self.agent_num)]

    def get_state_size(self):
        return self.get_state_agent(0).shape[0]

    def plot_scene(self, idx):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])

        ax1.imshow(self.get_global_obs())
        ax2.imshow(self.get_agt_obs(0))
        ax3.imshow(self.get_agt_obs(1))
        plt.savefig('./alice_and_bob/images/step_{}'.format(idx))
        plt.clf()

    def render(self):

        obs = self.get_global_obs()
        enlarge = 30

        new_obs = np.zeros((self.length*enlarge, self.width*enlarge, 3))
        for i in range(self.length):
            for j in range(self.width):
                if np.sum(obs[i][j]) > 0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), obs[i][j][::-1]*255, -1)

        cv2.imshow('image', new_obs)
        cv2.waitKey(100)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.agent_num)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)
# ==============================================================
# file: gym_env.py
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from PIL import Image
from swimmer_model import target, MagSwimmer
from matplotlib.text import TextPath
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import os
import gym
from gym import spaces
from gym.spaces import Box
ARROW_MARKER = TextPath((0, 0), "/\/->")

# gym environment used for RL experiments
class MicroMagSwimmerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, num_of_bots = 4, L=5, dia=0.3, delta_t = 0.1, tfinal = 50, render_mode = 'human'):        
        self.num_of_bots = num_of_bots
        self.L = L #length of helical swimmer
        self.dia = dia #diameter of helical swimmer
        self.delta_t = delta_t
        self.tfinal = tfinal
        self.bot_list = []
        self.render_num = 0
        self.max_episode_steps = 100
        self.render_mode = render_mode
        
        # setting the dimension of the action space and state/observation space
        self.action_space = Box(-179.9,180,shape=(1,),dtype=np.float32)
        self.obs_space_low = np.array(
            [-np.inf]*2*self.num_of_bots + [-179.9]*self.num_of_bots + [-100.0, -100.0, 5.0])
        self.obs_space_high = np.array(
            [np.inf]*2*self.num_of_bots + [180.0]*self.num_of_bots + [100.0, 100.0, 20.0])
        observation_space = spaces.Box(
           low=self.obs_space_low, high=self.obs_space_high, dtype=np.float32)
        self.observation_space = observation_space

        self.set_goal(self.sample_goal_for_rollout())
        self.calc_state()
    
    # random target position
    def sample_goal_for_rollout(self):
        pos_min = [-100, -100]
        pos_max = [100, 100]
        rad_min = [5]
        rad_max = [20]
        return np.random.uniform(low=np.array(pos_min + rad_min), high=np.array(pos_max + rad_max))

    # setting the target state
    def set_goal(self, goal):
        self.target_env = target(goal[0], goal[1], goal[2])

    # start of an episode
    def reset(self, seed=None, options={'goal':None}):
        super().reset(seed=seed)
        print(options['goal'])
        row_n = int(np.ceil(np.sqrt(self.num_of_bots)))
        pos_max = int(np.floor(row_n*10/3.3))
        init_pos = [(x,y) for x in np.linspace(-pos_max,pos_max,row_n) for y in np.linspace(-pos_max,pos_max,row_n)]
        init_orn = 360*(self.np_random.random()-0.5)
        MagSwim_list = []
        for idx in range(self.num_of_bots):
            MagSwim_list.append(MagSwimmer(init_pos[idx], init_orn, self.L, self.dia, self.delta_t))
        self.bot_list = MagSwim_list
        self.calc_state()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
        
    # function called at every time step in the episode
    def step(self, action):
        f =10
        print('action taken: ',action)
        prev_bot_list = self.bot_list
        new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(prev_bot_list, f, action) for bot in self.bot_list)
        self.bot_list = new_bot_list
        self.calc_state()
        observation = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()
        return observation, reward, terminated, False, info
    
    # calculating swimmer state
    def calc_state(self):
        self.pos_x_list = [bot.pos_x for bot in self.bot_list]
        self.pos_y_list = [bot.pos_y for bot in self.bot_list]
        self.orn_list = [bot.orientation for bot in self.bot_list]
        while len(self.pos_x_list) != self.num_of_bots:
            self.pos_x_list.append(self.target_env.pos[0])
            self.pos_y_list.append(self.target_env.pos[1])
            self.orn_list.append(0)
          
    # calculating the environment state to be sent to the RL agent
    def _get_obs(self):
        obs = np.concatenate((np.array(self.pos_x_list, dtype=np.float32),np.array(self.pos_y_list, dtype=np.float32),np.array(self.orn_list, dtype=np.float32),self.target_env.pos.astype('float32'),np.array([self.target_env.rad], dtype=np.float32)))
        return obs
    
    # calculating the reward to be sent to the RL agent
    def _get_reward(self):
        new_bot_list = [bot for bot in self.bot_list if not bot.intarget(self.target_env)]
        num_in_target = len(self.bot_list) - len(new_bot_list)
        reward = num_in_target
        a = self._get_info()
        self.bot_list = new_bot_list
        if not new_bot_list or a['distance'] > 200:
            terminated = True
        else:
            terminated = False
        return reward, terminated
    
    #additional information sent for logging purpose
    def _get_info(self):
        _target_location = self.target_env.pos
        x_mean = np.mean(np.array(self.pos_x_list))
        y_mean = np.mean(np.array(self.pos_y_list))
        _agent_location = np.array([x_mean, y_mean])
        return {"distance": np.linalg.norm(_agent_location - _target_location),
                "target_pos": _target_location,
                "mean_pos": _agent_location,
                "target_radius": self.target_env.rad,
                "num_of_bots": self.num_of_bots,
                "bot_length": self.L,
                "bot_diameter": self.dia}
    
    # rendering the environment
    def render(self):
        plt.rcParams['figure.figsize'] = [14, 12]
        ax = plt.axes()
        marker_list = [MarkerStyle(ARROW_MARKER,transform = Affine2D().scale(5).rotate_deg(orn)) for orn in self.orn_list]
        for x,y,z in zip(self.pos_x_list,self.pos_y_list,marker_list):
            ax.plot(x,y,marker=z,color='blue')
        circle = plt.Circle(( self.target_env.pos[0] , self.target_env.pos[1] ),  self.target_env.rad,  color='#00ffff', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.grid(True)
        plt.xlim((-100,100))
        plt.ylim((-100,100))
        filename = f'im{self.render_num}.png'
        self.render_num += 1
        plt.savefig(os.path.join('imgs',filename))
        plt.clf()
        if self.render_mode == 'human':
            plt.draw()
        elif self.render_mode == 'rgb_array':
            image = Image.open(os.path.join('imgs',filename))
            return np.asarray(image)
        
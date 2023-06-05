# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:59:43 2021

@author: Akshatha
"""

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import shapely.geometry as sp
from joblib import Parallel, delayed

# =============================================================================
# Target
# =============================================================================
class target:
    def __init__(self, target_x, target_y, target_rad):
        self.pos = np.array([target_x, target_y])
        self.rad = target_rad

# =============================================================================
# An self-propelled particle
# =============================================================================
class Particle:
    def __init__(self,init_pos, v, R, delta_t, dens_dep_vel, vicinity_particle_max = 50, vicinity_rad = 10):
        #initializations
        #initial positions of swarm
        self.pos_x = init_pos[0];
        self.pos_y = init_pos[1];
        self.v = v #10;%speed (um per sec)
        self.R = R
        k_B = const.k#physconst('Boltzmann');
        T = 293; #Kelvin
        eta = 1.0016e-3; #N/m^2
        self.D_t = k_B*T/(6e-12*np.pi*eta*R); #(um^2/s)
        self.D_r = k_B*T/(8*np.pi*eta*pow(R,3)); #(rad^2/s)
        # self.D_t = D_t#0.2; %(um^2/s) k_B*T/(6*pi*eta*R);
        # self.D_r = D_r#0.170; %(rad^2/s) k_B*T/(8*pi*eta*R^3);
        self.timeval = 0
        self.delta_t = delta_t
        self.orientation = np.random.uniform(0,359.999)*np.pi/180
        
    def normalize(self, arr):
        mag = np.sqrt(pow(arr[0],2) + pow(arr[1],2))
        arr_normalized = arr/(mag + 1e-15)
        return arr_normalized
        
    def position(self):
        # print("The position is ", self.pos_x, self.pos_y)
        return (self.pos_x, self.pos_y)
        
    def euc_dist(self, x2, y2):
        return np.sqrt(pow(y2-self.pos_y,2)+pow(x2-self.pos_x,2))
      
    def intarget(self, target):
        if pow(self.pos_x - target.pos[0],2) + pow(self.pos_y - target.pos[1],2) < pow(target.rad,2):
            return True
        else:
            return False
        

# =============================================================================
# A bot or particle of the type ABP
# =============================================================================        
class ABP(Particle):
    def __init__(self,init_pos = (0,0), v=10, R=1e-6, delta_t=0.1, dens_dep_vel=0, vicinity_particle_max = 50, vicinity_rad = 10):
        super().__init__(init_pos, v, R, delta_t, dens_dep_vel, vicinity_particle_max, vicinity_rad)

    def next_pos(self, env, bot_list=[]):
        self.timeval = round(self.timeval + self.delta_t,1)
        #if self.timeval % 1 == 0.0:
        self.orientation = self.orientation + (np.sqrt(2*self.D_r*self.delta_t) * np.random.normal(0,1))
        pos_noise_x = np.sqrt(2*self.D_t*self.delta_t)*np.random.normal(0,1)
        pos_noise_y = np.sqrt(2*self.D_t*self.delta_t)*np.random.normal(0,1)
        x_addn = self.v*(np.cos(self.orientation))*self.delta_t + pos_noise_x
        y_addn = self.v*(np.sin(self.orientation))*self.delta_t + pos_noise_y
        self.pos_x = self.pos_x + x_addn
        self.pos_y = self.pos_y + y_addn
        # print(self.pos_x,self.pos_y)
        return self
    
# =============================================================================
# A bot or particle of the type RTP
# =============================================================================
class RTP(Particle):
    def __init__(self,init_pos = (0,0), v=10, R=1e-6, alpha=1.28, delta_t=0.1, dens_dep_vel=0, vicinity_particle_max = 50, vicinity_rad = 10):
        super().__init__(init_pos, v, R, delta_t, dens_dep_vel, vicinity_particle_max, vicinity_rad)
        self.alpha = alpha
        self.run_duration = round(np.random.exponential(scale=1/alpha),2)#first run duration
        while(self.run_duration == 0): self.run_duration = round(np.random.exponential(scale=1/alpha),1)
        self.present_run_time = 0
        
    def next_pos(self, env, bot_list=[]):
        self.timeval = self.timeval + self.delta_t
        self.present_run_time = round(self.present_run_time + self.delta_t,2)
        if self.present_run_time >= self.run_duration: #run cycle is over
            self.present_run_time = 0
            self.run_duration = round(np.random.exponential(scale=1/self.alpha),2)
            while(self.run_duration == 0): self.run_duration = round(np.random.exponential(scale=1/self.alpha),2)
            self.orientation += np.random.uniform(0,359.999)*np.pi/180
        pos_noise_x = np.sqrt(2*self.D_t*self.delta_t)*np.random.normal(0,1)
        pos_noise_y = np.sqrt(2*self.D_t*self.delta_t)*np.random.normal(0,1)
        x_addn = self.v*(np.cos(self.orientation))*(self.delta_t) + pos_noise_x
        y_addn = self.v*(np.sin(self.orientation))*(self.delta_t) + pos_noise_y
        self.pos_x = self.pos_x + x_addn
        self.pos_y = self.pos_y + y_addn
        return self
    
# =============================================================================
# A bot or particle of the type Chiral ABP
# =============================================================================        
class Chiral_ABP(Particle):
    def __init__(self, init_pos = (0,0), v=10, R=1e-6, w=1, delta_t=0.1, dens_dep_vel=0, vicinity_particle_max = 50, vicinity_rad = 10): 
        super().__init__(init_pos, v, R, delta_t, dens_dep_vel, vicinity_particle_max, vicinity_rad)
        self.w = w
        
    def next_pos(self, env, bot_list=[]):
        self.timeval = round(self.timeval + self.delta_t,1)
        self.orientation = self.orientation + self.w*self.delta_t + (np.sqrt(2*self.D_r*self.delta_t) * np.random.normal(0,1))
        pos_noise_x = np.sqrt(2*self.D_t*self.delta_t)*np.random.normal(0,1)
        pos_noise_y = np.sqrt(2*self.D_t*self.delta_t)*np.random.normal(0,1)
        x_addn = self.v*(np.cos(self.orientation))*(self.delta_t) + pos_noise_x
        y_addn = self.v*(np.sin(self.orientation))*(self.delta_t) + pos_noise_y
        self.pos_x = self.pos_x + x_addn
        self.pos_y = self.pos_y + y_addn
        return self
    
# =============================================================================
# environment class with obstacles and path (in the future)        
# =============================================================================
class environment:
    def __init__(self):
        #create_obstacles
        self.obstacles = []
        self.numofobstacles = len(self.obstacles)   
        
    def near_polygon(self,x,y,R):
        pinput = sp.Point(x,y)
        flag = False
        index = np.NAN
        for idx in range(self.numofobstacles):
            if self.obstacles[idx].distance(pinput) <= (2*R/1e-6):
                flag = True
                index = idx 
        return flag,index
        
    def disp_env(self):
        plt.grid(True)
        for idx in range(self.numofobstacles):
            plt.plot(*self.obstacles[idx].exterior.xy)
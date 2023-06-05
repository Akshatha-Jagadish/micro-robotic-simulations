# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:36:15 2021

@author: Akshatha
"""

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import shapely.geometry as sp
import time
import copy
from joblib import Parallel, delayed

# =============================================================================
# Target
# =============================================================================
class target:
    def __init__(self, target_x, target_y, target_z, target_rad):
        self.pos = np.array([target_x, target_y, target_z])
        self.rad = target_rad;
        
# =============================================================================
# A bot or particle of the type ABP
# =============================================================================
class ABP:
    def __init__(self,init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel):
        #initializations
        #initial positions of swarm
        self.pos = np.array([init_pos_x0, init_pos_y0, init_pos_z0])
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        self.orientation = self.normalize(temp)
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
        
    def normalize(self, arr):
        # mag = np.sqrt(pow(arr[0],2) + pow(arr[1],2) + pow(arr[2],2))
        # print(mag)
        mag = np.linalg.norm(arr)
        arr1 = arr/mag
        return arr1
        # print(arr)
        
    def position(self):
        print("The position is ", self.pos[0], self.pos[1], self.pos[2])
        
    def euc_dist(self, x2, y2, z2):
        return np.sqrt(pow(y2-self.pos[1],2)+pow(x2-self.pos[0],2)+pow(z2-self.pos[2],2))
    
    def intarget(self, target):
        if pow(self.pos[0] - target.pos[0],2) + pow(self.pos[1] - target.pos[1],2) + pow(self.pos[2] - target.pos[2],2) < pow(target.rad,2):
            return True
        else:
            return False
        
    def rotation(self, ref_vec):
        pass
        
    def next_pos(self, bot_list=[]):
        self.timeval = self.timeval + self.delta_t
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        temp = self.normalize(temp)
        # orientation_noise = np.sqrt(2*self.D_r*self.delta_t)*np.random.normal(0,1)
        # add = self.orientation + orientation_noise
        # cross = self.normalize(np.cross(temp,self.orientation))
        deltheta = np.sqrt(2*self.D_r*self.delta_t)*np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        theta = np.linalg.norm(deltheta)
        cross1 = np.cross(deltheta,self.orientation)
        cross2 = np.cross(deltheta, cross1)
        self.orientation = self.normalize(self.orientation + np.sin(theta)*cross1/theta + (1-np.cos(theta))*cross2/pow(theta,2))
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        pos_noise = np.sqrt(2*self.D_t*self.delta_t) * temp
        self.pos = self.pos + self.v*self.orientation*self.delta_t + pos_noise
        return self

# =============================================================================
# A bot or particle of the type RTP
# =============================================================================        
class RTP:
    def __init__(self,init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel):
        #initializations
        #initial positions of swarm
        self.pos = np.array([init_pos_x0, init_pos_y0, init_pos_z0])
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        self.orientation = self.normalize(temp)
        self.v = v #10;%speed (um per sec)
        self.R = R
        k_B = const.k#physconst('Boltzmann');
        T = 293; #Kelvin
        eta = 1.0016e-3; #N/m^2
        self.D_t = k_B*T/(6e-12*np.pi*eta*R); #(um^2/s)
        self.D_r = k_B*T/(8*np.pi*eta*pow(R,3)); #(rad^2/s)
        self.timeval = 0
        self.alpha = alpha
        # print(self.D_r)
        # print(self.alpha)
        self.delta_t = delta_t
        self.run_duration = np.random.exponential(scale=1/alpha)#first run duration
        while(self.run_duration == 0): self.run_duration = np.random.exponential(scale=1/alpha)
        self.present_run_time = 0
        
    def normalize(self, arr):
        # mag = np.sqrt(pow(arr[0],2) + pow(arr[1],2) + pow(arr[2],2))
        mag = np.linalg.norm(arr)
        arr1 = arr/mag
        return arr1
        
    def position(self):
        print("The position is ", self.pos[0], self.pos[1], self.pos[2])
        
    def euc_dist(self, x2, y2, z2):
        return np.sqrt(pow(y2-self.pos[1],2)+pow(x2-self.pos[0],2)+pow(z2-self.pos[2],2))
    
    
    def intarget(self, target):
        if pow(self.pos[0] - target.pos[0],2) + pow(self.pos[1] - target.pos[1],2) + pow(self.pos[2] - target.pos[2],2) < pow(target.rad,2):
            return True
        else:
            return False
        
    def next_pos(self, bot_list=[]):
        # print(self.alpha)
        self.timeval = self.timeval + self.delta_t
        self.present_run_time = self.present_run_time + self.delta_t
        # print('Run Duration:',self.run_duration)
        # print('present run time:',self.present_run_time)
        if self.present_run_time >= self.run_duration: #run cycle is over
            self.present_run_time = 0
            self.run_duration = np.random.exponential(scale=1/self.alpha)
            while(self.run_duration == 0): self.run_duration = np.random.exponential(scale=1/self.alpha)
            # print(self.orientation)
            temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
            temp = self.normalize(temp)
            self.orientation = temp
            
            # theta = np.random.uniform(0,359.9)*np.pi/180
            # cross1 = np.cross(temp, self.orientation)
            # cross2 = np.cross(temp, cross1)
            
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        pos_noise = np.sqrt(2*self.D_t*self.delta_t) * temp
        self.pos = self.pos + self.v*self.orientation*self.delta_t + pos_noise
        return self

# # =============================================================================
# # A bot or particle of the type Chiral ABP
# # =============================================================================        
class Chiral_ABP:
    def __init__(self,init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel):
        #initializations
        #initial positions of swarm
        self.pos = np.array([init_pos_x0, init_pos_y0, init_pos_z0])
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        self.orientation = self.normalize(temp)
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        self.w = self.normalize(temp) #angular velocity (rad/s)
        self.w_mag = w
        # print(np.linalg.norm(self.w))
        self.v = v #10; %speed (um per sec)
        # self.D_t = D_t#0.2; %(um^2/s) k_B*T/(6*pi*eta*R);
        # self.D_r = D_r#0.170; %(rad^2/s) k_B*T/(8*pi*eta*R^3);
        k_B = const.k#physconst('Boltzmann');
        T = 293; #Kelvin
        eta = 1.0016e-3; #N/m^2
        self.D_t = k_B*T/(6e-12*np.pi*eta*R); #(um^2/s)
        self.D_r = k_B*T/(8*np.pi*eta*pow(R,3)); #(rad^2/s)
        self.timeval = 0
        self.delta_t = delta_t
        
    def normalize(self, arr):
        # mag = np.sqrt(pow(arr[0],2) + pow(arr[1],2) + pow(arr[2],2))
        mag = np.linalg.norm(arr)
        arr1 = arr/mag
        return arr1
        
    def position(self):
        print("The position is ", self.pos[0], self.pos[1], self.pos[2])
        
    def euc_dist(self, x2, y2, z2):
        return np.sqrt(pow(y2-self.pos[1],2)+pow(x2-self.pos[0],2)+pow(z2-self.pos[2],2))
    
    def intarget(self, target):
        if pow(self.pos[0] - target.pos[0],2) + pow(self.pos[1] - target.pos[1],2) + pow(self.pos[2] - target.pos[2],2) < pow(target.rad,2):
            return True
        else:
            return False
        
    def next_pos(self, bot_list=[]):
        self.timeval = self.timeval + self.delta_t
        
        delthetaw = self.w_mag*self.delta_t*self.w
        thetaw = np.linalg.norm(delthetaw)#self.w_mag*self.delta_t
        crossw1 = np.cross(delthetaw,self.orientation)
        crossw2 = np.cross(delthetaw, crossw1) 
                
        deltheta = np.sqrt(2*self.D_r*self.delta_t)*np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        theta = np.linalg.norm(deltheta)
        cross1 = np.cross(deltheta,self.orientation)
        cross2 = np.cross(deltheta, cross1)
        
        w_add = np.sin(thetaw)*crossw1/thetaw + (1-np.cos(thetaw))*crossw2/pow(thetaw,2)
        noise_add = np.sin(theta)*cross1/theta + (1-np.cos(theta))*cross2/pow(theta,2)
        self.orientation = self.normalize(self.orientation + w_add + noise_add)
        
        temp = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        pos_noise = np.sqrt(2*self.D_t*self.delta_t) * temp
        self.pos = self.pos + pos_noise + self.v*self.orientation*self.delta_t
        return self


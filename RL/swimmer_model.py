# ==============================================================
# file: swimmer_model.py
# ==============================================================

import numpy as np
from joblib import Parallel, delayed

# ==============================================================
# Swimmer 
# ==============================================================
class Swimmer:
    #initializations
    def __init__(self, init_pos, init_orn, L, dia, delta_t):
        #initial positions of swarm
        self.pos_x = init_pos[0];
        self.pos_y = init_pos[1];
        self.L = L #length of helical swimmer
        self.dia = dia #diameter of helical swimmer
        self.timeval = 0
        self.delta_t = delta_t
        self.orientation = init_orn #degrees
        self.vicinity_rad = 10
        self.vicinity_particle_max = 10
        
    #determining the corners of the 2d swimmer
    def corners(self):
        r = self.dia/2
        sin_orn = np.sin(self.orientation*np.pi/180)
        cos_orn = np.cos(self.orientation*np.pi/180)
        pt1 = [self.pos_x + r*sin_orn, self.pos_y - r*cos_orn]
        pt2 = [self.pos_x + r*sin_orn + self.L*cos_orn, self.pos_y - r*cos_orn + self.L*sin_orn]
        pt3 = [self.pos_x - r*sin_orn + self.L*cos_orn, self.pos_y + r*cos_orn + self.L*sin_orn]
        pt4 = [self.pos_x - r*sin_orn, self.pos_y + r*cos_orn]
        return np.array([pt1,pt2,pt3,pt4])
       
    # calculating the euclidean distance between a given point and current swimmer position
    def euc_dist(self, x2, y2):
        return np.sqrt(pow(y2-self.pos_y,2)+pow(x2-self.pos_x,2))
       
    # calculating orientation error coefficient due to hydrodynamic effect
    def orn_dist_density_coeff(self, bot_list):
        vic_dist_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.euc_dist)(eachbot.pos_x,eachbot.pos_y) for eachbot in bot_list)
        vic_dist_inv_list = [2/pow(x,2) for x in vic_dist_list if x!=0]
        vic_cum_dist_inv = sum(vic_dist_inv_list)
        if vic_cum_dist_inv >= 1:
            vic_cum_dist_inv = 1
        return vic_cum_dist_inv
        
    # checking if swimmer is inside target
    def intarget(self, target):
        corner_pts = self.corners()
        for idx in range(0,corner_pts.shape[0]):
            if pow(corner_pts[idx,0] - target.pos[0],2) + pow(corner_pts[idx,1] - target.pos[1],2) < pow(target.rad,2):
                return True
        return False
 
# ==============================================================
# A magnetic swimmer
# ==============================================================
class MagSwimmer(Swimmer):
    # initializing parameters
    def __init__(self,init_pos = (0,0), init_orn=20, L=5, dia=0.3, delta_t=0.1):
        super().__init__(init_pos, init_orn, L, dia, delta_t)

    # calculating the next position of the swimmer 
    def next_pos(self, bot_list=[], f=10, orn=45):
        self.timeval = round(self.timeval + self.delta_t,1)
        if self.orientation > orn:
            self.orientation -= 5
        elif self.orientation < orn:
            self.orientation += 5
        vel = 0.5*f
        error_orientation = self.orientation-90
        dens_wt = super().orn_dist_density_coeff(bot_list)
        new_orientation = (dens_wt*error_orientation) + ((1-dens_wt)*self.orientation)
        x_addn = vel*(np.cos(new_orientation*np.pi/180))*(self.delta_t)
        y_addn = vel*(np.sin(new_orientation*np.pi/180))*(self.delta_t)
        self.pos_x = self.pos_x + x_addn 
        self.pos_y = self.pos_y + y_addn 
        return self
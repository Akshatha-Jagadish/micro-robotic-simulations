# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:59:44 2021

@author: Akshatha
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from joblib import Parallel, delayed
from models import ABP,Chiral_ABP,RTP,target,environment
import imageio
import os
from matplotlib.text import TextPath
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

ARROW_MARKER1 = TextPath((0, 0), "⟿")
ARROW_MARKER2 = TextPath((0, 0), "⇝")

# Function 1:  
# Slow function - use for num of particles < 100 for dens_dep_vel; useful for creating videos for presentation
# Usage example:
# ABP_list = []
# for idx in range(num_of_bots):     
#     ABP_list.append(ABP(init_pos[idx], v, R, delta_t, dens_dep_vel))
# absorbing_target_with_detailed_video(env1, ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel)
#    
def absorbing_target_with_detailed_video(env, bot_list_in, tfinal, delta_t, target, num_of_bots, dens_dep_vel):
    a = []
    pos_x_list = []
    pos_y_list = []
    bot_list = bot_list_in
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        num_in_target = 0
        new_bot_list = [bot for bot in bot_list if not bot.intarget(target)]
        num_in_target = len(bot_list) - len(new_bot_list)
        bot_list = new_bot_list
        prev_bot_list = bot_list
        new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(env, prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        plt.scatter(pos_x_list,pos_y_list,'.','blue') 
        circle = plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.grid(True)
        plt.xlim((-100,100))
        plt.ylim((-100,100))
        plt.draw()
        env.disp_env()
        plt.pause(0.0000000005)
        pos_x_list.clear()
        pos_y_list.clear()
        plt.clf()
        for bot in bot_list:
            pos_x_list.append(bot.pos_x)
            pos_y_list.append(bot.pos_y)
        plt.scatter(pos_x_list,pos_y_list,'.','blue') 
        circle = plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.grid(True)
        plt.xlim((-100,100))
        plt.ylim((-100,100))
        plt.draw()
        env.disp_env()
        plt.pause(0.0000000005)
        pos_x_list.clear()
        pos_y_list.clear()
        plt.clf()
        a.append(num_in_target/num_of_bots)
        print(timeidx)
    cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
    return cap_eff


def save_video(env, bot_list_in_time, target, path_loc, fileloc):
    filenames = []
    i = 1
    # path = os.path.join('C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch','outputs\\varying_w_Chiral_ABP')
    # plt.figure(figsize=(14,12))
    plt.rcParams['figure.figsize'] = [14, 12]
    for bot_array in bot_list_in_time:
        pos_x_list = [bot.pos_x for bot in bot_array]
        pos_y_list = [bot.pos_y for bot in bot_array]
        plt.scatter(pos_x_list,pos_y_list,marker='.',color='blue') #,markersize = 10)
        circle = plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.grid(True)
        plt.xlim((-40,40))
        plt.ylim((-40,40))
        #plt.pause(0.005)
        plt.draw()
        env.disp_env()
        plt.pause(0.005)
        filename = f'{i}.png'
        filenames.append(filename)
        # save frame[]
        plt.savefig(os.path.join(path_loc,'imgs',filename))
        plt.close()
        i = i+1
        plt.clf()
    with imageio.get_writer(os.path.join(path_loc,fileloc), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_loc,'imgs',filename))
            writer.append_data(image)

def save_video_mixed(env, bot_list_in_time, target, path_loc, fileloc):
    filenames = []
    i = 1
    # path = os.path.join('C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch','outputs\\varying_w_Chiral_ABP')
    # plt.figure(figsize=(14,12))
    plt.rcParams['figure.figsize'] = [14, 12]
    for bot_array in bot_list_in_time:
        pos_x_list_ABP = [bot.pos_x for bot in bot_array if isinstance(bot, ABP)]
        pos_y_list_ABP = [bot.pos_y for bot in bot_array if isinstance(bot, ABP)]
        pos_x_list_RTP = [bot.pos_x for bot in bot_array if isinstance(bot, RTP)]
        pos_y_list_RTP = [bot.pos_y for bot in bot_array if isinstance(bot, RTP)]
        pos_x_list_chiral = [bot.pos_x for bot in bot_array if isinstance(bot, Chiral_ABP)]
        pos_y_list_chiral = [bot.pos_y for bot in bot_array if isinstance(bot, Chiral_ABP)]
        plt.scatter(pos_x_list_ABP,pos_y_list_ABP,marker='.',color='blue') #,markersize = 10)
        plt.scatter(pos_x_list_RTP,pos_y_list_RTP,marker='.',color='green') #,markersize = 10)
        plt.scatter(pos_x_list_chiral,pos_y_list_chiral,marker='.',color='red') #,markersize = 10)
        circle = plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.grid(True)
        plt.xlim((-40,40))
        plt.ylim((-40,40))
        #plt.pause(0.005)
        plt.draw()
        env.disp_env()
        plt.pause(0.005)
        filename = f'{i}.png'
        filenames.append(filename)
        # save frame[]
        plt.savefig(os.path.join(path_loc,'imgs',filename))
        plt.close()
        i = i+1
        plt.clf()
    with imageio.get_writer(os.path.join(path_loc,fileloc), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_loc,'imgs',filename))
            writer.append_data(image)
            
def plot_video(env, bot_list_in_time, target):
    # plt.figure(figsize=(14,12))
    plt.rcParams['figure.figsize'] = [14, 12]
    for bot_array in bot_list_in_time:
        pos_x_list = [bot.pos_x for bot in bot_array]
        pos_y_list = [bot.pos_y for bot in bot_array]
        plt.scatter(pos_x_list,pos_y_list,'.','blue') #,markersize = 10)
        circle = plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.grid(True)
        plt.xlim((-100,100))
        plt.ylim((-100,100))
        #plt.pause(0.005)
        plt.draw()
        env.disp_env()
        plt.pause(0.005)
        plt.clf()
        
def calc_MSD(bot_list_in_time):
    MSD_in_time = []
    for bot_array in bot_list_in_time:
        sd = [bot.pos_x**2 + bot.pos_y**2 for bot in bot_array]
        MSD_in_time.append(sum(sd)/len(sd))
    return MSD_in_time

def calc_all_MSD(env, num_of_bots, init_pos, v, R, w, alpha, delta_t, tfinal, dens_dep_vel):
    ABP_list = []
    ChiralABP_list = []
    RTP_list = []
    # start = time.time()
    for idx in range(num_of_bots):     
        ABP_list.append(ABP(init_pos[idx], v, R, delta_t, dens_dep_vel))
        ChiralABP_list.append(Chiral_ABP(init_pos[idx], v, R, w, delta_t, dens_dep_vel))
        RTP_list.append(RTP(init_pos[idx], v, R, alpha, delta_t, dens_dep_vel))
    MSD_ABP = MSD_bot_list(env, ABP_list, tfinal, delta_t)
    MSD_chiral = MSD_bot_list(env, ChiralABP_list, tfinal, delta_t)
    MSD_RTP = MSD_bot_list(env, RTP_list, tfinal, delta_t)
    return MSD_ABP,MSD_chiral,MSD_RTP

def MSD_chiral_theoretical(D_t, D_r, v, w, time_bar, k=1):
    MSD = []
    for t in time_bar:
        cos_phi0 = (pow(D_r,2) - pow(w,2))/(pow(D_r,2) + pow(w,2))
        phi0 = np.arccos(cos_phi0)
        term_1 = 4*D_t*t
        term_2 = k*pow(v,2)*D_r*t/(pow(D_r,2) + pow(w,2))
        term_3 = k*pow(v,2)*(np.exp(-1*D_r*t)*np.cos((w*t)+phi0)-cos_phi0)/(pow(D_r,2) + pow(w,2))
        MSD.append(term_1 + term_2 + term_3)
    return MSD

def MSD_RTP_theoretical(alpha, D_t, v, time_bar, k=1):
    MSD = []
    for t in time_bar:
        term_1 = 4*D_t*t
        term_2 = k*pow(v,2)*t/alpha
        term_3 = k*pow(v,2)*(np.exp(-alpha*t)-1)/pow(alpha,2)
        MSD.append(term_1 + term_2 + term_3)
    return MSD

def MSD_ABP_theoretical(D_r, D_t, v, time_bar, k=1):
    MSD = []
    for t in time_bar:
        term_1 = 4*D_t*t
        term_2 = k*pow(v,2)*t/D_r
        term_3 = k*pow(v,2)*(np.exp(-1*D_r*t)-1)/pow(D_r,2)
        MSD.append(term_1 + term_2 + term_3)
    return MSD

def plot_MSD(ax, ABP_MSD, ChiralABP_MSD, RTP_MSD, tfinal, delta_t):
    time_bar = [*np.arange(0,tfinal,delta_t)]
    ax.plot(time_bar, ABP_MSD,color = '#062A06',label = 'ABP')
    ax.plot(time_bar,ChiralABP_MSD,color = '#FF0000',label = 'Chiral_ABP')
    ax.plot(time_bar,RTP_MSD,color = '#0000FF',label = 'RTP')

# Group function 2:
# Usage example:
# ABP_list = []
# for idx in range(num_of_bots):     
#     ABP_list.append(ABP(init_pos[idx], v, R, delta_t, dens_dep_vel))
# absorbing_target_video(env1, ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
def absorbing_target_video(env, bot_list, tfinal, delta_t, target, num_of_bots, dens_dep_vel, video_flag, MSD_flag):
    a = []
    new_bot_list = []
    if video_flag or MSD_flag:
        bot_list_in_time =[]
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        if video_flag or MSD_flag:
            bot_list_in_time.append(copy.deepcopy(bot_list))
        num_in_target = 0
        # print(len(bot_list))
        new_bot_list = [bot for bot in bot_list if not bot.intarget(target)]
        num_in_target = len(bot_list) - len(new_bot_list)
        bot_list = new_bot_list
        prev_bot_list = bot_list
        new_bot_list = Parallel(n_jobs=4, prefer="threads")(delayed(bot.next_pos)(env, prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        a.append(num_in_target/num_of_bots)
        # print("time",timeidx)
        #print(len(bot_list_in_time))
    cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
    if video_flag:
        plot_video(env, bot_list_in_time, target)
    if MSD_flag:
        return cap_eff, bot_list_in_time
    else:
        return cap_eff, []

def MSD_bot_list(env, bot_list, tfinal, delta_t, dens_dep_vel=0):
    a = []
    new_bot_list = []
    bot_list_in_time =[]
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        bot_list_in_time.append(copy.deepcopy(bot_list))
        num_in_target = 0
        # print(len(bot_list))
        # new_bot_list = [bot for bot in bot_list if not bot.intarget(target)]
        # num_in_target = len(bot_list) - len(new_bot_list)
        # bot_list = new_bot_list
        prev_bot_list = bot_list
        new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(env, prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        # a.append(num_in_target/num_of_bots)
        # print("time",timeidx)
        #print(len(bot_list_in_time))
    MSD = calc_MSD(bot_list_in_time)
    return MSD
    


#function to save the video in a file
# Usage example:
# ChiralABP_list = []
# for idx in range(num_of_bots):     
#     ChiralABP_list.append(Chiral_ABP(init_pos[idx], v, R, w, delta_t, dens_dep_vel))
# cap_eff_Chiral_ABP,bot_list_chiral = absorbing_target_save_video(env1, ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, path_loc)    
def absorbing_target_save_video(env, bot_list, tfinal, delta_t, target, num_of_bots, dens_dep_vel, path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\', save_file_name = 'test.gif', mixed_flag = 0):
    if mixed_flag:
        a_ABP = []
        a_RTP = []
        a_chiral = []
    else:
        a = []
    new_bot_list = []
    bot_list_in_time =[]
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        bot_list_in_time.append(copy.deepcopy(bot_list))
        num_in_target = 0
        #print(len(bot_list))
        new_bot_list = [bot for bot in bot_list if not bot.intarget(target)]
        if mixed_flag:
            temp = [1 for bot in bot_list if isinstance(bot, ABP) and bot.intarget(target)]
            num_in_target_ABP = len(temp)
            temp = [1 for bot in bot_list if isinstance(bot, RTP) and bot.intarget(target)]
            num_in_target_RTP = len(temp)
            temp = [1 for bot in bot_list if isinstance(bot, Chiral_ABP) and bot.intarget(target)]
            num_in_target_chiral = len(temp)
        else:
            num_in_target = len(bot_list) - len(new_bot_list)
        bot_list = new_bot_list
        # print(len(bot_list))
        prev_bot_list = bot_list
        new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(env, prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        if mixed_flag:
            a_ABP.append(num_in_target_ABP/num_of_bots)
            a_RTP.append(num_in_target_RTP/num_of_bots)
            a_chiral.append(num_in_target_chiral/num_of_bots)
        else:
            a.append(num_in_target/num_of_bots)
        print("time",timeidx)
        #print(len(bot_list_in_time))
    if mixed_flag:
        cap_eff_ABP = [sum(a_ABP[0:x+1]) for x in range(0,len(a_ABP))]
        cap_eff_RTP = [sum(a_RTP[0:x+1]) for x in range(0,len(a_RTP))]
        cap_eff_chiral = [sum(a_chiral[0:x+1]) for x in range(0,len(a_chiral))]
        save_video_mixed(env, bot_list_in_time, target, path_loc, save_file_name)
        return cap_eff_ABP, cap_eff_RTP, cap_eff_chiral, []
    else:
        cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
        save_video(env, bot_list_in_time, target, path_loc, save_file_name)
        return cap_eff, []
#Function 2 end    
 

# Function3:
#a sample video to look at how ABPs move around
# Usage example:
# ABP_sample_video(env1, init_pos, v, R, delta_t, tfinal, num_of_bots, dens_dep_vel)
def ABP_sample_video(env, init_pos, v, R, delta_t, tfinal, num_of_bots, dens_dep_vel):
    # plt.figure(figsize=(8,8))
    ABP_list = []
    for idx in range(num_of_bots):     
        ABP_list.append(ABP(init_pos[idx], v, R, delta_t, dens_dep_vel))
    # plt.ion()
    pos_x_list = []
    pos_y_list = []
    i = 1
    filenames = []
    path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\ABP_video'
    for idx in np.arange(delta_t,tfinal,delta_t):
        for ABPart in ABP_list:
            ABPart.next_pos(env)
            pos_x_list.append(ABPart.pos_x)
            pos_y_list.append(ABPart.pos_y)
        plt.scatter(pos_x_list,pos_y_list,'.','blue') 
        plt.grid(True)
        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.draw()
        env.disp_env()
        plt.pause(0.05)
        pos_x_list.clear()
        pos_y_list.clear()
        filename = f'{i}.png'
        filenames.append(filename)
        # save frame[]
        plt.savefig(os.path.join(path_loc,'imgs',filename))
        plt.close()
        i = i+1
        plt.clf()
    with imageio.get_writer(os.path.join(path_loc,'random.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_loc,'imgs',filename))
            writer.append_data(image)

# Function4:
#a sample video to look at how RTPs move around
# Usage example:
# RTP_sample_video(env1, init_pos, v, R, alpha, delta_t, tfinal, num_of_bots, dens_dep_vel)
def RTP_sample_video(env, init_pos, v, R, alpha, delta_t, tfinal, num_of_bots, dens_dep_vel):
    RTP_list = []
    # plt.figure(figsize=(8,8))
    path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\RTP_video'
    for idx in range(num_of_bots):     
        RTP_list.append(RTP(init_pos[idx], v, R, alpha, delta_t, dens_dep_vel))
    pos_x_list = []
    pos_y_list = []
    i = 1
    filenames = []
    for idx in np.arange(delta_t,tfinal,delta_t):
        for RTPart in RTP_list:
            RTPart.next_pos(env)
            pos_x_list.append(RTPart.pos_x)
            pos_y_list.append(RTPart.pos_y)
        plt.scatter(pos_x_list,pos_y_list,marker='.',color='blue') 
        plt.grid(True)
        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.draw()
        env.disp_env()
        plt.pause(0.05)
        pos_x_list.clear()
        pos_y_list.clear()
        filename = f'{i}.png'
        filenames.append(filename)
        # save frame[]
        plt.savefig(os.path.join(path_loc,'imgs',filename))
        plt.close()
        i = i+1
        plt.clf()
    with imageio.get_writer(os.path.join(path_loc,'random2.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_loc,'imgs',filename))
            writer.append_data(image)

# Function5:
#a sample video to look at how Chiral ABPs move around 
# Usage example:   
# Chiral_ABP_sample_video(env1, init_pos, v, R, w, delta_t, tfinal, num_of_bots, dens_dep_vel)             
def Chiral_ABP_sample_video(env, init_pos, v, R, w, delta_t, tfinal, num_of_bots, dens_dep_vel):
    Chiral_ABP_list = []
    # plt.figure(figsize=(8,8))
    for idx in range(num_of_bots):     
        Chiral_ABP_list.append(Chiral_ABP(init_pos[idx], v, R, w, delta_t, dens_dep_vel))
    pos_x_list = []
    pos_y_list = []
    i = 1
    filenames = []
    path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\Chiral_ABP_video'
    for idx in np.arange(delta_t,tfinal,delta_t):
        for Chiral_ABPart in Chiral_ABP_list:
            Chiral_ABPart.next_pos(env)
            pos_x_list.append(Chiral_ABPart.pos_x)
            pos_y_list.append(Chiral_ABPart.pos_y)
        plt.scatter(pos_x_list,pos_y_list,'.','blue' )
        plt.grid(True)
        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.draw()
        env.disp_env()
        plt.pause(0.05)
        pos_x_list.clear()
        pos_y_list.clear()
        # plt.clf()
        filename = f'{i}.png'
        filenames.append(filename)
        # save frame[]
        plt.savefig(os.path.join(path_loc,'imgs',filename))
        plt.close()
        i = i+1
        plt.clf()
    with imageio.get_writer(os.path.join(path_loc,'random.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_loc,'imgs',filename))
            writer.append_data(image)
    
def abs_target_cap_eff(target1, env1, ax, init_pos, v, R, w, alpha, delta_t, dens_dep_vel, tfinal, num_of_bots, video_flag, MSD_flag):
    ABP_list = []
    ChiralABP_list = []
    RTP_list = []
    # start = time.time()
    for idx in range(num_of_bots):     
        ABP_list.append(ABP(init_pos[idx], v, R, delta_t, dens_dep_vel))
        ChiralABP_list.append(Chiral_ABP(init_pos[idx], v, R, w, delta_t, dens_dep_vel))
        RTP_list.append(RTP(init_pos[idx], v, R, alpha, delta_t, dens_dep_vel))
    cap_eff_ABP,bot_list_ABP = absorbing_target_video(env1, ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    cap_eff_Chiral_ABP,bot_list_chiral = absorbing_target_video(env1, ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    cap_eff_RTP,bot_list_RTP = absorbing_target_video(env1, RTP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    time_bar = [*np.arange(0,tfinal,delta_t)]
    # plt.clf()
    ax.plot(time_bar,cap_eff_ABP,color = '#A1F1A1')
    ax.plot(time_bar,cap_eff_Chiral_ABP,color = '#FFCCCC')
    ax.plot(time_bar,cap_eff_RTP,color = '#ADD8E6')
    ax.grid(True)
    # ax.legend()
    if MSD_flag:
        ABP_list = []
        ChiralABP_list = []
        RTP_list = []
        # start = time.time()
        for idx in range(num_of_bots):     
            ABP_list.append(ABP(init_pos[idx], v, R, delta_t, dens_dep_vel))
            ChiralABP_list.append(Chiral_ABP(init_pos[idx], v, R, w, delta_t, dens_dep_vel))
            RTP_list.append(RTP(init_pos[idx], v, R, alpha, delta_t, dens_dep_vel))
        MSD_ABP = MSD_bot_list(env1, ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel)
        MSD_chiral = MSD_bot_list(env1, ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel)
        MSD_RTP = MSD_bot_list(env1, RTP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel)
        return cap_eff_ABP,cap_eff_Chiral_ABP,cap_eff_RTP,MSD_ABP,MSD_chiral,MSD_RTP
    else:
        return cap_eff_ABP,cap_eff_Chiral_ABP,cap_eff_RTP,[],[],[]
    # end = time.time()
    # print(end-start)

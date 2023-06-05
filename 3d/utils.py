# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:30:36 2023

@author: Akshatha
"""
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import copy
from models_3D import target, ABP, RTP, Chiral_ABP
import os
import imageio

# =============================================================================
# Group-fns begin:
# =============================================================================        
def absorbing_target(bot_list, tfinal, delta_t, target, num_of_bots, dens_dep_vel):
    a = []
    new_bot_list = []
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        num_in_target = 0
        #print(len(bot_list))
        for bot in bot_list:
            if bot.intarget(target):
                bot_list.remove(bot)
                num_in_target = num_in_target+1 
        prev_bot_list = bot_list
        if dens_dep_vel == 0:
            #for bot in bot_list: bot.next_pos()
            new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)() for bot in bot_list)
            # print(bot.pos)
        elif dens_dep_vel == 1:
            #for bot in bot_list: bot.next_pos(prev_bot_list) 
            new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        a.append(num_in_target/num_of_bots)
        print(timeidx)
        #print(new_bot_list[:10])
    cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
    return cap_eff

# def absorbing_target_with_video(env, bot_list_in, tfinal, delta_t, target, num_of_bots, dens_dep_vel):
#     a = []
#     pos_x_list = []
#     pos_y_list = []
#     bot_list = bot_list_in
#     if dens_dep_vel == 1: prev_bot_list = [] 
#     for timeidx in np.arange(0,tfinal,delta_t):
#         num_in_target = 0
#         for bot in bot_list:
#             pos_x_list.append(bot.pos_x)
#             pos_y_list.append(bot.pos_y)
#             if bot.intarget(target):
#                 bot_list.remove(bot)
#                 num_in_target = num_in_target+1
#         prev_bot_list = bot_list
#         if dens_dep_vel == 0:
#             #for bot in bot_list: bot.next_pos()
#             new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)() for bot in bot_list)
#         elif dens_dep_vel == 1:
#             #for bot in bot_list: bot.next_pos(prev_bot_list) 
#             new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(prev_bot_list) for bot in bot_list)
#         bot_list = new_bot_list
#         plt.plot(pos_x_list,pos_y_list,'b.') 
#         circle = plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
#         plt.gca().add_patch(circle)
#         plt.grid(True)
#         plt.xlim((-100,100))
#         plt.ylim((-100,100))
#         plt.draw()
#         env.disp_env()
#         plt.pause(0.0000000005)
#         pos_x_list.clear()
#         pos_y_list.clear()
#         plt.clf()
#         for bot in bot_list:
#             pos_x_list.append(bot.pos_x)
#             pos_y_list.append(bot.pos_y)
#         plt.plot(pos_x_list,pos_y_list,'b.') 
#         circle=plt.Circle(( target.pos[0] , target.pos[1] ),  target.rad,  color='#00ffff', alpha=0.5)
#         plt.gca().add_patch(circle)
#         plt.grid(True)
#         plt.xlim((-100,100))
#         plt.ylim((-100,100))
#         plt.draw()
#         env.disp_env()
#         plt.pause(0.0000000005)
#         pos_x_list.clear()
#         pos_y_list.clear()
#         plt.clf()
#         a.append(num_in_target/num_of_bots)
#         print(timeidx)
#     cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
#     return cap_eff   

# https://stackoverflow.com/questions/40460960/how-to-plot-a-sphere-when-we-are-given-a-central-point-and-a-radius-size
def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

def plot_target(ax, target):
    frame_xs, frame_ys, frame_zs = WireframeSphere(target.pos, target.rad, 40)
    ax.plot_surface(frame_xs, frame_ys, frame_zs, color="yellow", alpha=0.3) 
    
def plot_video(bot_list_in_time, target):
    for bot_array in bot_list_in_time:
        pos_x_list = [bot.pos[0] for bot in bot_array]
        pos_y_list = [bot.pos[1] for bot in bot_array]
        pos_z_list = [bot.pos[2] for bot in bot_array]
        ax = plt.axes(projection='3d')
        plot_target(ax, target)
        ax.scatter3D(pos_x_list, pos_y_list, pos_z_list, color='blue', s = 10)
        plt.grid(True)
        ax.set_xlim((-10,15))
        ax.set_ylim((-10,10))
        ax.set_zlim((-10,10))
        # ax.view_init(elev=40, azim=-100)
        plt.draw()
        # env.disp_env()
        plt.pause(0.005)
        plt.clf()

def save_video(bot_list_in_time, target, path_loc, fileloc):
    filenames = []
    i = 1
    # path = os.path.join('C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch','outputs\\varying_w_Chiral_ABP')
    # plt.figure(figsize=(14,12))
    plt.rcParams['figure.figsize'] = [14, 12]
    for bot_array in bot_list_in_time:
        pos_x_list = [bot.pos[0] for bot in bot_array]
        pos_y_list = [bot.pos[1] for bot in bot_array]
        pos_z_list = [bot.pos[2] for bot in bot_array]
        ax = plt.axes(projection='3d')
        plot_target(ax, target)
        ax.scatter3D(pos_x_list, pos_y_list, pos_z_list, color='blue', s = 10)
        plt.grid(True)
        ax.set_xlim((-10,15))
        ax.set_ylim((-10,10))
        ax.set_zlim((-10,10))
        #plt.pause(0.005)
        plt.draw()
        # env.disp_env()
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
            
def absorbing_target_save_video(bot_list, tfinal, delta_t, target, num_of_bots, dens_dep_vel, path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs', save_file_name = 'test.gif'):
    a = []
    new_bot_list = []
    bot_list_in_time =[]
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        bot_list_in_time.append(copy.deepcopy(bot_list))
        num_in_target = 0
        #print(len(bot_list))
        new_bot_list = [bot for bot in bot_list if not bot.intarget(target)]
        num_in_target = len(bot_list) - len(new_bot_list)
        bot_list = new_bot_list
        # print(len(bot_list))
        prev_bot_list = bot_list
        new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)() for bot in bot_list)
        bot_list = new_bot_list
        a.append(num_in_target/num_of_bots)
        print("time",timeidx)
        #print(len(bot_list_in_time))
    cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
    save_video(bot_list_in_time, target, path_loc, save_file_name)
    return cap_eff, []

def absorbing_target_video(bot_list, tfinal, delta_t, target, num_of_bots, dens_dep_vel, video_flag, MSD_flag):
    a = []
    new_bot_list = []
    if video_flag or MSD_flag:
        bot_list_in_time =[]
    if dens_dep_vel == 1: prev_bot_list = [] 
    for timeidx in np.arange(0,tfinal,delta_t):
        if video_flag or MSD_flag:
            bot_list_in_time.append(copy.deepcopy(bot_list))
        num_in_target = 0
        #print(len(bot_list))
        for bot in bot_list:
            if bot.intarget(target):
                bot_list.remove(bot)
                num_in_target = num_in_target+1 
        prev_bot_list = bot_list
        if dens_dep_vel == 0:
            #for bot in bot_list: bot.next_pos()
            new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)() for bot in bot_list)
        elif dens_dep_vel == 1:
            #for bot in bot_list: bot.next_pos(prev_bot_list) 
            new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        a.append(num_in_target/num_of_bots)
        # print(timeidx)
        #print(len(bot_list_in_time))
    cap_eff = [sum(a[0:x+1]) for x in range(0,len(a))]
    if video_flag:
        plot_video(bot_list_in_time, target)
    if MSD_flag:
        return cap_eff, bot_list_in_time
    else:
        return cap_eff, []

#a sample video to look at how ABPs move around
def ABP_video(init_pos_x0, init_pos_y0, v, D_t, D_r, delta_t, tfinal, num_of_bots, dens_dep_vel):
    ABP_list = []
    for idx in range(num_of_bots):     
        ABP_list.append(ABP(init_pos_x0, init_pos_y0, v, D_t, D_r, delta_t, dens_dep_vel))
    plt.ion()
    pos_x_list = []
    pos_y_list = []
    for idx in np.arange(delta_t,tfinal,delta_t):
        for ABPart in ABP_list:
            ABPart.next_pos()
            pos_x_list.append(ABPart.pos_x)
            pos_y_list.append(ABPart.pos_y)
        plt.plot(pos_x_list,pos_y_list,'b.') 
        plt.grid(True)
        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.draw()
        plt.pause(0.2)
        pos_x_list.clear()
        pos_y_list.clear()
        plt.clf()

#a sample video to look at how RTPs move around
def RTP_video(init_pos_x0, init_pos_y0, v, alpha, delta_t, tfinal, num_of_bots, dens_dep_vel):
    RTP_list = []
    for idx in range(num_of_bots):     
        RTP_list.append(RTP(init_pos_x0, init_pos_y0, v, alpha, delta_t, dens_dep_vel))
    plt.ion()
    pos_x_list = []
    pos_y_list = []
    for idx in np.arange(delta_t,tfinal,delta_t):
        for RTPart in RTP_list:
            RTPart.next_pos()
            pos_x_list.append(RTPart.pos_x)
            pos_y_list.append(RTPart.pos_y)
        plt.plot(pos_x_list,pos_y_list,'b.') 
        plt.grid(True)
        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.draw()
        plt.pause(0.2)
        pos_x_list.clear()
        pos_y_list.clear()
        plt.clf()

#a sample video to look at how Chiral ABPs move around                 
def Chiral_ABP_video(init_pos_x0, init_pos_y0, v, D_t, D_r, w, delta_t, tfinal, num_of_bots, dens_dep_vel):
    Chiral_ABP_list = []
    for idx in range(num_of_bots):     
        Chiral_ABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, v, D_t, D_r, w, delta_t, dens_dep_vel))
    plt.ion()
    pos_x_list = []
    pos_y_list = []
    for idx in np.arange(delta_t,tfinal,delta_t):
        for Chiral_ABPart in Chiral_ABP_list:
            Chiral_ABPart.next_pos()
            pos_x_list.append(Chiral_ABPart.pos_x)
            pos_y_list.append(Chiral_ABPart.pos_y)
        plt.plot(pos_x_list,pos_y_list,'b.') 
        plt.grid(True)
        plt.xlim((-50,50))
        plt.ylim((-50,50))
        plt.draw()
        plt.pause(0.2)
        pos_x_list.clear()
        pos_y_list.clear()
        plt.clf()
   
def MSD_chiral_theoretical(D_t, D_r, v, w, time_bar, k=1):
    MSD = []
    for t in time_bar:
        cos_phi0 = (pow(D_r,2) - pow(w,2))/(pow(D_r,2) + pow(w,2))
        phi0 = np.arccos(cos_phi0)
        term_1 = 4*D_t*t
        term_2 = k*pow(v,2)*D_r*t/(pow(D_r,2) + pow(w,2))
        term_3 = k*pow(v,2)*(np.exp(-D_r*t)*np.cos((w*t)+phi0)-cos_phi0)/(pow(D_r,2) + pow(w,2))
        MSD.append(1.5*term_1 + term_2 + term_3)
    return MSD

def MSD_RTP_theoretical(alpha, D_t, v, time_bar, k=1):
    MSD = []
    for t in time_bar:
        term_1 = 4*D_t*t
        term_2 = k*pow(v,2)*t/alpha
        term_3 = k*pow(v,2)*(np.exp(-1*alpha*t)-1)/pow(alpha,2)
        MSD.append(1.5*term_1 + term_2 + term_3)
    return MSD

def MSD_ABP_theoretical(D_r, D_t, v, time_bar, k=1):
    MSD = []
    for t in time_bar:
        term_1 = 4*D_t*t
        term_2 = k*pow(v,2)*t/D_r
        term_3 = k*pow(v,2)*(np.exp(-D_r*t)-1)/pow(D_r,2)
        MSD.append(1.5*term_1 + term_2 + term_3)
    return MSD

def calc_MSD(bot_list_in_time):
    MSD_in_time = []
    for bot_array in bot_list_in_time:
        sd = [(bot.pos[0]**2)+(bot.pos[1]**2)+(bot.pos[2]**2) for bot in bot_array]
        MSD_in_time.append(sum(sd)/len(sd))
    return MSD_in_time

def plot_MSD(ax, ABP_MSD,ChiralABP_MSD,RTP_MSD, tfinal, delta_t):
    time_bar = [*np.arange(0,tfinal,delta_t)]
    ax.plot(time_bar, ABP_MSD,color = '#062A06',label = 'ABP')
    ax.plot(time_bar,ChiralABP_MSD,color = '#FF0000',label = 'Chiral_ABP')
    ax.plot(time_bar,RTP_MSD,color = '#0000FF',label = 'RTP')

def calc_all_MSD(num_of_bots, init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, alpha, delta_t, tfinal, dens_dep_vel):
    ABP_list = []
    ChiralABP_list = []
    RTP_list = []
    # start = time.time()
    for idx in range(num_of_bots):     
        ABP_list.append(ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel))
        ChiralABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel))
        RTP_list.append(RTP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
    MSD_ABP = MSD_bot_list(ABP_list, tfinal, delta_t)
    MSD_chiral = MSD_bot_list(ChiralABP_list, tfinal, delta_t)
    MSD_RTP = MSD_bot_list(RTP_list, tfinal, delta_t)
    return MSD_ABP,MSD_chiral,MSD_RTP

def MSD_bot_list(bot_list, tfinal, delta_t):
    new_bot_list = []
    bot_list_in_time =[] 
    for timeidx in np.arange(0,tfinal,delta_t):
        bot_list_in_time.append(copy.deepcopy(bot_list))
        prev_bot_list = bot_list
        # Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)() for bot in bot_list)
        new_bot_list = Parallel(n_jobs=2, prefer="threads")(delayed(bot.next_pos)(prev_bot_list) for bot in bot_list)
        bot_list = new_bot_list
        # print("time",timeidx)
        #print(len(bot_list_in_time))
    MSD = calc_MSD(bot_list_in_time)
    return MSD
    
def abs_target_cap_eff(target1, ax, init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, alpha, delta_t, dens_dep_vel, tfinal, num_of_bots, video_flag, MSD_flag):
    ABP_list = []
    ChiralABP_list = []
    RTP_list = []
    # start = time.time()
    for idx in range(num_of_bots): 
        ABP_list.append(ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel))
        ChiralABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel))
        RTP_list.append(RTP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
    cap_eff_ABP,bot_list_ABP = absorbing_target_video(ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    cap_eff_Chiral_ABP,bot_list_chiral = absorbing_target_video(ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    cap_eff_RTP,bot_list_RTP = absorbing_target_video(RTP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    time_bar = [*np.arange(0,tfinal,delta_t)]
    # plt.clf()
    ax.plot(time_bar,cap_eff_ABP,color = '#A1F1A1')
    ax.plot(time_bar,cap_eff_Chiral_ABP,color = '#FFCCCC')
    ax.plot(time_bar,cap_eff_RTP,color = '#ADD8E6')
    # ax.grid(True)
    # ax.legend()
    if MSD_flag:
        MSD_ABP,MSD_chiral,MSD_RTP = calc_all_MSD(num_of_bots, init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, alpha, delta_t, tfinal, dens_dep_vel)
        return cap_eff_ABP,cap_eff_Chiral_ABP,cap_eff_RTP,MSD_ABP,MSD_chiral,MSD_RTP
    else:
        return cap_eff_ABP,cap_eff_Chiral_ABP,cap_eff_RTP,[],[],[]
# =============================================================================
# Group-fns end   
# =============================================================================
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:39:45 2021

@author: Akshatha
"""

#import random
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
# import shapely.geometry as sp
import time
import copy
from models_3D import target, ABP, RTP, Chiral_ABP
import pickle
from utils import *
#%matplotlib qt

# =============================================================================
# Main-fn begins
# =============================================================================
if __name__ == "__main__":
    #initializations begin
    R = 0.5e-6;#bot size (radius) in m
    R_scale = R/1e-6
    v = 20*R_scale;#speed (um per sec)
    tfinal = 5;
    num_of_bots = 100;
    delta_t = 0.05;
    w = 3*R_scale;#%rad/s angular velocity for Chiral ABP
    init_pos_x0 = 0;
    init_pos_y0 = 0;
    init_pos_z0 = 0;
    time_bar = [*np.arange(0,tfinal,delta_t)]
    
    #tags for complexity in environment
    #density dependent velocity
    dens_dep_vel = 0; #on or off
    #watch the video 
    video_flag = 0; 
    #plot MSD
    MSD_flag = 0;
    #plot error_bar
    error_bar_flag = 0;
    
    #Derived constants
    k_B = const.k#physconst('Boltzmann');
    T = 293; #Kelvin
    eta = 1.0016e-3; #N/m^2
    D_t = k_B*T/(6e-12*np.pi*eta*R); #(um^2/s)
    D_r = k_B*T/(8*np.pi*eta*pow(R,3)); #(rad^2/s)
    alpha = 2*D_r;#%event rate for RTP 
    P_l = v/(D_r) #persistence length in um
    P_e = v/np.sqrt(D_t*D_r) #peclet number
    
    #target specifications
    target_x = 15*R_scale;
    target_y = 0;
    target_z = 0;
    target_rad = 10*R_scale;
    target1 = target(target_x, target_y, target_z, target_rad)
        
    
    print('Chiral ABP')
    ChiralABP_list = []
    for idx in range(num_of_bots):     
        ChiralABP_list.append(Chiral_ABP(init_pos_x0,init_pos_y0,init_pos_z0, v, R, w, delta_t, dens_dep_vel))
    cap_eff_Chiral_ABP,bot_list_chiral = absorbing_target_save_video(ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel,
                                                                      path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\normal',
                                                                      save_file_name = 'Chiral_ABP_2.gif')
    print(cap_eff_Chiral_ABP)
    
    print("ABP")
    ABP_list = []
    for idx in range(num_of_bots):     
        ABP_list.append(ABP(init_pos_x0,init_pos_y0,init_pos_z0, v, R, delta_t, dens_dep_vel))
    cap_eff_ABP,bot_list_ABP = absorbing_target_save_video(ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel,
                                                                     path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\normal',
                                                                     save_file_name = 'ABP_2.gif')
    print(cap_eff_ABP)
    
    print("RTP")
    RTP_list = []
    for idx in range(num_of_bots):     
        RTP_list.append(RTP(init_pos_x0,init_pos_y0,init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
    cap_eff_RTP,bot_list_RTP = absorbing_target_save_video(RTP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel,
                                                                      path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\normal',
                                                                      save_file_name = 'RTP_2.gif')
    print(cap_eff_RTP)
    
    # #MSD plot
    # MSD_ABP,MSD_chiral,MSD_RTP = calc_all_MSD(num_of_bots, init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, alpha, delta_t, tfinal, dens_dep_vel)
    # fig_MSD, axes_MSD = plt.subplots()
    # axes_MSD.plot(time_bar, MSD_ABP, color = '#062A06',label = 'ABP')
    # axes_MSD.plot(time_bar, MSD_chiral, color = '#FF0000',label = 'Chiral_ABP')
    # axes_MSD.plot(time_bar, MSD_RTP, color = '#0000FF',label = 'RTP')
    # MSDth_RTP = MSD_RTP_theoretical(alpha, D_t, v, time_bar, k=2)
    # MSDth_ABP = MSD_ABP_theoretical(2*D_r, D_t, v, time_bar, k=2)
    # MSDth_chiral = MSD_chiral_theoretical(D_t, 2*D_r, v, w, time_bar, k=2)
    # axes_MSD.plot(time_bar, MSDth_ABP, marker = '.',color = '#062A06',label = 'thABP')
    # axes_MSD.plot(time_bar, MSDth_chiral, marker = '.',color = '#FF0000',label = 'thChiral_ABP')
    # axes_MSD.plot(time_bar, MSDth_RTP, marker = '.', color = '#0000FF',label = 'thRTP')
    # axes_MSD.legend()
    
    # RTP_video(init_pos_x0, init_pos_y0, v, alpha, delta_t, tfinal, num_of_bots, dens_dep_vel)
    
    # Chiral_ABP_video(init_pos_x0, init_pos_y0, v, D_t, D_r, w, delta_t, tfinal, num_of_bots, dens_dep_vel)
    
    # ABP_list = []
    # for idx in range(num_of_bots):     
    #     ABP_list.append(ABP(init_pos_x0, init_pos_y0, v, D_t, D_r, delta_t, dens_dep_vel))
    # cap_eff = absorbing_target_with_video(ABP_list, tfinal, delta_t, target_x, target_y, target_rad, num_of_bots, dens_dep_vel)
    # time_bar = [*np.arange(0,tfinal,delta_t)]
    # plt.plot(time_bar,cap_eff,'b-')
    # plt.grid(True)
    
    # env1 = environment()
    # env1.disp_env()
    # env1.poly1.distance(sp.Point(9,15))
    # ax = plt.gca(projection='3d')
    
    #all test runs
    # print("ABP")
    # start = time.time()
    # RTP_list = []
    # ABP_list = []
    # ChiralABP_list = []
    # for idx in range(num_of_bots): 
    #     ABP_list.append(ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel))
    #     # RTP_list.append(RTP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
    #     # ChiralABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel))
    # cap_eff,_ = absorbing_target_video(ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    # time_bar = [*np.arange(0,tfinal,delta_t)]
    # plt.clf()
    # plt.plot(time_bar,cap_eff,'b-')
    # plt.grid(True)
    # end = time.time()
    # print(end-start)
    
    # print("Chiral ABP")
    # start = time.time()
    # RTP_list = []
    # ABP_list = []
    # ChiralABP_list = []
    # for idx in range(num_of_bots): 
    #     # ABP_list.append(ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel))
    #     # RTP_list.append(RTP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
    #     ChiralABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel))
    # cap_eff,_ = absorbing_target_video(ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    # time_bar = [*np.arange(0,tfinal,delta_t)]
    # plt.clf()
    # plt.plot(time_bar,cap_eff,'b-')
    # plt.grid(True)
    # end = time.time()
    # print(end-start)
    
    # print("RTP")
    # start = time.time()
    # RTP_list = []
    # ABP_list = []
    # ChiralABP_list = []
    # for idx in range(num_of_bots): 
    #     # ABP_list.append(ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel))
    #     RTP_list.append(RTP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
    #     # ChiralABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel))
    # cap_eff,_ = absorbing_target_video(RTP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    # time_bar = [*np.arange(0,tfinal,delta_t)]
    # plt.clf()
    # plt.plot(time_bar,cap_eff,'b-')
    # plt.grid(True)
    # end = time.time()
    # print(end-start)
    
    # Capture efficiency and MSD plot for N simulations averaging it out
    start = time.time()
    num_sim = 20
    num_rows = 2
    num_cols = 3
    num_of_plots = num_cols*num_rows
    fig, axes = plt.subplots(num_rows,num_cols,)
    if MSD_flag:
        fig_MSD, axes_MSD = plt.subplots()
    # x_values = np.linspace(5,35,num_of_plots)
    x_values = [6, 12, 15.5, 17, 20, 28]
    # x_values = [3, 6, 7.75, 8.5, 10, 14]
    i = 0
    time_bar = [*np.arange(0,tfinal,delta_t)]
    MA_avg = [0]*len(time_bar)
    MC_avg = [0]*len(time_bar)
    MR_avg = [0]*len(time_bar)
    for row in range(num_rows):
        for col in range(num_cols):
            CEA_all =[]
            CEC_all =[]
            CER_all =[]
            print(f'graph {row},{col}')
            for sim in range(num_sim):
                print(f'simulation {sim}')
                target_x = x_values[i]*R_scale
                target1 = target(target_x, target_y, target_z, target_rad)
                ax = axes[row,col]
                if (row,col) == (1,2):
                    CEA,CEC,CER,MA,MC,MR = abs_target_cap_eff(target1, ax, init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, alpha, delta_t, dens_dep_vel, tfinal, num_of_bots, video_flag, MSD_flag)
                else:
                    CEA,CEC,CER,MA,MC,MR = abs_target_cap_eff(target1, ax, init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, alpha, delta_t, dens_dep_vel, tfinal, num_of_bots, video_flag, 0)
                # ABP_list = []
                # RTP_list = []
                # ChiralABP_list = []
                # for idx in range(num_of_bots): 
                #     ABP_list.append(ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, delta_t, dens_dep_vel))
                #     ChiralABP_list.append(Chiral_ABP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, w, delta_t, dens_dep_vel))
                #     RTP_list.append(RTP(init_pos_x0, init_pos_y0, init_pos_z0, v, R, alpha, delta_t, dens_dep_vel))
                # CEA = absorbing_target_video(ABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
                # CER = absorbing_target_video(RTP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
                # CEC = absorbing_target_video(ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
                CEC_all.append(CEC)
                CEA_all.append(CEA)
                CER_all.append(CER)
                # ax.plot(time_bar,CEA,color = '#A1F1A1')
                # ax.plot(time_bar,CER,color = '#ADD8E6')
                # ax.plot(time_bar,CEC,color = '#FFCCCC')
                if MSD_flag and (row,col) == (1,2):
                    MA_avg = [(x+y)/2 for x,y in zip(MA_avg,MA)]
                    MC_avg = [(x+y)/2 for x,y in zip(MC_avg,MC)]
                    MR_avg = [(x+y)/2 for x,y in zip(MR_avg,MR)]
            if MSD_flag:
                axes_MSD.plot(time_bar, [target_x**2]*len(time_bar),color = ((59+i*30)/255,0,(59+i*30)/255),label=f'target_distance={np.round(target_x/P_l,2)}$P_l$')
            CEA_data = np.array(CEA_all)
            CEA_avg = np.mean(CEA_data,axis=0)
            ax.plot(time_bar,CEA_avg,color = '#062A06',label = 'ABP')
            CER_data = np.array(CER_all)
            CER_avg = np.mean(CER_data,axis=0)
            ax.plot(time_bar,CER_avg,color = '#0000FF',label = 'RTP')
            CEC_data = np.array(CEC_all)
            CEC_avg = np.mean(CEC_data,axis=0)
            with open('data_sim_{0}_{1}.pkl'.format(row,col), 'wb') as f:
                pickle.dump([CEA_data, CEC_data, CER_data, CEA_avg, CEC_avg, CER_avg], f)
            ax.plot(time_bar,CEC_avg,color = '#FF0000',label = 'Chiral_ABP')
            ax.plot()
            # ax.legend()
            # ax.grid(True)
            ax.set_title(f'target distance = {np.round(target_x/P_l,2)}$P_l$', fontsize = 26)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            i+=1
    if MSD_flag:
        with open('data_sim_MSD_{0}_{1}.pkl'.format(row,col), 'wb') as f:
            pickle.dump([MA_avg, MC_avg, MR_avg], f)
        # with open('data_100_sims/data_sim_MSD_{0}_{1}.pkl'.format(row,col), 'rb') as f:
        #     MA_avg, MC_avg, MR_avg = pickle.load(f)
        # MthC = MSD_chiral_theoretical(D_t,D_r,v,w, time_bar, k = 1)
        # axes_MSD.plot(time_bar, MthC, color='#000000', label = 'MSD_theoretical_chiral')
        plot_MSD(axes_MSD, MA_avg, MC_avg, MR_avg, tfinal, delta_t)
        axes_MSD.legend()
        # axes_MSD.grid()
        fig_MSD.suptitle(f'Varying target location MSD (P_l = {P_l:0.3f}, P_e = {P_e:0.3f}, t_r = {1/D_r : 0.3f})', fontsize=20)
        fig_MSD.supxlabel('time (s)', fontsize = 20)
        fig_MSD.supylabel('MSD ($\mu$m)', fontsize = 20)
        plt.subplots_adjust(top=0.88,
                            bottom=0.09,
                            left=0.1,
                            right=0.95,
                            hspace=0.3,
                            wspace=0.18)
    # fig.suptitle(f'Varying target location capture efficiency (P_l = {P_l:0.3f}, P_e = {P_e:0.3f}, t_r = {1/D_r : 0.3f})')
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3], fontsize = 15, loc = (0.2,0.55))
    fig.supxlabel('time (s)', fontsize = 30)
    fig.supylabel('Capture efficiency', fontsize = 30)
    plt.subplots_adjust(top=0.876,
                        bottom=0.094,
                        left=0.100,
                        right=0.947,
                        hspace=0.305,
                        wspace=0.184)
    #fig.tight_layout()
    end = time.time()
    print(f'time taken = {(end-start)/60 : 0.2f}')
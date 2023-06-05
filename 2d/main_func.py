# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:12:19 2021

@author: Akshatha
"""

import numpy as np
import scipy.constants as const
import time
from models import ABP,Chiral_ABP,RTP,target,environment
from utility import *
import matplotlib.pyplot as plt
import pickle

# =============================================================================
# Main-fn begins
# =============================================================================
if __name__ == "__main__":
    #initializations begin
    R = 0.5e-6;#bot size (radius) in m
    R_scale = R/1e-6
    v = 10#20*R_scale;#speed (um per sec)
    tfinal = 100;
    num_of_bots = 1000;
    delta_t = 0.1;
    w = 2*R_scale;#%rad/s angular velocity for Chiral ABP
    init_pos = [(0,0) for i in range(num_of_bots)];
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
    alpha = D_r;#%event rate for RTP 
    P_l = v/D_r #persistence length in um
    P_e = v/np.sqrt(D_t*D_r) #peclet number
    
    #target specifications
    target_x = 40*R_scale;
    target_y = 0;
    target_rad = 0*R_scale;
    target1 = target(target_x, target_y, target_rad)
    
    #create environment
    env1 = environment()
    #initializations end
    
    # #MSD plot
    # # with open('data_sim_0_MSD.pkl', 'rb') as f:
    # #     MSD_ABP, MSD_chiral, MSD_RTP = pickle.load(f)
    # MSD_ABP,MSD_chiral,MSD_RTP = calc_all_MSD(env1, num_of_bots, init_pos, v, R, w, alpha, delta_t, tfinal, dens_dep_vel)
    # fig_MSD, axes_MSD = plt.subplots()
    # axes_MSD.plot(time_bar, MSD_ABP, color = '#062A06',label = 'ABP')
    # axes_MSD.plot(time_bar, MSD_chiral, color = '#FF0000',label = 'Chiral_ABP')
    # axes_MSD.plot(time_bar, MSD_RTP, color = '#0000FF',label = 'RTP')
    # MSDth_RTP = MSD_RTP_theoretical(alpha, D_t, v, time_bar, k=2)
    # MSDth_ABP = MSD_ABP_theoretical(D_r, D_t, v, time_bar, k=2)
    # MSDth_chiral = MSD_chiral_theoretical(D_t, D_r, v, w, time_bar, k=2)
    # with open('data_sim_1_MSD.pkl', 'wb') as f:
    #     pickle.dump([MSD_ABP, MSD_chiral, MSD_RTP], f)
    # # marker = '.'
    # axes_MSD.plot(time_bar, MSDth_ABP,marker = '.',color = '#062A06',label = 'thABP')
    # axes_MSD.plot(time_bar, MSDth_chiral,marker = '.',color = '#FF0000',label = 'thChiral_ABP')
    # axes_MSD.plot(time_bar, MSDth_RTP, marker = '.',color = '#0000FF',label = 'thRTP')
    # axes_MSD.legend()
    
          
    # ABP_sample_video(env1, init_pos, v, R, delta_t, tfinal, num_of_bots, dens_dep_vel)
    # RTP_sample_video(env1, init_pos, v, R, alpha, delta_t, tfinal, num_of_bots, dens_dep_vel)
    # Chiral_ABP_sample_video(env1, init_pos, v, R, w, delta_t, tfinal, num_of_bots, dens_dep_vel)             
    
    
        # #Capture efficiency and MSD plot for N simulations averaging it out
    # path_loc = 'C:\\Users\\Akshatha\\OneDrive - Indian Institute of Science\\Work_3rd_year\\Research_Aug_2019\\Simulations\\Python_switch\\outputs\\testing_new'
    # start = time.time()
    # # init_pos = [(x,y) for x in np.linspace(-30,30,31) for y in np.linspace(-30,30,31)]
    # init_pos = [(0,0) for i in range(num_of_bots)]
    # num_sim = 20
    # num_rows = 2
    # num_cols = 3
    # num_of_plots = num_cols*num_rows
    # fig, axes = plt.subplots(num_rows,num_cols,)
    # if MSD_flag:
    #     fig_MSD, axes_MSD = plt.subplots()
    # # x_values = np.linspace(5,35,num_of_plots)
    # x_values_org = [6, 12, 15.5, 17, 20, 28]
    # x_values = [1*x for x in x_values_org]
    # i = 0
    # time_bar = [*np.arange(0,tfinal,delta_t)]
    # MA_avg = [0]*len(time_bar)
    # MC_avg = [0]*len(time_bar)
    # MR_avg = [0]*len(time_bar)
    # for row in range(num_rows):
    #     for col in range(num_cols):
    #         CEA_all =[]
    #         CEC_all =[]
    #         CER_all =[]
    #         print(f'graph {row},{col}')
    #         # with open('data_100_sims/data_sim_{0}_{1}.pkl'.format(row,col), 'rb') as f:
    #         #     CEA_data, CEC_data, CER_data, CEA_avg, CEC_avg, CER_avg = pickle.load(f)
    #         for sim in range(num_sim):
    #             target_x = x_values[i]*R_scale
    #             ax = axes[row,col]
    #             # ax.plot(time_bar,CEA_data[sim],color = '#A1F1A1')
    #             # ax.plot(time_bar,CEC_data[sim],color = '#FFCCCC')
    #             # ax.plot(time_bar,CER_data[sim],color = '#ADD8E6')
    #             print(f'simulation {sim}')
    #             target1 = target(target_x, target_y, target_rad)
    #             if (row,col) == (1,2):
    #                 CEA,CEC,CER,MA,MC,MR = abs_target_cap_eff(target1, env1, ax, init_pos, v, R, w, alpha, delta_t, dens_dep_vel, tfinal, num_of_bots, video_flag, MSD_flag)
    #             else:
    #                 CEA,CEC,CER,MA,MC,MR = abs_target_cap_eff(target1, env1, ax, init_pos, v, R, w, alpha, delta_t, dens_dep_vel, tfinal, num_of_bots, video_flag, 0)
    #             ax.plot(time_bar,CEA,color = '#A1F1A1')
    #             ax.plot(time_bar,CEC,color = '#FFCCCC')
    #             ax.plot(time_bar,CER,color = '#ADD8E6')
    #             CEA_all.append(CEA)
    #             CEC_all.append(CEC)
    #             CER_all.append(CER)
    #             # if MSD_flag and (row,col) == (1,2):
    #             #     MA_avg = [(x+y)/2 for x,y in zip(MA_avg,MA)]
    #             #     MC_avg = [(x+y)/2 for x,y in zip(MC_avg,MC)]
    #             #     MR_avg = [(x+y)/2 for x,y in zip(MR_avg,MR)]
    #         if MSD_flag:
    #             axes_MSD.plot(time_bar, [target_x**2]*len(time_bar),color = ((59+i*30)/255,0,(59+i*30)/255),label=f'target_distance={np.round(target_x/P_l,2)}$P_l$')
    #         CEA_data = np.array(CEA_all)
    #         CEC_data = np.array(CEC_all)
    #         CER_data = np.array(CER_all)
    #         CEA_avg = np.mean(CEA_data,axis=0)
    #         CEC_avg = np.mean(CEC_data,axis=0)
    #         CER_avg = np.mean(CER_data,axis=0)
    #         with open('data_sim_{0}_{1}.pkl'.format(row,col), 'wb') as f:
    #             pickle.dump([CEA_data, CEC_data, CER_data, CEA_avg, CEC_avg, CER_avg], f)      
    #         ax.plot(time_bar,CEA_avg,color = '#062A06',label = 'ABP')
    #         ax.plot(time_bar,CEC_avg,color = '#FF0000',label = 'Chiral_ABP')
    #         ax.plot(time_bar,CER_avg,color = '#0000FF',label = 'RTP')
    #         ax.plot()
    #         ax.grid(False)
    #         # ax.legend()
    #         ax.set_title(f'target distance = {np.round(target_x/P_l,2)}$P_l$', fontsize = 20)
    #         ax.tick_params(axis='x', labelsize=20)
    #         ax.tick_params(axis='y', labelsize=20)
    #         i+=1
    # if MSD_flag:
    #     # with open('data_sim_MSD_{0}_{1}.pkl'.format(row,col), 'wb') as f:
    #     #     pickle.dump([MA_avg, MC_avg, MR_avg], f)
    #     with open('data_100_sims/data_sim_MSD_{0}_{1}.pkl'.format(row,col), 'rb') as f:
    #         MA_avg, MC_avg, MR_avg = pickle.load(f)
    #     # MthC = MSD_chiral_theoretical(D_t,D_r,v,w, time_bar, k = 1)
    #     # axes_MSD.plot(time_bar, MthC, color='#000000', label = 'MSD_theoretical_chiral')
    #     plot_MSD(axes_MSD, MA_avg, MC_avg, MR_avg, tfinal, delta_t)
    #     axes_MSD.legend()
    #     # axes_MSD.grid()
    #     fig_MSD.suptitle(f'Varying target location MSD (P_l = {P_l:0.3f}, P_e = {P_e:0.3f}, t_r = {1/D_r : 0.3f})', fontsize=20)
    #     fig_MSD.supxlabel('time (s)', fontsize = 20)
    #     fig_MSD.supylabel('MSD ($\mu$m)', fontsize = 20)
    #     plt.subplots_adjust(top=0.88,
    #                         bottom=0.09,
    #                         left=0.1,
    #                         right=0.95,
    #                         hspace=0.3,
    #                         wspace=0.18)
    # # fig.suptitle(f'Varying target location capture efficiency (P_l = {P_l:0.3f}, P_e = {P_e:0.3f}, t_r = {1/D_r : 0.3f})')
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines[:3], labels[:3], fontsize = 15, loc = (0.8,0.55))
    # fig.supxlabel('time (s)', fontsize = 20)
    # fig.supylabel('Capture efficiency', fontsize = 20)
    # plt.subplots_adjust(top=0.876,
    #                     bottom=0.094,
    #                     left=0.100,
    #                     right=0.947,
    #                     hspace=0.305,
    #                     wspace=0.184)
    # # fig.tight_layout()
    # plt.savefig(os.path.join(path_loc,'imgs1'))
    # end = time.time()
    # print(f'time taken = {(end-start)/60 : 0.2f}')
    
    
    # #effect of w
    # start = time.time()
    # w_vals = [0,0.2,0.5,0.8,1,3.14,5,6.28,8,10]
    # tfinal = 35
    # time_bar = [*np.arange(0,tfinal,delta_t)]
    # CEC_all = []
    # fig_w, ax = plt.subplots()
    # times = [0.5, round(1/D_r,1), 5, 10, 15, 30]
    # fig_2, ax_2 = plt.subplots(nrows=2,ncols=3)
    # for w in w_vals:
    #     ChiralABP_list = []
    #     for idx in range(num_of_bots):     
    #         ChiralABP_list.append(Chiral_ABP(init_pos[idx], v, R, w, delta_t, dens_dep_vel))
    #     cap_eff_Chiral_ABP,bot_list_chiral = absorbing_target_video(env1, ChiralABP_list, tfinal, delta_t, target1, num_of_bots, dens_dep_vel, video_flag, MSD_flag)
    #     CEC_all.append(cap_eff_Chiral_ABP)
    #     MSD_chiral = calc_MSD(bot_list_chiral)
    #     ax.plot(time_bar,cap_eff_Chiral_ABP,label = w)
    #     # ax.grid(True)
    #     ax.legend()
    # CEC_data = np.array(CEC_all)
    # fig_w.suptitle('effect of $\omega$ on Chiral ABP capture efficiency')
    # fig_w.supxlabel('time (s)')
    # fig_w.supylabel('Capture efficiency')
    # i = 0
    # for row in range(2):
    #     for col in range(3):
    #         ax2 = ax_2[row,col]
    #         t = times[i]
    #         ax2.plot(w_vals,CEC_data[:,int(t/delta_t)],'*-') 
    #         # ax2.grid(True)
    #         ax2.set_title(f'at time = {np.round(t*D_r,2)}$\\tau _r$',fontsize=30)
    #         ax2.tick_params(axis='x', labelsize=20)
    #         ax2.tick_params(axis='y', labelsize=20)
    #         i+=1
    # fig_2.suptitle('effect of $\omega$ on Chiral ABP capture efficiency',fontsize=20)
    # fig_2.supxlabel('$\omega$ (rad/s)', fontsize = 30)
    # fig_2.supylabel('Capture efficiency', fontsize = 30)
    # plt.subplots_adjust(top=0.876,
    #                     bottom=0.094,
    #                     left=0.100,
    #                     right=0.947,
    #                     hspace=0.305,
    #                     wspace=0.184)
    # # plt.yticks(fontsize=20)
    # # plt.xticks(fontsize=20)
    # end = time.time()
    # print(f'time taken = {(end-start)/60 : 0.2f}')
#!/usr/bin/env python
# coding: utf-8

# In[172]:


import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import pathlib
np.set_printoptions(precision=3, suppress=True)


# In[ ]:


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
arg_pass=argparse.ArgumentParser()
arg_pass.add_argument(
  "--Env",
  help='Env name; default RW - RW:RandomWalk, TC: Temperature Control, FT: Four Tank',
  default='RW', 
)
arg_pass.add_argument(
  "--input_dir",
  help='Input file directory; default./Results/RandomWalk/RW_s5_r100_e100.csv',
  default='./Results/RandomWalk/RW_s5_r100_e100.csv',
)
arg_pass.add_argument(
  "--TA_dir",
  help='TA-explore file directory; default ./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta1_E4000.csv',
  default='./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta1_E4000.csv',
)
arg_pass.add_argument(
  "--PPO_dir",
  help='Baseline file directory; default ./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta0_E4000.csv',
  default='./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta0_E4000.csv',
)
arg_pass.add_argument(
  "--A2C_dir",
  help='Baseline file directory; default ./Results/Temperature_Control/TF/Temperature_Control_A2C_omega1_R^T_reward_seed0/progress.csv',
  default='./Results/Temperature_Control/TF/Temperature_Control_A2C_omega1_R^T_reward_seed0/progress.csv',
)
arg_pass.add_argument(
  "--DDPG_dir",
  help='Baseline file directory; default ./Results/Temperature_Control/TF/Temperature_Control_DDPG_omega1_R^T_reward_seed0/progress.csv',
  default='./Results/Temperature_Control/TF/Temperature_Control_DDPG_omega1_R^T_reward_seed0/progress.csv',
)
arg_pass.add_argument(
  "--SAC_dir",
  help='Baseline file directory; default ./Results/Temperature_Control/TF/Temperature_Control_SAC_omega1_R^T_reward_seed0/progress.csv',
  default='./Results/Temperature_Control/TF/Temperature_Control_SAC_omega1_R^T_reward_seed0/progress.csv',
)
arg_pass.add_argument(
  "--TD3_dir",
  help='Baseline file directory; default ./Results/Temperature_Control/TF/Temperature_Control_TD3_omega1_R^T_reward_seed0/progress.csv',
  default='./Results/Temperature_Control/TF/Temperature_Control_TD3_omega1_R^T_reward_seed0/progress.csv',
)
arg_pass.add_argument(
  "--save_dir",
  help='Save plots directory; default ./Results/RandomWalk/',  
  default='./Results/RandomWalk/',
)
arg_pass.add_argument(
  "--O_RT",
  nargs='*', help='Plot O_RT; default False',
  type=str2bool, 
  default=False,
)
arg_pass.add_argument(
  "--O_RA",
  nargs='*', help='Plot O_RA; default False',
  type=str2bool,  
  default=False,
)

arg_pass.add_argument(
  "--x_min",
   help='lower bound xlim; default 0',
  type=int,  
  default=0,
)
arg_pass.add_argument(
  "--x_max",
   help='upper bound xlim; default 8000',
  type=int,  
  default=8000,
)
arg_pass.add_argument(
  "--y_min",
   help='lower bound ylim; default -500',
  type=int,  
  default=-500,
)
arg_pass.add_argument(
  "--y_max",
   help='upper bound ylim; default 500',
  type=int,  
  default=500,
)
args = arg_pass.parse_args()
Env  = args.Env
Input_file_dir = args.input_dir
TA_DIR  = args.TA_dir
PPO_DIR = args.PPO_dir
A2C_DIR = args.A2C_dir
DDPG_DIR= args.DDPG_dir
SAC_DIR = args.SAC_dir
TD3_DIR = args.TD3_dir
Save_fig_dir= args.save_dir
O_RT = args.O_RT[0]
O_RA = args.O_RA[0]
X_min= args.x_min
X_max= args.x_max
Y_min= args.y_min
Y_max= args.y_max


# In[173]:


def RW_plot(input_file_dir, save_fig_dir, o_RT, o_RA):
    with open(input_file_dir, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
    data=np.array(data).astype(float)
    episode = np.amax(data[:,3]).astype(int)

    only_RT, only_RA, TA = [], [], []
    for i in range(len(data)):
        if data[i,1]==0 and data[i,2]==0:
            only_RT.append(data[i])
        if data[i,1]==1 and data[i,2]==1:
            only_RA.append(data[i])
        if (not(data[i,1]==1 and data[i,2]==1)) and (not(data[i,1]==0 and data[i,2]==0)):
            TA.append(data[i])

    name=input_file_dir.split("/")[-1][:-4]
    nstate=name.split("_")[1][1:]
    isExist = os.path.exists(save_fig_dir)
    if not isExist:
        os.makedirs(save_fig_dir)
    rt=np.array(only_RT)
    ra=np.array(only_RA)
    ta=np.array(TA)
    plt.figure(figsize=[10, 7], dpi=72)
    if rt.size == 0:
        print('Warning: There are no Only R_T Results in inserted address')
    if ra.size == 0:
        print('Warning: There are no Only R_A Results in inserted address')
    if o_RT and (not rt.size == 0):
        plt.plot(rt[:,3], rt[:,5],'.', label=r'Only $R^T$ (Baseline)',linewidth=3, c='k')
    if o_RA and (not ra.size == 0):
        plt.plot(ra[:,3], ra[:,5],'-.', label=r'Only $R^A$',linewidth=3, c='gray') 
    for i in range(int(len(ta)/episode)):
        plt.plot(ta[i*episode:(i+1)*episode ,3], ta[i*episode:(i+1)*episode ,5], 
                 label=r'TA-Explore with $λ = {}$'.format(ta[i*episode,2]),linewidth=3)    

    #plt.margins(.05)
    plt.xlabel('Walks / Episodes')
    plt.ylabel('RMS error')
    plt.legend()
    #plt.title("RMS error of Random Walk ({} States)".format(nstate))
    parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    plt.savefig(save_fig_dir+name+'.png', dpi=144, format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    
    plt.figure()
    for i in range(int(len(ta)/episode)):
        plt.plot(ta[i*episode:(i+1)*episode ,3], ta[i*episode:(i+1)*episode ,4], 
                 label=r'$\beta(0)= {}, λ = {}$'.format(ta[i*episode,1], ta[i*episode,2]),linewidth=3)  
    plt.xlabel('Episodes')
    plt.ylabel(r'$\beta(e) = \beta(0)\lambda^{e}$')
    plt.legend()
    #plt.title(r'$\beta(e) = \beta(0)\lambda^{e}$ for Random Walk')
    parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    plt.savefig(save_fig_dir+name+'beta.png', dpi=144, format=None, metadata=None, bbox_inches=None, 
                pad_inches=0.1, facecolor='auto', edgecolor='auto',backend=None)  
    print("The figures saved in:"+save_fig_dir+name+'.png and ' +save_fig_dir+name+'beta.png')


# In[336]:


def TC_plot(TA_dir, ppo_dir, a2c_dir, ddpg_dir, sac_dir, td3_dir , save_fig_dir, x_min, x_max):
    name=TA_dir.split("/")[-1][:-4]
    isExist = os.path.exists(save_fig_dir)
    if not isExist:
        os.makedirs(save_fig_dir)
    omega=name.split("_")[3][5:]
    dfta = pd.read_csv(TA_dir)
    dfppo = pd.read_csv(ppo_dir)
    dfa2c = pd.read_csv(a2c_dir)
    dfddpg = pd.read_csv(ddpg_dir)
    dfsac = pd.read_csv(sac_dir)
    dftd3 = pd.read_csv(td3_dir)

    dfa2c['time/total_timesteps'] = dfa2c['time/total_timesteps']/100
    dfddpg['time/total_timesteps'] = dfddpg['time/total_timesteps']/100
    dfsac['time/total_timesteps'] = dfsac['time/total_timesteps']/100
    dftd3['time/total_timesteps'] = dftd3['time/total_timesteps']/100
    if int(omega)==100:
        plt.figure(figsize=[10, 6.8], dpi=72)
    else:
        plt.figure(figsize=[10, 5], dpi=72)
    ax = plt.gca()
    parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
    dfta.plot(x="episode", y="average R^T",ax=ax, label='TA-Explore',linewidth=3, c='k')
    dfppo.plot(x="episode", y="average R^T",ax=ax,label='PPO', linewidth=3, c='b')
    dfa2c.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='A2C', linewidth=3, c='g')
    dfddpg.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='DDPG', linewidth=3, c='r')
    dfsac.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='SAC', linewidth=3, c='m') #c
    dftd3.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='TD3', linewidth=3, c='y')
    #plt.title(r'Optimal Temperate Control with Constraint (${}\Vert a\Vert ^2$)'.format(omega))
    plt.ylabel('Average Reward $R^T$')
    if int(omega)==100:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3, fancybox=True, shadow=False)
        plt.xlabel('Episodes')
    else:
        plt.legend().set_visible(False)
        plt.xlabel('')
        
    plt.xlim((x_min, x_max)) 
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    plt.savefig(save_fig_dir+name+'.png', dpi=144, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto', backend=None)

    plt.figure(figsize=[10, 5], dpi=72)
    ax = plt.gca()
    dfta.plot(x="episode", y="beta",ax=ax, label='TA-Explore',linewidth=3, c='k')
    #plt.title(r'$\beta(e) = max[(E-e)\beta(0)/E,0]$ for Optimal Temperate Control with Constraint')
    plt.xlabel('Episodes')
    plt.ylabel(r'$\beta(e)$')
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    plt.savefig(save_fig_dir+name+'beta.png', dpi=144, format=None, metadata=None, bbox_inches=None, 
                    pad_inches=0.1, facecolor='auto', edgecolor='auto',backend=None)  
    print("The figures saved in:"+save_fig_dir+name+'.png and ' +save_fig_dir+name+'beta.png')


# In[344]:


def FT_plot(TA_dir, ppo_dir, a2c_dir, ddpg_dir, sac_dir, td3_dir , save_fig_dir, x_min, x_max, y_min, y_max):
    name=TA_dir.split("/")[-1][:-4]
    isExist = os.path.exists(save_fig_dir)
    if not isExist:
        os.makedirs(save_fig_dir)
    omega=name.split("_")[3][5:]
    dfta = pd.read_csv(TA_dir)
    dfppo = pd.read_csv(ppo_dir)
    dfa2c = pd.read_csv(a2c_dir)
    dfddpg = pd.read_csv(ddpg_dir)
    dfsac = pd.read_csv(sac_dir)
    dftd3 = pd.read_csv(td3_dir)

    dfa2c['time/total_timesteps'] = dfa2c['time/total_timesteps']/100
    dfddpg['time/total_timesteps'] = dfddpg['time/total_timesteps']/100
    dfsac['time/total_timesteps'] = dfsac['time/total_timesteps']/100
    dftd3['time/total_timesteps'] = dftd3['time/total_timesteps']/100

    plt.figure(figsize=[10, 10], dpi=72)
    ax = plt.gca()
    parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
    dfta.plot(x="episode", y="average R^T",ax=ax, label='TA-Explore',linewidth=3, c='k')
    dfppo.plot(x="episode", y="average R^T",ax=ax,label='PPO', linewidth=3, c='b')
    dfa2c.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='A2C', linewidth=11, style='g')
    dfddpg.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='DDPG', linewidth=9, style='r', alpha=0.7)
    dfsac.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='SAC', linewidth=3, c='m') #c
    dftd3.plot(x="time/total_timesteps", y="rollout/ep_rew_mean",ax=ax,label='TD3', linewidth=3, c='y', alpha=0.5)
    #plt.title(r'Coupled Four Tank MIMO System (${}\Vert a\Vert ^2$)'.format(omega))
    plt.ylabel('Average Reward $R^T$')
    plt.xlabel('Episodes')
    plt.xlim((x_min, x_max)) 
    plt.ylim((y_min, y_max)) 
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    plt.savefig(save_fig_dir+name+'.png', dpi=144, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto', backend=None)

    plt.figure(figsize=[10, 5], dpi=72)
    ax = plt.gca()
    dfta.plot(x="episode", y="beta",ax=ax, label='TA-Explore',linewidth=3, c='k')
    #plt.title(r'$\beta(e) = max[(E-e)\beta(0)/E,0]$ for Coupled Four Tank MIMO System')
    plt.xlabel('Episodes')
    plt.ylabel(r'$\beta(e)$')
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    plt.savefig(save_fig_dir+name+'beta.png', dpi=144, format=None, metadata=None, bbox_inches=None, 
                    pad_inches=0.1, facecolor='auto', edgecolor='auto',backend=None)  
    print("The figures saved in:"+save_fig_dir+name+'.png and ' +save_fig_dir+name+'beta.png')


# In[ ]:


if Env=='RW' or Env=='RandomWalk':
    print ('Env:', Env, '\nInput_file_dir:', Input_file_dir, '\nSave_fig_dir:', 
       Save_fig_dir, '\nShow O_RT:', O_RT, '\nShow O_RA:', O_RA)
    RW_plot(input_file_dir=Input_file_dir, save_fig_dir=Save_fig_dir,o_RT=O_RT,o_RA=O_RA)

if Env=='TC'or Env=='Temperate_Control':
    print ('Env:', Env,'\nTA_DIR:', TA_DIR, '\nPPO_DIR:', PPO_DIR, '\nA2C_DIR:', A2C_DIR, 
           '\nDDPG_DIR:', DDPG_DIR, '\nSAC_DIR:', SAC_DIR,  '\nTD3_DIR:', TD3_DIR, 
           '\nSave_fig_dir:', Save_fig_dir, '\nX_lim: [', X_min, X_max, ']')
    TC_plot(TA_dir=TA_DIR, ppo_dir=PPO_DIR, a2c_dir=A2C_DIR, ddpg_dir=DDPG_DIR, sac_dir=SAC_DIR, td3_dir=TD3_DIR, save_fig_dir=Save_fig_dir, x_min=X_min, x_max=X_max)
    
if Env=='FT'or Env=='Four_Tank':
    print ('Env:', Env,'\nTA_DIR:', TA_DIR, '\nPPO_DIR:', PPO_DIR, '\nA2C_DIR:', A2C_DIR, 
           '\nDDPG_DIR:', DDPG_DIR, '\nSAC_DIR:', SAC_DIR,  '\nTD3_DIR:', TD3_DIR, 
           '\nSave_fig_dir:', Save_fig_dir, '\nX_lim: [', X_min, X_max, ']', '\nY_lim: [', Y_min, Y_max, ']')
    FT_plot(TA_dir=TA_DIR, ppo_dir=PPO_DIR, a2c_dir=A2C_DIR, ddpg_dir=DDPG_DIR, sac_dir=SAC_DIR, td3_dir=TD3_DIR, save_fig_dir=Save_fig_dir, x_min=X_min, x_max=X_max, y_min=Y_min, y_max=Y_max)


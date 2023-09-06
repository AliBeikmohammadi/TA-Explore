#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1:cpu, 0:first gpu
import argparse
import numpy as np
import pandas as pd
import gymnasium #gym
from gymnasium.wrappers import TransformReward
from gymnasium import RewardWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO, A2C, DDPG, SAC, TD3, HER
from stable_baselines3.common.logger import configure

import random
import pylab
import copy
import math
#from tensorboardX import SummaryWriter
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


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
  "--a",
  help='Algorithm (A2C, DDPG, SAC, TD3); default DDPG',
  type=str,  
  default='DDPG',
)
arg_pass.add_argument(
  "--s",
  help='Seed number; default 0',
  type=int,  
  default=0,
)
arg_pass.add_argument(
  "--w",
  help='omega, weighting values to the control objective; default 1',
  type=int, 
  default=1,
)
arg_pass.add_argument(
  "--save_dir", help='Save Directory; default ./Results/Four_Tank/',
  default='./Results/Four_Tank/',
)
arg_pass.add_argument(
  "--d",
  nargs='*', help='debug_level; default False  False:nothing print True:print result per each episode',
  type=str2bool,  
  default=False,
)
args = arg_pass.parse_args()
algo=args.a
SEED=args.s
A0= args.w
Save_results= args.save_dir
Debug= args.d


# In[49]:


class CFTMS(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, a0=1, constraint_based_termination= True, generate_obs_in_valid_range= True, n_state=4, n_action=2, 
                 time_horizon= 100, x_min = 3, x_max = 30, mean_noise=None, cov_noise=None):
        super(CFTMS, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.x_min = x_min
        self.x_max = x_max
        self.L = time_horizon
        self.S = 15**2
        self.s = math.pi*(0.24**2)
        self.s42 = math.pi*(0.19**2)
        self.s32 = math.pi*(0.155**2)
        self.s41 = math.pi*(0.155**2)
        self.s31 = math.pi*(0.19**2)
        self.Kp = 7.687
        self.g = 981
        self.mean_noise= mean_noise if mean_noise else np.zeros(self.n_state)
        self.cov_noise= cov_noise if cov_noise else np.zeros(self.n_state)
        self.t = 0
        self.constraint_based_termination= constraint_based_termination
        self.generate_obs_in_valid_range= generate_obs_in_valid_range
        self.action_space = gymnasium.spaces.Box(low=0, high=12, shape=(self.n_action,), dtype=np.float32)
        self.observation_shape = (self.n_state,)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float64)
        self.observation_space3 = gymnasium.spaces.Box(low=self.x_min, high=self.x_max, shape=self.observation_shape, dtype=np.float64)
        self.a0 = a0
        
    def step(self, action):
        done = False
        self.t +=1
        action_true_range = np.array([((nn-(-1))*(12-0)/(1-(-1)))+0 for nn in action]) #tanh -1 1 to 0 12
        self.new_state1 = (-(self.s*np.sqrt(2*self.g*self.current_state[0])/self.S))+(self.s31*np.sqrt(2*self.g*self.current_state[2])/self.S)+(self.s41*np.sqrt(2*self.g*self.current_state[3])/self.S)
        self.new_state2 = (-(self.s*np.sqrt(2*self.g*self.current_state[1])/self.S))+(self.s32*np.sqrt(2*self.g*self.current_state[2])/self.S)+(self.s42*np.sqrt(2*self.g*self.current_state[3])/self.S)
        self.new_state3 = (-((self.s31+self.s32)*np.sqrt(2*self.g*self.current_state[2])/self.S))+(self.Kp*action_true_range[0]/self.S)
        self.new_state4 = (-((self.s41+self.s42)*np.sqrt(2*self.g*self.current_state[3])/self.S))+(self.Kp*action_true_range[1]/self.S)
        self.new_state = np.array([self.new_state1, self.new_state2, self.new_state3, self.new_state4]) 
        constraint_check = np.all((self.new_state >= self.x_min) & (self.new_state <= self.x_max))
        if constraint_check:
            reward = 0
        else:
            reward = -100
            if self.constraint_based_termination:
                done = True
        cost= -np.sum(np.square(action_true_range))                                                                                                                                                             
        self.current_state = self.new_state
        if self.t==self.L:
            done = True
        if done:
            self.t= 0
        info = {'constraint_reward': reward, 'action_cost':cost}
        actual_total_reward = (self.a0*cost) + reward
        return self.new_state, actual_total_reward, done, False, info
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        if self.generate_obs_in_valid_range:
            self.current_state = self.observation_space3.sample()   
        else:
            self.current_state = self.observation_space.sample() 
        info = {'constraint_reward': 0, 'action_cost':0}
        return self.current_state, info                                                             
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass
env = CFTMS()
check_env(env, warn=True)


# In[54]:


def main(SEED= 0, algo='DDPG', A0=1, Save_results='./Results/Four_Tank/', 
         en='Four_Tank', Debug=False, Constraint_Based_Termination= True,
         Generate_Obs_in_Valid_Range= True):
    if SEED==None:
        np.random.seed(SEED)
    if A0==1: 
        total_timesteps=1500000
    elif A0==10:
        total_timesteps=1500000
    elif A0==100:
        total_timesteps=1500000
    if Debug:
        verbose=1
    else:
        verbose=0
    env = CFTMS(a0=A0, constraint_based_termination= Constraint_Based_Termination, generate_obs_in_valid_range= Generate_Obs_in_Valid_Range)

    if algo=='DDPG':
        fname=en+'_DDPG'+"_omega"+str(A0)+"_R^T_reward"+"_seed"+str(SEED)
        new_logger = configure(Save_results+'TF/'+fname, ["stdout", "csv", "tensorboard"])
        model = DDPG('MlpPolicy', env, learning_rate=0.001, buffer_size=1000000, 
                 learning_starts=100, batch_size=100, tau=0.005, gamma=0.99, 
                 train_freq=(1, 'episode'), gradient_steps=-1, 
                 action_noise=None, replay_buffer_class=None, 
                 replay_buffer_kwargs=None, optimize_memory_usage=False, 
                 tensorboard_log=Save_results+'TF/'+fname, policy_kwargs=None, 
                 verbose=verbose, seed=SEED, device='auto', _init_setup_model=True)
    elif algo=='A2C':
        fname=en+'_A2C'+"_omega"+str(A0)+"_R^T_reward"+"_seed"+str(SEED)
        new_logger = configure(Save_results+'TF/'+fname, ["stdout", "csv", "tensorboard"])
        model = A2C('MlpPolicy', env, learning_rate=0.0007, n_steps=5, gamma=0.99, 
                gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, 
                rms_prop_eps=1e-05, use_rms_prop=True, use_sde=False, 
                sde_sample_freq=-1, normalize_advantage=False, stats_window_size=100, 
                tensorboard_log=Save_results+'TF/'+fname, policy_kwargs=None, 
                verbose=verbose, seed=SEED, device='auto', _init_setup_model=True)
    elif algo=='SAC':
        fname=en+'_SAC'+"_omega"+str(A0)+"_R^T_reward"+"_seed"+str(SEED)
        new_logger = configure(Save_results+'TF/'+fname, ["stdout", "csv", "tensorboard"])
        model = SAC('MlpPolicy', env, learning_rate=0.0003, buffer_size=1000000, 
                learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, 
                train_freq=1, gradient_steps=1, action_noise=None, 
                replay_buffer_class=None, replay_buffer_kwargs=None, 
                optimize_memory_usage=False, ent_coef='auto', 
                target_update_interval=1, target_entropy='auto', 
                use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False, 
                stats_window_size=100, tensorboard_log=Save_results+'TF/'+fname, 
                policy_kwargs=None, verbose=verbose, seed=SEED, 
                device='auto', _init_setup_model=True)
    elif algo=='TD3':
        fname=en+'_TD3'+"_omega"+str(A0)+"_R^T_reward"+"_seed"+str(SEED)
        new_logger = configure(Save_results+'TF/'+fname, ["stdout", "csv", "tensorboard"])
        model = TD3('MlpPolicy', env, learning_rate=0.001, buffer_size=1000000, 
                learning_starts=100, batch_size=100, tau=0.005, gamma=0.99, 
                train_freq=(1, 'episode'), gradient_steps=-1, action_noise=None, 
                replay_buffer_class=None, replay_buffer_kwargs=None, 
                optimize_memory_usage=False, policy_delay=2, 
                target_policy_noise=0.2, target_noise_clip=0.5, 
                stats_window_size=100, tensorboard_log=Save_results+'TF/'+fname, 
                policy_kwargs=None, verbose=verbose, seed=SEED, 
                device='auto', _init_setup_model=True)
    else:
         raise NotImplementedError
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps, tb_log_name=fname)
    model.save(Save_results+'Weights/'+fname) 


# In[ ]:


main(SEED= SEED, algo=algo, A0=A0, Save_results=Save_results, Debug=False, 
     en='Four_Tank', Constraint_Based_Termination= True, Generate_Obs_in_Valid_Range= True)


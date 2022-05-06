#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
np.set_printoptions(precision=3, suppress=True)

arg_pass=argparse.ArgumentParser()
arg_pass.add_argument(
  "--N",
  help='Number of states (including the ending states); default 5',
  type=int,
  default=5, 
)
arg_pass.add_argument(
  "--E",
  help='Number of episodes per each test; default 100',
  type=int,  
  default=100,
)
arg_pass.add_argument(
  "--R",
  help='Number of independent test; default 100',
  type=int,  
  default=100,
)
arg_pass.add_argument(
  "--b",
  nargs='*', help='initial_beta; default [0, 1, 100]  if 0: only R^T, if 100: only only R^A, \nif (0, 1) if adaptive beta*R^A + (1-adaptive beta)*R^T \nwhere adaptive beta = initial_beta*((lambda)**episode)',
  type=float, 
  default=[0 , 1, 100],
)
arg_pass.add_argument(
  "--l",
  nargs='*', help='lambda; default [0.1, 0.5, 0.9] ',
  type=float,  
  default=[0.1 , 0.5, 0.9],
)
arg_pass.add_argument(
  "--d",
  help='debug_level; default 0  0:nothing print 1:print average result over runs, 2:print result after end of each episode, 3:print all information, like actions and reward in each state ',
  type=int, 
  default=0,
)
arg_pass.add_argument(
  "--save_dir", help='Save Directory; default ./Results/RandomWalk/',
  default='./Results/RandomWalk/',
)
args = arg_pass.parse_args()

NumberOfTotalStates= args.N
N_episodes= args.E
N_run = args.R
initial_beta = args.b
cut= args.l
debug_level= args.d
save_results= args.save_dir

Alpha=[0.1] 
main_reward=1    #R^T 
assistant_reward=1  #R^A
adaptive_beta= True
initial_values= np.append(0,np.append(np.random.randn(NumberOfTotalStates-2),0))  # samples from the standard normal distribution (same for all runs)
# def initial_values(NumberOfTotalStates): # samples from the standard normal distribution (different for each run)
#     iv= np.append(0,np.append(np.random.randn(NumberOfTotalStates-2),0))  # samples from the standard normal distribution
#     return iv

print ('debug_level:', debug_level, 
      '\nNumberOfTotalStates:', NumberOfTotalStates, '\nN_episodes:', N_episodes, '\nN_run:', N_run, 
      '\ninitial_beta:', initial_beta, '\nlambda:', cut, '\nAlpha:', Alpha, '\ninitial_values:', initial_values, 
        '\nmain_reward:', main_reward, '\nassistant_reward:', assistant_reward, '\nadaptive_beta:', adaptive_beta,
      '\nsave_results', save_results)

class RandomWalk:
    gamma = 1       # Discount factor
    def __init__(self, alpha = 0.1, NUM_STATES=7, values = np.ones(7)*0.5, debug=1, MR = 1, Beta = 0, AR = 1):
        self.nodes, self.nodes[0], self.nodes[-1]=["state {}".format (i) for i in range(NUM_STATES)], "Left", "Right" 
        self.position = int((NUM_STATES-1)/2)     # Initial position is "C"
        self.node = self.nodes[self.position]
        self.values = values
        self.terminated = False
        self.alpha = alpha
        self.debug = debug
        self.states_track=[self.node]
        self.MR=MR
        self.AR=AR
        self.Beta=Beta
        if self.debug==3:
                    print ("¤¤¤¤¤¤¤ Start state: {}".format(self.node)) 
    def move(self):
        if not self.terminated:
            direction = np.random.choice(["left", "right"])
            if direction == "left":
                new_position = self.position - 1
                if self.debug==3:
                    print ("Action: Moving to the left.| Reward: {} | New state: {}"
                           .format(self.get_reward(self.position, new_position), self.nodes[new_position]))
            elif direction == "right":
                new_position = self.position + 1
                if self.debug==3:
                    print ("Action: Moving to the right.| Reward: {} | New state: {}"
                           .format(self.get_reward(self.position, new_position), self.nodes[new_position]))

            self.update_value_table_future_reward(self.position, new_position)
            self.position = new_position
            self.node = self.nodes[self.position]
            self.states_track.append(self.node)
            if (self.node == "Left") or (self.node == "Right"):
                if self.debug==3:
                    print("¤¤¤¤¤¤¤ End at state {}. | Number of states {}".format(self.node, len(self.states_track)))
                self.terminated = True
        else:
            print ("Moving is not possible as the random walk has already terminated.")

    def get_reward(self, old_position,new_position):
        if new_position > old_position:
            return (self.Beta*self.AR)+((1-self.Beta)*self.MR) if self.nodes[new_position] == "Right" else self.Beta*self.AR
        else:
            return 0

    def update_value_table_immediate_reward(self, old_position, new_position):
        reward = self.get_reward(old_position,new_position)
        self.values[old_position] += self.alpha * (reward - self.values[old_position])

    def update_value_table_future_reward(self, old_position, new_position):
        if self.nodes[new_position] in ("Left","Right"):
            reward = self.get_reward(old_position, new_position)
        else:
            reward = self.get_reward(old_position,new_position) + self.gamma * self.values[new_position]
        self.values[old_position] += self.alpha * (reward - self.values[old_position])

all_values=np.full([len(cut), len(initial_beta), len(Alpha), N_run, N_episodes], None)
rms_over_value_estimates=np.full([len(cut), len(initial_beta), len(Alpha), N_run, N_episodes], None)
av_rms=np.full([len(cut), len(initial_beta), len(Alpha), N_episodes], None)
betass=np.full([len(cut), len(initial_beta)], None)
actual_state_values = np.arange(0, NumberOfTotalStates, 1) / (NumberOfTotalStates-1) 
results_l1= []

for c in range(len(cut)):
    for i in range(len(initial_beta)):
        for a in range(len(Alpha)):
            for n in range(N_run):
                #value_estimates = initial_values(NumberOfTotalStates) #different random values in each run
                value_estimates = np.copy(initial_values) #same random values in all runs
                betas = []
                for episode in range(N_episodes):
                    if adaptive_beta:
                        if initial_beta[i]==100:
                            beta=1
                        else:    
                            beta = initial_beta[i]*((cut[c])**episode)
                    else: 
                        beta=initial_beta[i]
                    RW = RandomWalk(values = value_estimates, debug=debug_level, alpha = Alpha[a], MR = main_reward, 
                                   Beta = beta, AR = assistant_reward, 
                                   NUM_STATES=NumberOfTotalStates)
                    while not RW.terminated:
                        RW.move()
                    all_values[c,i,a,n,episode]=np.copy(RW.values)
                    rms_over_value_estimates[c,i,a,n,episode]=np.sqrt(np.mean([er**2 for er in actual_state_values[1:-1] 
                                                                           - all_values[c,i,a,n,episode][1:-1]]))
                    value_estimates = RW.values                 
                    betas.append(beta)
                    if debug_level==2 or debug_level==3:
                        print("α {:.2f}| Beta {}| lambda {}| Run {}| Episode {}|Estimate state values {}| RMS {:.5f}"
                              .format(Alpha[a],beta, cut[c], n+1, episode+1, all_values[c,i,a,n,episode][1:-1], rms_over_value_estimates[c,i,a,n,episode]))
        betass[c,i]=betas
b0=True
b1=True
for i in range(len(initial_beta)):
    for c in range(len(cut)):
        for a in range(len(Alpha)):
            for episode in range(N_episodes):
                av_rms[c,i,a,episode] = np.mean(rms_over_value_estimates[c,i,a,:,episode], axis=0)
                if (initial_beta[i]==100) and b1:
                    results_l1.append([Alpha[a], betass[c,i][0], 1, episode+1, betass[c,i][episode], av_rms[c,i,a,episode]])
                if (initial_beta[i]==0) and b0:
                    results_l1.append([Alpha[a], betass[c,i][0], 0, episode+1, betass[c,i][episode], av_rms[c,i,a,episode]])
                if (not initial_beta[i]==0) and (not initial_beta[i]==100):
                    results_l1.append([Alpha[a], betass[c,i][0], cut[c], episode+1, betass[c,i][episode], av_rms[c,i,a,episode]])
                if debug_level==1 or debug_level==2 or debug_level==3:
                    print("α {:.2f}| Beta {}| lambda {}| Episode {}| Total Run {} | AV_RMS {:.5f}"
                          .format(Alpha[a], betass[c,i][episode], cut[c], episode+1, N_run, av_rms[c,i,a,episode]))
            if initial_beta[i]==0:
                b0=False
            if initial_beta[i]==100:
                b1=False

data1 = pd.DataFrame(results_l1)
data1.columns = ["alpha", "beta0", "lambda", "Episode" , "beta", "Average over RMS of Value Estimates"]

isExist = os.path.exists(save_results)
if not isExist:
    os.makedirs(save_results)
data1.to_csv(save_results+'RW_s{}_r{}_e{}.csv'.format(NumberOfTotalStates, N_run, N_episodes), index = False)
print("The results saved in:"+save_results+'RW_s{}_r{}_e{}.csv'.format(NumberOfTotalStates, N_run, N_episodes))
            


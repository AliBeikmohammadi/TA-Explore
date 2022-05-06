#!/usr/bin/env python
# coding: utf-8

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1:cpu, 0:first gpu
import argparse
import numpy as np
import pandas as pd
import gym
import random
import pylab
import copy
import math
from tensorboardX import SummaryWriter
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
  "--e",
  help='Total episodes to train through all environments; default 30000',
  type=int,  
  default=30000,
)
arg_pass.add_argument(
  "--b",
  help='initial_beta; default 0.5  if 0: only R^T, \nif (0, 1] adaptive_beta*R^A + (1-adaptive_beta)*R^T \nwhere beta = max((E-e)*initial_beta/E),0)',
  type=float, 
  default=0.5,
)
arg_pass.add_argument(
  "--E",
  help='Episode in which the beta is zero; default 3000',
  type=int,  
  default=3000,
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

n_EPISODES= args.e
Initial_Beta = args.b
CUT = args.E
A0= args.w
Save_results= args.save_dir
Debug= args.d[0]

en= 'Four_Tank' 
Constraint_Based_Termination= True
Generate_Obs_in_Valid_Range= True

print('Env Name:', 'Four_Tank','\nNumber of total episodes:', n_EPISODES,
      '\nInitial_Beta:', Initial_Beta, '\nEpisode in which the beta is zero:', CUT, '\nOmega:', A0,
      '\nConstraint_Based_Termination:', Constraint_Based_Termination, 
      '\nGenerate_Obs_in_Valid_Range:', Generate_Obs_in_Valid_Range,
      '\nSave_results=', './Results/Four_Tank/', '\nDebug:', Debug)

class Control(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, constraint_based_termination= True, generate_obs_in_valid_range= True, n_state=4, n_action=2, 
                 time_horizon= 100, x_min = 3, x_max = 30, mean_noise=None, cov_noise=None):
        super(Control, self).__init__()
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
        self.action_space = gym.spaces.Box(low=0, high=12, shape=(self.n_action,), dtype=np.float32)
        self.observation_shape = (self.n_state,)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        self.observation_space3 = gym.spaces.Box(low=self.x_min, high=self.x_max, shape=self.observation_shape, dtype=np.float32)
    
    def step(self, action_scaled):
        done = False
        self.t +=1
        action_true_range = np.array([((nn-(-1))*(12-0)/(1-(-1)))+0 for nn in action_scaled]) #tanh -1 1 to 0 12
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
        info = {}
        return self.new_state, reward, cost, done, info
    
    def reset(self, seed = None):
        if self.generate_obs_in_valid_range:
            self.current_state = self.observation_space3.sample()
        else:
            self.current_state = self.observation_space.sample()                                                          
        return self.current_state                                                               
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass
                                                                       
    def seed(self, seed=None):                                                                                                              
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="tanh")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(learning_rate=lr))

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages)
        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        V = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(learning_rate=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        return loss
    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

class PPOAgent:

    def __init__(self, EPISODES, env, env_name, initial_beta, cut, a0, save_results, debug, model_name=""):
        self.initial_beta = initial_beta
        self.cut = cut
        self.a0 = a0
        self.debug = debug
        self.save_results=save_results
        self.env_name = env_name
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = EPISODES 
        self.episode = 0
        self.max_average = -10000
        self.lr = 0.00025
        self.epochs = 10
        self.shuffle = True
        self.Training_batch = 512
        self.optimizer = Adam

        self.replay_count = 0
#         self.writer = SummaryWriter(logdir= self.save_results+self.env_name+"_e"+str(self.EPISODES)+"_omega"+str(self.a0)+"_beta"+str(self.initial_beta)+"_E"+str(self.cut))
        self.scores_, self.episodes_, self.average_ = [], [], []
        self.o_scores_, self.o_episodes_,self.o_average_ = [], [], []
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

        self.Actor_name = f"{self.env_name}_e{self.EPISODES}_omega{self.a0}_beta{self.initial_beta}_E{self.cut}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_e{self.EPISODES}_omega{self.a0}_beta{self.initial_beta}_E{self.cut}_PPO_Critic.h5"
        #self.load() # uncomment to continue training from old weights

        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)
        
        isExist = os.path.exists(self.save_results)
        if not isExist:
            os.makedirs(self.save_results)

    def act(self, state):
        pred = self.Actor.predict(state)
        low, high = -1.0, 1.0
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)
        logp_t = self.gaussian_likelihood(action, pred, self.log_std)
        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        y_true = np.hstack([advantages, actions, logp_ts])
        
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)
#         self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
#         self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
#         self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
#         self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.save_results+self.Actor_name)
        self.Critic.Critic.load_weights(self.save_results+self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.save_results+self.Actor_name)
        self.Critic.Critic.save_weights(self.save_results+self.Critic_name)

    pylab.figure("TA_Explore_reward", figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode, save=True):
        pylab.figure("TA_Explore_reward")
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
#         if str(episode)[-2:] == "00":
#             pylab.plot(self.episodes_, self.scores_, 'b')
#             pylab.plot(self.episodes_, self.average_, 'r')
#             pylab.ylabel('Score', fontsize=18)
#             pylab.xlabel('Steps', fontsize=18)
#             try:
#                 pylab.grid(True)
#                 pylab.savefig(self.save_results+self.env_name+"_e"+str(self.EPISODES)+"_omega"+str(self.a0)+"_beta"+str(self.initial_beta)+"_E"+str(self.cut)+"_TA_Explore_reward.png")
#             except OSError:
#                 pass
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
        else:
            SAVING = ""
        return self.average_[-1], SAVING

    pylab.figure("R^T_reward", figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def o_PlotModel(self, o_score, o_episode, o_save=False):
        pylab.figure("R^T_reward")
        self.o_scores_.append(o_score)
        self.o_episodes_.append(o_episode)
        self.o_average_.append(sum(self.o_scores_[-50:]) / len(self.o_scores_[-50:]))
#         if str(o_episode)[-2:] == "00":
#             pylab.plot(self.o_episodes_, self.o_scores_, 'g')
#             pylab.plot(self.o_episodes_, self.o_average_, 'k')
#             pylab.ylabel('Score', fontsize=18)
#             pylab.xlabel('Steps', fontsize=18)
#             try:
#                 pylab.grid(True)
#                 pylab.savefig(self.save_results+self.env_name+"_e"+str(self.EPISODES)+"_omega"+str(self.a0)+"_beta"+str(self.initial_beta)+"_E"+str(self.cut)+"_R^T_reward.png")
#             except OSError:
#                 pass
        if self.o_average_[-1] >= self.max_average and o_save:
            self.max_average = self.o_average_[-1]
            self.save()
            o_SAVING = "SAVING"
        else:
            o_SAVING = ""
        return self.o_average_[-1] , o_SAVING

    def run_batch(self):
        results = []
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        o_score = 0
        while True:
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            o_rewards=[]
            for t in range(self.Training_batch):
                self.env.render()
                action, logp_t = self.act(state)
                next_state, constraint_reward, action_cost, done, _ = self.env.step(action[0])
                reward, o_reward =self.adaptive_reward(constraint_reward, action_cost, self.episode)
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                o_rewards.append(o_reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                o_score += o_reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode, save=True)
                    o_average, o_SAVING = self.o_PlotModel(o_score, self.episode, o_save=False)
                    if self.debug:
                        print("episode: {}/{}, o_score: {}, o_average: {:.2f} {}".format(self.episode, self.EPISODES, o_score, o_average, o_SAVING))
#                     print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
#                     self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
#                     self.writer.add_scalar(f'Workers:{1}/o_score_per_episode', o_score, self.episode)
#                     self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
#                     self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)
#                     self.writer.add_scalar(f'Workers:{1}/o_average_score',  o_average, self.episode)       
                    results.append([self.a0, self.initial_beta, self.cut, self.episode, max((self.cut-self.episode)*self.initial_beta/self.cut,0), o_average])         
                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    o_score, o_SAVING = 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])
            self.replay(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break
        self.env.close()
        data = pd.DataFrame(results)
        data.columns = ["omega", "beta0", "E", "episode" , "beta", "average R^T"]
        data.to_csv(self.save_results+'Four_Tank_e{}_omega{}_beta{}_E{}.csv'.format(self.EPISODES, self.a0, self.initial_beta, self.cut), index = False)
        print("The results saved in:"+Save_results+'Four_Tank_e{}_omega{}_beta{}_E{}.csv'.format(self.EPISODES, self.a0, self.initial_beta, self.cut))
        
    def test(self, test_episodes = 100):
        self.load()
        for e in range(test_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.Actor.predict(state)[0]
                state, constraint_reward, action_cost, done, _ = self.env.step(action)
                _, o_reward =self.adaptive_reward(constraint_reward, action_cost, e+1)
                state = np.reshape(state, [1, self.state_size[0]])
                score += o_reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e+1, test_episodes, score, average))
                    break
        self.env.close()
   
    def adaptive_reward(self,constraint_reward, action_cost, epi, adaptive_beta=True):
        if adaptive_beta:
            beta = round(max((self.cut-epi)*self.initial_beta/self.cut,0),8)
        else: 
            beta=self.initial_beta
        actual_total_reward = (self.a0*action_cost) + constraint_reward
        adaptive_reward = (beta*constraint_reward)+((1-beta)*actual_total_reward)
        return adaptive_reward, actual_total_reward        

if __name__ == "__main__":
    e = Control(constraint_based_termination= Constraint_Based_Termination, generate_obs_in_valid_range= Generate_Obs_in_Valid_Range)
    agent = PPOAgent(EPISODES = n_EPISODES, env = e, env_name = en, initial_beta=Initial_Beta , cut=CUT, a0=A0, save_results=Save_results, debug=Debug)
    agent.run_batch()
    #agent.test()


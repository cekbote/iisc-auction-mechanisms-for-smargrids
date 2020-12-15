'''
Sites and pages that are helpful:
DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
DDPG Code: https://github.com/germain-hug/Deep-RL-Keras/tree/master/DDPG
DDPG Code Updated: https://github.com/samhiatt/ddpg_agent/blob/master/ddpg_agent/agents/agent.py
Keras.Gradient: https://www.tensorflow.org/api_docs/python/tf/gradients
Keras.Function: https://www.tensorflow.org/api_docs/python/tf/keras/backend/function
Zip: https://www.geeksforgeeks.org/zip-in-python/
TensorFlow version used: less than 2.0.0 as then tf.gradients does work
'''

import tensorflow as tf
import collections
import random
import numpy as np
import math
import copy


class ReplayBuffer_A_R_P_Q:

    def __init__(self, a_r_dim, p_dim, q_dim, q_traded_dim, len):
        self.len = len
        self.states_a_r = collections.deque(maxlen=len)
        self.states_p_q = collections.deque(maxlen=len)
        self.actions_a_r = collections.deque(maxlen=len)
        self.actions_p = collections.deque(maxlen=len)
        self.actions_q = collections.deque(maxlen=len)
        self.actions_q_traded = collections.deque(maxlen=len)
        self.rewards = collections.deque(maxlen=len)
        self.via_matrix = collections.deque(maxlen=len)
        self.customized_status = collections.deque(maxlen=len)

        self.a_r_dim = a_r_dim
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.q_traded_dim = q_traded_dim

    def store_transition_state_a_r(self, state_a_r):
        self.states_a_r.append(state_a_r)

    def store_transition_state_p_q(self, state_p_q):
        self.states_p_q.append(state_p_q)

    def store_transition_action_a_r(self, action_a_r):
        self.actions_a_r.append(action_a_r)

    def store_transition_action_p_q(self, action_p, action_q, action_q_traded):
        self.actions_p.append(action_p)
        self.actions_q.append(action_q)
        self.actions_q_traded.append(action_q_traded)

    def store_transition_reward(self, reward):
        self.rewards.append(reward)

    def store_transition_via_matrix(self, via_matrix):
        self.via_matrix.append(via_matrix)

    def store_transition_customized_status(self, customized_status):
        self.customized_status.append(customized_status)

    def sample_buffer(self, batch_size):

        batch_reward = np.random.choice((int(self.len/4) - 1), int(math.floor(batch_size/4))) * 4 - 1
        batch_may_be_reward = list(np.random.choice((self.len - 1), (batch_size - int(math.floor(batch_size/4)))))
        batch_may_be_reward.extend(batch_reward)
        batch = np.asarray(batch_may_be_reward)


        current_states_a_r = np.array([self.states_a_r[i] for i in batch]).astype(np.float32)
        current_states_p_q = np.array([self.states_p_q[i] for i in batch]).astype(np.float32)
        actions_a_r = np.array([self.actions_a_r[i] for i in batch]).astype(np.float32).reshape(-1, self.a_r_dim)
        actions_p = np.array([self.actions_p[i] for i in batch]).astype(np.float32).reshape(-1, self.p_dim)
        actions_q = np.array([self.actions_q[i] for i in batch]).astype(np.float32).reshape(-1, self.q_dim)
        actions_q_traded = np.array([self.actions_q_traded[i] for i in batch]).astype(np.float32).reshape(-1, self.q_traded_dim)

        rewards = np.array([self.rewards[i] for i in batch]).astype(np.float32)

        next_states_a_r = np.array([self.states_a_r[i] for i in (batch + 1)]).astype(np.float32)
        next_states_p_q = np.array([self.states_p_q[i] for i in (batch + 1)]).astype(np.float32)
        next_actions_a_r = np.array([self.actions_a_r[i] for i in (batch + 1)]).astype(np.float32).reshape(-1, self.a_r_dim)
        next_actions_p = np.array([self.actions_p[i] for i in (batch + 1)]).astype(np.float32).reshape(-1, self.p_dim)
        next_actions_q = np.array([self.actions_q[i] for i in (batch + 1)]).astype(np.float32).reshape(-1, self.q_dim)
        next_actions_q_traded = np.array([self.actions_q_traded[i] for i in (batch + 1)]).astype(np.float32).reshape(-1, self.q_traded_dim)

        via_matrix = np.array([self.via_matrix[i] for i in batch]).astype(np.float32)
        customized_status = np.array([self.customized_status[i] for i in batch]).astype(np.float32)

        next_via_matrix = np.array([self.via_matrix[i] for i in (batch + 1)]).astype(np.float32)
        next_customized_status = np.array([self.customized_status[i] for i in (batch + 1)]).astype(np.float32)

        return current_states_a_r, current_states_p_q, actions_a_r, actions_p, actions_q, actions_q_traded, rewards, next_states_a_r, \
               next_states_p_q, next_actions_a_r, next_actions_p, next_actions_q, next_actions_q_traded, via_matrix, customized_status, \
               next_via_matrix, next_customized_status

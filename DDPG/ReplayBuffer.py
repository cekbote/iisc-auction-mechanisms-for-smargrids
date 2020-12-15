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


class ReplayBuffer:

    def __init__(self, action_dim, len):

        self.len = len
        self.states = collections.deque(maxlen = len)
        self.actions = collections.deque(maxlen = len)
        self.rewards = collections.deque(maxlen = len)
        self.action_dim = action_dim

    def store_transition_state(self, state):
        self.states.append(state)

    def store_transition_action(self, action):
        self.actions.append(action)

    def store_transition_reward(self, reward):
        self.rewards.append(reward)

    def sample_buffer(self, batch_size):

        batch_reward = np.random.choice((int(self.len/4) - 1), int(math.floor(batch_size/4))) * 4 - 1
        batch_may_be_reward = list(np.random.choice((self.len - 1), (batch_size - int(math.floor(batch_size/4)))))
        batch_may_be_reward.extend(batch_reward)
        batch = np.asarray(batch_may_be_reward)

        current_states = np.array([self.states[i] for i in batch]).astype(np.float32)
        actions = np.array([self.actions[i] for i in batch]).astype(np.float32).reshape(-1, self.action_dim)
        rewards = np.array([self.rewards[i] for i in batch]).astype(np.float32).reshape(-1, 1)
        next_states = np.array([self.states[i] for i in (batch + 1)]).astype(np.float32)
        next_actions = np.array([self.actions[i] for i in (batch + 1)]).astype(np.float32).reshape(-1, self.action_dim)

        return current_states, actions, rewards, next_states, next_actions
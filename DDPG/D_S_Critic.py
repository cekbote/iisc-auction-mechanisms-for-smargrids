'''
Sites and pages that are helpful:
DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
DDPG Code: https://github.com/germain-hug/Deep-RL-Keras/tree/master/DDPG
DDPG Code Updated: https://github.com/samhiatt/ddpg_agent/blob/master/ddpg_agent/agents/agent.py
Keras.Gradient: https://www.tensorflow.org/api_docs/python/tf/gradients
Keras.Function: https://www.tensorflow.org/api_docs/python/tf/keras/backend/function
Zip: https://www.geeksforgeeks.org/zip-in-python/
TensorFlow version used: less than 2.0.0 as then tf.gradients dosent work
'''

import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
import keras
import collections
import random
import numpy as np
import math
import copy


class D_S_Critic:

    def __init__(self, state_dim, act_dim):
        self.gamma = 0.8
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.model = self.network()
        self.network()

    def network(self):
        state_inp = layers.Input(shape=(self.state_dim,))

        action_inp = layers.Input(shape=(self.act_dim,))
        #
        x = layers.concatenate([state_inp, action_inp])
        x = layers.Dense(32, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        #
        x = layers.Dense(32, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        #
        x = layers.Dense(32, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        #
        output = layers.Dense(1, activation='linear', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)

        self.model = models.Model(inputs=[state_inp, action_inp], outputs=output)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[self.model.input[0], self.model.input[1], K.learning_phase()],
            outputs=K.gradients(self.model.output, [self.model.input[1]]))

    def gradients(self, state_inp, action_inp):
        return self.get_action_gradients([state_inp, action_inp, 0])

    def reward_value(self, state_inp, action_inp):
        return self.model.predict([state_inp, action_inp])

    def train(self, state_inp, action_inp, state_inp_next, action_inp_next, total_rewards):
        y = total_rewards / 400 + self.gamma * self.model.predict([state_inp_next, action_inp_next])
        loss = self.model.train_on_batch(x=[state_inp, action_inp], y=y)

        return loss

    def save(self, path):
        self.model.save_weights(path + '_D_S_Critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
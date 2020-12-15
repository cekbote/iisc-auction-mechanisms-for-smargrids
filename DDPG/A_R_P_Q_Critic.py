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
from keras import layers, models, optimizers
from keras import backend as K
import keras
import collections
import random
import numpy as np
import math
import copy


class A_R_P_Q_Critic:

    def __init__(self, a_r_inp_dim, p_q_inp_dim, a_r_act_dim, p_act_dim, q_act_dim, log, q_traded_act_dim=1):
        self.a_r_inp_dim = a_r_inp_dim
        self.p_q_inp_dim = p_q_inp_dim
        self.a_r_act_dim = a_r_act_dim
        self.p_act_dim = p_act_dim
        self.q_act_dim = q_act_dim
        self.q_traded_act_dim = q_traded_act_dim
        self.network()
        self.log = log

    def network(self):

        a_r_inp = layers.Input(shape=(self.a_r_inp_dim,))
        p_q_inp = layers.Input(shape=(self.p_q_inp_dim,))
        a_r_act_inp = layers.Input(shape=(self.a_r_act_dim,))
        p_act_inp = layers.Input(shape=(self.p_act_dim,))
        q_act_inp = layers.Input(shape=(self.q_act_dim,))
        q_traded_act_inp = layers.Input(shape=(self.q_traded_act_dim,))
        #
        x = layers.concatenate([a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp], axis=-1)
        x = layers.Dense(64, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        #
        x = layers.Dense(64, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        #
        x = layers.Dense(64, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        #
        output = layers.Dense(1, activation='linear',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)

        self.model = models.Model(inputs=[a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp], outputs=output)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        self.get_action_gradients = K.function( inputs= [self.model.input[0], self.model.input[1], self.model.input[2],
                                                       self.model.input[3], self.model.input[4], self.model.input[5], K.learning_phase()],
                                                      outputs = K.gradients(self.model.output,
                                                                                 [self.model.input[2],
                                                                                  self.model.input[3],
                                                                                  self.model.input[4],
                                                                                  self.model.input[5]]))

    def gradients(self, a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp):

        return self.get_action_gradients([a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp, 0])

    def reward_value(self, a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp):

        return self.model.predict([a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp])

    def train(self, a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp,
              a_r_inp_next, p_q_inp_next, a_r_act_inp_next, p_act_inp_next, q_act_inp_next, q_traded_act_inp_next,
              total_rewards):

        y = self.model.predict([a_r_inp_next, p_q_inp_next, a_r_act_inp_next, p_act_inp_next, q_act_inp_next, q_traded_act_inp_next])

        for i in range(len(total_rewards)):
            if total_rewards[i][0] != 0.0:
                y[i][0] = total_rewards[i][0] / 400

        loss = self.model.train_on_batch(x=[a_r_inp, p_q_inp, a_r_act_inp, p_act_inp, q_act_inp, q_traded_act_inp], y=y)

        return loss

    def save(self, path):

        self.model.save_weights(path + '_A_R_P_Q_Critic.h5')

    def load_weights(self, path):

        self.model.load_weights(path)
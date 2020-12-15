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



class P_Q_Actor:
    '''
        Actor Class for Accept Reject Network
    '''

    def __init__(self, inp_dim, lr, gaussian_std, out_dim, clip, grid_price, lower_price, constraint,  log):
        self.clip = clip
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.lr = lr
        self.gaussian_std = gaussian_std
        self.network()
        self.grid_price = grid_price
        self.lower_price = lower_price
        self.constraint = constraint
        self.log = log

    def network(self):
        inputs = layers.Input(shape=(self.inp_dim,))
        #
        x = layers.Dense(64, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        # x = layers.Dense(64, activation='relu',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        # x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        x = layers.Dense(64, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        x = layers.Dense(64, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        output_q_traded = layers.Dense(1, activation='linear',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        output_q_traded = layers.GaussianNoise(0.5)(output_q_traded)
        # output_q_traded = layers.Lambda(lambda x: K.clip(x, -self.clip, self.clip), (1,))(output_q_traded)
        output_q = layers.Dense(self.out_dim, activation='softmax',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        output_p = layers.Dense(self.out_dim, activation='linear',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        output_p = layers.GaussianNoise(0.5)(output_p)
        # output_p = layers.Lambda(lambda x: K.clip(x, -self.clip, self.clip), (self.out_dim,))(output_p)
        #
        self.model = models.Model(input=inputs, output=[output_p, output_q, output_q_traded])
        '''
            output_q_traded: is a portion of the total energy demanded or supplied by the D_S_Net
            output_q: is the softmax (distribution) of energy amongst buyers and sellers
            output_p: is the price at which the trade will occur
        '''

        p_act_grad = layers.Input(shape=(self.out_dim,))
        q_act_grad = layers.Input(shape=(self.out_dim,))
        q_traded_grad = layers.Input(shape=(1,))

        self.loss_1 = p_act_grad * output_p
        self.loss_2 = q_act_grad * output_q
        self.loss_3 = q_traded_grad * output_q_traded
        loss = 0.000001 * (K.mean(- p_act_grad * output_p) + K.mean(-q_act_grad * output_q) + K.mean(-q_traded_grad * output_q_traded))

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, p_act_grad, q_act_grad, q_traded_grad, K.learning_phase()],
            outputs=[loss],
            updates=updates_op)

    def summary(self):
        print(self.model.summary())

    def action(self, state, epsilon):

        shape = np.shape(state)

        if self.constraint == 0:
            # No constraint. The P_Q agent works intelligently.
            p, q, q_traded = self.model.predict(state)
            self.log.info('P: {} | Q: {} | Q_Trad: {}'.format(p,q,q_traded))

            check = np.random.uniform(0, 1)

            if check < epsilon:
                action_p = 2 * self.clip * np.random.random_sample((shape[0], self.out_dim)) - self.clip
                action_q = np.random.random_sample((shape[0], self.out_dim))
                action_q = action_q / (np.sum(action_q, axis= -1, keepdims=True))
                action_q_traded = 2 * self.clip * np.random.random_sample((shape[0], 1)) - self.clip
                return action_p, action_q, action_q_traded

            else:
                return np.clip(self.model.predict(state)[0], -self.clip, self.clip), self.model.predict(state)[1], np.clip(self.model.predict(state)[2], -self.clip, self.clip)

        else:

            if self.constraint == 1:
                # Grid buys at 17, and sells at 19. q_traded is always 1. q is distributed uniformly across all agents.

                # action_p qoutes to all agents.  Gets rectified in the main function.
                action_p = (np.matmul(np.diag(state[:, 2]) > 0, np.ones((shape[0], self.out_dim))) * 19 + np.matmul(np.diag(state[:, 2]) < 0, np.ones((shape[0], self.out_dim))) * 17 - (self.grid_price + self.lower_price)/(2 * self.clip)) /((self.grid_price - self.lower_price)/(2 * self.clip))
                # The energy is qouted equally amongst all agents. Will get rectified in the main function.
                action_q = np.ones((shape[0], self.out_dim)) / self.out_dim
                # The entire energy has to be traded
                action_q_traded = 2 * self.clip * np.ones((shape[0], 1)) - self.clip

                return action_p, action_q, action_q_traded

    def train(self, states, p_grads, q_grads,  q_traded_grads):
        # Grads will be supplied by the overall critic
        self.log.info('P_Grad: {} | Q_Grad: {} | Q_Trad_Grad: {}'.format(p_grads, q_grads, q_traded_grads))
        return self.train_fn([states, p_grads, q_grads, q_traded_grads, 1])

    def save(self, path):
        self.model.save_weights(path + '_P_Q_Actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

    def constraint_value(self):

        return self.constraint
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


class A_R_Actor:
    '''
        Actor Class for Accept Reject Network
    '''

    def __init__(self, inp_dim, lr, gaussian_std, out_dim, clip, constraint, log):

        self.clip = clip
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.lr = lr
        self.gaussian_std = gaussian_std
        self.network()
        self.constraint = constraint
        self.log = log

    def network(self):

        inputs = layers.Input(shape=(self.inp_dim,))
        #
        x = layers.Dense(32, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        # x = layers.Dense(64, activation='relu',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        # x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        x = layers.Dense(32, activation='tanh',
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                         bias_initializer='zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        outputs = layers.Dense(self.out_dim, activation='sigmoid',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        outputs = layers.GaussianNoise(0.5)(outputs)
        outputs = layers.Lambda(lambda x: K.clip(x, 0, 1), (self.out_dim,))(outputs)
        #
        self.model = models.Model(input=inputs, output=outputs)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.out_dim,))
        self.test = -action_gradients * outputs
        self.loss = K.mean(-action_gradients * outputs)
        loss = 0.000001 * K.mean(-action_gradients * outputs)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[loss],
            updates=updates_op)

    def summary(self):

        print(self.model.summary())

    def action(self, state, epsilon):

        shape = np.shape(state)

        if self.constraint == 0:

            self.log.info('A_R: {}'.format(self.model.predict(state)))
            check = np.random.uniform(0, 1)

            if check < epsilon:
                action = np.random.random_sample((shape[0], self.out_dim))
                return action

            else:
                return self.model.predict(state)

        else:

            if self.constraint == 1:
                # Buyer buys if price is less than 17. Seller sells if price is higher than 19

                action_a_r = np.matmul(np.diag(np.asarray(state[:, 2] < 0)), np.asarray(state[:, (2 + self.out_dim + 1): (2 + 2*self.out_dim + 1)]) <= 17) + np.matmul(np.diag(np.asarray(state[:, 2] > 0)), np.asarray(state[:, (2 + self.out_dim + 1): (2 + 2*self.out_dim + 1)]) >= 19)

                return action_a_r

    def train(self, states, action_gradients):

        #Grads will be supplied by the overall critic
        return self.train_fn([states, action_gradients, 1])

    def save(self, path):

        self.model.save_weights(path + '_A_R_Actor.h5')

    def load_weights(self, path):

        self.model.load_weights(path)

    def constraint_value(self):

        return self.constraint
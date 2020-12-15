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
import collections
import random
import numpy as np
import math
import keras
import copy

class D_S_Actor:

    def __init__(self, state_dim, lr, gaussian_std,  clip, logs, max_battery, constraint, act_dim=1):
        self.clip = clip
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gaussian_std = gaussian_std
        self.logs = logs
        self.max_battery = max_battery
        self.constraint = constraint
        self.network()
#         self.optimizer = self.optim()

    def network(self):
        states = layers.Input(shape= (self.state_dim,))
        #
        x = layers.Dense(32, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(states)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        # x = layers.Dense(64, activation='relu',
        #                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
        #                  bias_initializer='zeros')(x)
        # x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        x = layers.Dense(32, activation='tanh',kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(self.gaussian_std)(x)
        #
        actions = layers.Dense(self.act_dim, activation='linear', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros')(x)
        actions = layers.GaussianNoise(0.5)(actions)
        # actions = layers.Lambda(lambda x: K.clip(x, -self.clip, self.clip), (1, ))(actions)

        #
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.act_dim,))
        loss = 0.000001*K.mean(-action_gradients * actions)

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


        # No constraint. Intelligent behaviour.
        if (self.constraint == 0):

            self.logs.info('D_S: {}'.format(self.model.predict(state)))
            check = np.random.uniform(0, 1)
            shape = np.shape(state)
            if check < epsilon:
                action = 2*self.clip * np.random.random_sample((shape[0], self.act_dim)) - self.clip
                return action

            else:
                return np.clip(self.model.predict(state), -self.clip, self.clip)

        # Only cares about satisfying its demand for the current time step.
        else:
            # We have to output only one value. The batches output is only required for training, which we don't need.
            # state: [j % time_steps_per_day, agent[i]['Battery'], agent[i]['Renewable'],agent[i]['Demand'], grid_price]
            lower_bound = max(- self.max_battery, state[0, 1] + state[0, 2] - state[0,3] - self.max_battery)
            upper_bound = state[0, 1] + state[0, 2]

            if self.constraint == 1:
                # Where only the demand has to be satisfied. The rest is either sold or bought.

                return [[(state[0, 1] + state[0, 2] - state[0,3] - (upper_bound + lower_bound) / (2 * self.clip)) / ((upper_bound - lower_bound) / (2 * self.clip))]]

            elif self.constraint == 2:
                # Very conservative policy. Agent has to have atleast 1/4 of its max_battery before it can sell.

                return [[(state[0, 1] + state[0, 2] - state[0, 3] - self.max_battery/4 - (upper_bound + lower_bound) / (2 * self.clip)) / (
                            (upper_bound - lower_bound) / (2 * self.clip))]]

            elif self.constraint == 3:
                # Sell everything
                return [[(state[0, 1] + state[0, 2] - (upper_bound + lower_bound) / (2 * self.clip)) / (
                            (upper_bound - lower_bound) / (2 * self.clip))]]

    def train(self, state, action_gradients):
        self.logs.info('D_S_Grads: {}'.format(action_gradients))
        return self.train_fn([state, action_gradients, 1])

    def save(self, path):

        self.model.save_weights(path + '_D_S_Actor.h5')

    def load_weights(self, path):

        self.model.load_weights(path)

    def constraint_value(self):

        return self.constraint

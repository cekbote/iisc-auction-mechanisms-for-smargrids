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
import collections
import random
import numpy as np
import math
import copy

class ADL_Actor:

    def __init__(self, inp_dim, lr, gaussian_std, out_dim=1):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.lr = lr
        self.gaussian_std = gaussian_std
        self.model = self.network()
        self.optimizer = self.optim()

    def network(self):
        inputs = tf.keras.layers.Input(self.inp_dim)
        #
        x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GaussianNoise(self.gaussian_std)(x)
        #
        x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GaussianNoise(self.gaussian_std)(x)
        #
        outputs = tf.keras.layers.Dense(self.out_dim, activation='tanh')(x)
        #
        model = tf.keras.models.Model(input=inputs, output=outputs)

        return model

    def summary(self):

        print(self.model.summary())

    def action(self, state):

        return self.model.predict(state)

    def optim(self):

        action_gdts = tf.keras.backend.placeholder(shape = (None, self.out_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)  #Check the -action_gdts
        grads = zip(params_grad, self.model.trainable_weights)

        return tf.keras.backend.function([self.model.input, action_gdts], [tf.compat.v1.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def train(self, states, actions, grads):

        #Grads will be supplied by the overall critic
        self.optimizer([states, grads])

    def save(self, path):

        self.model.save_weights(path + '_ADL_Actor.h5')

    def load_weights(self, path):

        self.model.load_weights(path)


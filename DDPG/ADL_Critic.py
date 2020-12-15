'''
Sites and pages that are helpful:
DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
DDPG Code: https://github.com/germain-hug/Deep-RL-Keras/tree/master/DDPG
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

class ADL_Critic:

    def __init__(self, state_dim, act_dim):

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.model = self.network()
        self.action_grads = tf.keras.backend.function([self.model.input[0], self.model.input[1]],
                                                      tf.keras.backend.gradients(self.model.output, [self.model.input[1]]))

    def network(self):

        state_inp = tf.keras.layers.Input(self.state_dim)
        action_inp = tf.keras.layers.Input(self.act_dim)
        #
        x = tf.keras.layers.concatenate([state_inp, action_inp], axis = -1)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #
        output = tf.keras.layers.Dense(1, activation='linear')

        model = tf.keras.models.Model(inputs=[state_inp, action_inp], outputs=output)

        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='Adam')

        return model

    def gradients(self, state_inp, action_inp):

        return self.action_grads([state_inp, action_inp])

    def reward_value(self, state_inp, action_inp):

        return self.model.predict([state_inp, action_inp])

    def train(self, state_inp, action_inp, state_inp_next, action_inp_next, total_rewards):

        y = total_rewards + self.model.predict([state_inp_next, action_inp_next])
        loss = self.model.train_on_batch(x=[state_inp, action_inp], y=y)

        return loss

    def save(self, path):

        self.model.save_weights(path + '_ADL_Critic.h5')

    def load_weights(self, path):

        self.model.load_weights(path)



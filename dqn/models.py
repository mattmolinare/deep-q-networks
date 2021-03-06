# -*- coding: utf-8 -*-

"""DQN model architectures
"""

import keras
import keras.backend as K
from keras.layers import Dense, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf

__all__ = [
    'build_dqn_model',
    'build_dueling_dqn_model',
    'load_model'
]


def compile_model(model, learning_rate):
    model.compile(optimizer=Adam(lr=learning_rate), loss=tf.losses.huber_loss)


def build_dqn_model(env, fc_layers, learning_rate):

    model = Sequential()

    # fully connected layers
    model.add(Dense(fc_layers[0], input_shape=env.observation_space.shape,
                    activation='relu'))
    for units in fc_layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    compile_model(model, learning_rate)

    return model


def dueling_layer(x):

    v = x[:, :1]
    a = x[:, 1:]

    return v + a - K.mean(a, axis=1, keepdims=True)


def build_dueling_dqn_model(env, fc_layers, learning_rate):

    model = Sequential()

    # fully connected layers
    model.add(Dense(fc_layers[0], input_shape=env.observation_space.shape,
                    activation='relu'))
    for units in fc_layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(env.action_space.n + 1, activation='linear'))

    # dueling layer
    model.add(Lambda(dueling_layer, output_shape=(env.action_space.n,)))

    compile_model(model, learning_rate)

    return model


def load_model(model_file):
    custom_objects = {'huber_loss': tf.losses.huber_loss}
    return keras.models.load_model(model_file, custom_objects=custom_objects)

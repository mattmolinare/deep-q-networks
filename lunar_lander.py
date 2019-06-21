#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam


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

    inputs = Input(env.observation_space.shape)

    # fully connected layers
    x = Dense(fc_layers[0], activation='relu')(inputs)
    for units in fc_layers[1:]:
        x = Dense(units, activation='relu')(x)
    x = Dense(env.action_space.n + 1, activation='linear')(x)

    # dueling layer
    outputs = Lambda(dueling_layer, output_shape=env.action_space.n)(x)

    model = Model(inputs, outputs)

    compile_model(model, learning_rate)

    return model


def compile_model(model, learning_rate):
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    fc_layers = [512]
    learning_rate = 0.0003

    K.clear_session()

    model = build_dqn_model(env, fc_layers, learning_rate)

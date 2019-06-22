#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import keras
import keras.backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
import numpy as np
import os
import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'


def compute_epsilon(t, min_eps, max_eps, lmbda):
    return min_eps + (max_eps - min_eps) * np.exp(-lmbda * t)


def compile_model(model, learning_rate):
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')


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
    model.add(Lambda(dueling_layer, output_shape=env.action_space.n))

    compile_model(model, learning_rate)

    return model


def set_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    K.set_session(sess)


def get_memory_dtype(env):

    state_shape = env.observation_space.shape
    state_dtype = env.observation_space.dtype

    dtype = np.dtype([
        ('state', state_dtype, state_shape),
        ('reward', np.float),
        ('action', env.action_space.dtype),
        ('next_state', state_dtype, state_shape),
        ('done', np.bool)
    ])

    return dtype


class DQNAgent:

    def __init__(self, env, learning_rate, replay_memory_size, max_eps,
                 min_eps, lmbda, batch_size, gamma):

        self.env = env
        self.learning_rate = learning_rate
        self.replay_memory_size = replay_memory_size
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.gamma = gamma

        self.model = self.build_model()
        self.target_model = self.build_model()

        self._memory = np.recarray(self.replay_memory_size,
                                   dtype=get_memory_dtype(self.env))

        self.t = 0
        self.eps = self.max_eps

    @property
    def memory(self):
        return self._memory[:self.t]

    def build_model(self):
        return build_dqn_model(self.env, self.fc_layers, self.learning_rate)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def sample(self):

        random_idx = np.random.choice(self.memory.size,
                                      size=min(self.t, self.batch_size),
                                      replace=False)

        batch = self.memory[random_idx]

        return batch

    def observe(self, state, reward, action, next_state, done):

        transition = self.memory[self.t % self.replay_memory_size]

        transition.state = state
        transition.reward = reward
        transition.action = action
        transition.next_state = next_state
        transition.done = done

        # increment counter
        self.t += 1
        self.eps = compute_epsilon(self.t, self.min_eps, self.max_eps,
                                   self.lmbda)

    def act(self, state):
        return self.model.predict_on_batch(state[np.newaxis])[0].argmax()

    def replay(self):

        batch = self.sample()
        idx = np.arange(batch.size)

        q_values = self.model.predict_on_batch(batch.state)
        target_q_values = self.target_model.predict_on_batch(batch.state)

        if self.use_double_dqn:
            actions = q_values.argmax(axis=1)
            discounted = target_q_values[idx, actions]
        else:
            discounted = target_q_values.max(axis=1)

        targets = np.where(
            batch.done,
            batch.reward,
            batch.reward + self.gamma * discounted
        )

        y = q_values.copy()
        y[idx, batch.action] = targets

        self.model.train_on_batch(batch.state, y)


class DuelingDQNAgent(DQNAgent):

    def build_model(self):
        self._model = build_dueling_dqn_model(self.env, self.fc_layers,
                                              self.learning_rate)


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    fc_layers = [512]
    learning_rate = 0.0003
    replay_memory_size = 10000
    max_eps = 1.0

    K.clear_session()

    model = build_dqn_model(env, fc_layers, learning_rate)

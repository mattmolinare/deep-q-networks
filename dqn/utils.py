# -*- coding: utf-8 -*-

import gym
import keras.backend as K
import numpy as np
import random
import tensorflow as tf

__all__ = [
    'compute_epsilon',
    'get_env',
    'get_transition_dtype',
    'new_session',
    'set_seeds'
]


def get_env():
    return gym.make('LunarLander-v2')


def compute_epsilon(t, min_eps, max_eps, lmbda):
    return min_eps + (max_eps - min_eps) * np.exp(-lmbda * t)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def new_session():
    K.clear_session()
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    K.set_session(sess)


def get_transition_dtype(env):

    state_dtype = env.observation_space.dtype
    state_shape = env.observation_space.shape

    dtype = np.dtype([
        ('state', state_dtype, state_shape),
        ('reward', np.float),
        ('action', env.action_space.dtype),
        ('next_state', state_dtype, state_shape),
        ('done', np.bool)
    ])

    return dtype

# -*- coding: utf-8 -*-

import keras.backend as K
import numpy as np
import random
import tensorflow as tf

__all__ = [
    'compute_epsilon',
    'get_memory_dtype',
    'set_seeds'
]


def compute_epsilon(t, min_eps, max_eps, lmbda):
    return min_eps + (max_eps - min_eps) * np.exp(-lmbda * t)


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

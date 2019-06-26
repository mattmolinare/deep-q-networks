# -*- coding: utf-8 -*-

import glob
import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# local imports
from . import utils

__all__ = [
    'get_all_scores',
    'get_scores',
    'load_model',
    'plot_all_scores',
    'plot_scores',
    'test'
]


def load_model(model_file):
    custom_objects = {'huber_loss': tf.losses.huber_loss}
    return keras.models.load_model(model_file, custom_objects=custom_objects)


def test(env, model, num_iterations=100, max_steps=1000, video_dir=None,
         render=False, verbose=True):

    if video_dir is not None:
        env = gym.wrapper.Monitor(env, video_dir, force=True)

    scores = np.empty(num_iterations)

    for i in range(num_iterations):

        state = env.reset()
        score = 0.0

        for _ in range(max_steps):

            if render:
                env.render()

            action = model.predict_on_batch(state[np.newaxis])[0].argmax()
            next_state, reward, done, _ = env.step(action)

            state = next_state
            score += reward

            if done:
                break

        scores[i] = score

        if verbose:
            print('Iteration: %i, Score: %.2f' % (i + 1, score))

    return scores


def chdir(func):
    def wrapped(dirname, *args, **kwargs):
        cwd = os.path.abspath(os.getcwd())
        os.chdir(dirname)
        try:
            return func(dirname, *args, **kwargs)
        finally:
            os.chdir(cwd)
    return wrapped


@chdir
def get_scores(output_dir):

    scores = np.load('scores.npy')
    average_scores = np.load('average_scores.npy')

    return scores, average_scores


@chdir
def get_all_scores(parent_dir):

    params = utils.read_yaml('config.yaml')

    output_dirs = glob.glob('repeat*')
    repeats = len(output_dirs)

    all_scores = np.zeros((repeats, params['num_episodes']))
    all_average_scores = np.zeros_like(all_scores)

    for i, output_dir in enumerate(output_dirs):

        scores, average_scores = get_scores(output_dir)

        all_scores[i, :scores.size] = scores
        all_average_scores[i, :average_scores.size] = average_scores

    return all_scores, all_average_scores


def plot_scores(output_dir, **fig_kwargs):

    scores, average_scores = get_scores(output_dir)

    fig = plt.figure(**fig_kwargs)
    fig.clf()
    ax = fig.gca()
    ax.plot(scores, c='skyblue')
    ax.plot(average_scores, c='orange', lw=2)

    return fig, ax


def plot_all_scores(parent_dir, **fig_kwargs):

    _, all_average_scores = get_all_scores(parent_dir)

    cond = all_average_scores.all(axis=0)
    a = all_average_scores[cond]
    episodes, = np.where(cond)

    fig = plt.figure(**fig_kwargs)
    fig.clf()
    ax = fig.gca()
    ax.plot(np.median(a, axis=0), lw=2)

    return fig, ax

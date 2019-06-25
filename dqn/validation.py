# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
import numpy as np
import os

__all__ = [
    'plot_scores',
    'test'
]


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


def plot_scores(output_dir, **fig_kwargs):

    cwd = os.getcwd()
    os.chdir(output_dir)

    scores = np.load('scores.npy')
    average_scores = np.load('average_scores.npy')

    fig = plt.figure(**fig_kwargs)
    fig.clf()
    ax = fig.gca()
    ax.plot(scores, c='skyblue')
    ax.plot(average_scores, c='orange', lw=2)

    os.chdir(cwd)

    return fig

# -*- coding: utf-8 -*-

import glob
from matplotlib import pyplot
import numpy as np
import os

__all__ = [
    'get_scores'
]


def get_scores(parent_dir):
    """Return arrays of scores and average scores for each repeat run in a
    parent directory
    """
    output_dirs = sorted(glob.iglob(os.path.join(parent_dir, 'repeat*')))

    scores = []
    average_scores = []

    for output_dir in output_dirs:

        scores.append(np.load(os.path.join(output_dir, 'scores.npy')))
        average_scores.append(np.load(os.path.join(output_dir,
                                                   'average_scores.npy')))

    scores = _to_masked_array(scores)
    average_scores = _to_masked_array(average_scores)

    return scores, average_scores


def _to_masked_array(scores):

    repeats = len(scores)
    num_episodes = max(len(x) for x in scores)

    ma = np.ma.empty((num_episodes, repeats))
    for i, x in enumerate(scores):
        ma[:x.size, i] = x
        ma[x.size:, i] = np.ma.masked

    return ma


def plot_scores(parent_dir, fig=None, color=None, label=None, **fig_kwargs):

    _, average_scores = get_scores(parent_dir)

    num_episodes = average_scores.shape[0]
    episodes = np.arange(1, num_episodes + 1)

    median = np.median(average_scores, axis=1)
    min_ = average_scores.min(axis=1)
    max_ = average_scores.max(axis=1)

    if fig is None:
        fig = pyplot.figure(**fig_kwargs)
        fig.clf()

    ax = fig.gca()

    ax.plot(median, c=color, lw=2, label=label)
    ax.fill_between(episodes, min_, max_, color=color, lw=0,
                    alpha=0.3)

    ax.hlines(200, 1, num_episodes, ls='-.')
    ax.set_xlim(1, num_episodes)

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Average reward')

    return fig

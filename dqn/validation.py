# -*- coding: utf-8 -*-

import glob
from matplotlib import pyplot
import numpy as np
import os

__all__ = [
    'get_scores',
    'plot_scores'
]


def get_scores(parent_dir):
    """Return arrays of scores and average scores for each repeat run in a
    parent directory
    """
    output_dirs = sorted(glob.iglob(os.path.join(parent_dir, 'repeat*')))

    scores_list = []
    average_scores_list = []

    for output_dir in output_dirs:

        try:
            scores = np.load(os.path.join(output_dir, 'scores.npy'))
            average_scores = np.load(os.path.join(output_dir,
                                                  'average_scores.npy'))
        except FileNotFoundError:
            pass

        scores_list.append(scores)
        average_scores_list.append(average_scores)

    s = _to_masked_array(scores_list)
    s_ave = _to_masked_array(average_scores_list)

    return s, s_ave


def _to_masked_array(scores_list):

    repeats = len(scores_list)
    num_episodes = max(map(len, scores_list))

    s = np.ma.masked_all((num_episodes, repeats))
    for i, scores in enumerate(scores_list):
        s[:scores.size, i] = scores

    return s


def plot_scores(parent_dir, fig=None, color=None, label=None, **fig_kwargs):

    _, s_ave = get_scores(parent_dir)

    num_episodes = s_ave.shape[0]
    episodes = np.arange(1, num_episodes + 1)

    if fig is None:
        fig = pyplot.figure(**fig_kwargs)
        fig.clf()

    ax = fig.gca()

    ax.plot(np.median(s_ave, axis=1), color=color, linewidth=2, label=label)
    ax.fill_between(episodes, s_ave.min(axis=1), s_ave.max(axis=1),
                    color=color, linewidth=0, alpha=0.3)

    ax.hlines(200, 1, num_episodes, linestyle='-.')
    ax.set_xlim(1, num_episodes)

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Reward')

    return fig

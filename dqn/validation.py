# -*- coding: utf-8 -*-

"""Validation utilties
"""

import glob
import matplotlib
from matplotlib import pyplot
import numpy as np
import os

# local imports
from . import agents
from . import utils

__all__ = [
    'get_agent',
    'get_best_weights_file',
    'get_test_scores',
    'get_train_scores',
    'get_weights_file',
    'load_best_agent',
    'plot_train_scores'
]


def get_agent(params):

    env = utils.get_env()

    agent_type = getattr(agents, params['agent_type'])
    agent = agent_type(env, **params)

    return agent


def _detect_output_dirs(parent_dir):
    return sorted(glob.iglob(os.path.join(os.path.abspath(parent_dir),
                                          'repeat*')))


def get_weights_file(output_dir, max_score=200):
    """Get the name of the first weights file in an output directory where the
    average score is at least `max_score`
    """
    weights_file = None

    weights_dir = os.path.join(os.path.abspath(output_dir), 'weights')
    paths = os.listdir(weights_dir)
    paths.sort()
    for path in paths:
        score = int(path[18:23])
        if score > max_score:
            weights_file = os.path.join(weights_dir, path)

    return weights_file


def get_best_weights_file(parent_dir):
    """Get the name of the weights file in a parent directory that has the
    maximum average score across all repeat runs
    """
    output_dirs = _detect_output_dirs(parent_dir)

    best_score = -np.inf
    weights_file = None

    for output_dir in output_dirs:
        weights_dir = os.path.join(output_dir, 'weights')
        for path in os.listdir(weights_dir):
            score = int(path[18:23])
            if score > best_score:
                best_score = score
                weights_file = os.path.join(weights_dir, path)

    return weights_file


def load_best_agent(parent_dir):
    """Load the agent with the highest average score at any point during
    training
    """
    params = utils.read_yaml(os.path.join(parent_dir, 'config.yaml'))
    agent = get_agent(params)

    weights_file = get_best_weights_file(parent_dir)
    print('Loading weights from ' + weights_file)
    agent.load_weights(weights_file)

    return agent


def _to_masked_array(scores_list):

    repeats = len(scores_list)
    num_episodes = max(map(len, scores_list))

    s = np.ma.masked_all((num_episodes, repeats))
    for i, scores in enumerate(scores_list):
        s[:scores.size, i] = scores

    return s


def get_train_scores(parent_dir):
    """Return arrays of scores and average scores for each repeat run in a
    parent directory
    """
    output_dirs = _detect_output_dirs(parent_dir)

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


def get_test_scores(parent_dir):
    """Return array of test scores for each repeat run in a parent directory
    """
    output_dirs = _detect_output_dirs(parent_dir)

    test_scores_list = []
    for output_dir in output_dirs:
        try:
            test_scores_list.append(np.load(os.path.join(output_dir,
                                                         'test_scores.npy')))
        except FileNotFoundError:
            pass

    s_test = _to_masked_array(test_scores_list)

    return s_test


def plot_train_scores(parent_dir, fig=None, color=None, label=None,
                      **fig_kwargs):
    """Plot the mean average scores over a set of repeated runs
    """
    _, s_ave = get_train_scores(parent_dir)

    num_episodes = s_ave.shape[0]
    episodes = np.arange(1, num_episodes + 1)

    mean = s_ave.mean(axis=1)
    std = s_ave.std(axis=1)

    if fig is None:
        fig = pyplot.figure(**fig_kwargs)
        fig.clf()

    ax = fig.gca()

    line, = ax.plot(episodes, mean, color=color, linewidth=2, label=label)
    ax.fill_between(episodes, mean - std, mean + std, color=line.get_color(),
                    linewidth=0, alpha=0.25)

    ax.hlines(200, 0, num_episodes, linestyle='-.')
    ax.set_xlim(0, num_episodes)

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Average reward')

    return fig

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Kick off a grid search over learning rates, minimum epsilons, and epsilon
 decay rates.
"""

import argparse
import itertools
import os
import yaml

# local imports
import dqn
from train import train


def _pretty_float(x):
    return ('%.15f' % x).rstrip('0').rstrip('.').replace('.', 'p')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str
    )
    parser.add_argument(
        'results_dir',
        type=str,
        nargs='?',
        default='.'
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        nargs='*'
    )
    parser.add_argument(
        '--min_eps',
        type=float,
        nargs='*'
    )
    parser.add_argument(
        '--lmbda',
        type=float,
        nargs='*'
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        params = yaml.safe_load(stream)

    if args.learning_rate is None:
        args.learning_rate = [params['learning_rate']]

    if args.min_eps is None:
        args.min_eps = [params['min_eps']]

    if args.lmbda is None:
        args.lmbda = [params['lmbda']]

    agent_types = {
        'DQNAgent': 'vanilla_dqn',
        'TargetDQNAgent': 'target_dqn',
        'DDQN': 'ddqn',
        'DuelingDQN': 'dueling_dqn',
        'DuelingDDQN': 'dueling_ddqn'
    }

    agent_str = agent_types[params['agent_type']]

    it = itertools.product(args.learning_rate, args.min_eps, args.lmbda)
    for lr, eps, lmbda in it:

        p = params.copy()
        p['learning_rate'] = lr
        p['min_eps'] = eps
        p['lmbda'] = lmbda

        parent_folder = 'results_%s_lr%s_eps%s_lmbda%s' % \
            (agent_str, _pretty_float(lr), _pretty_float(eps),
             _pretty_float(lmbda))

        args.parent_dir = os.path.join(args.results_dir, parent_folder)

        try:
            train(args, p)
        except IOError:
            pass


if __name__ == '__main__':
    with dqn.Profiler(['time'], [10]):
        main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import yaml

# local imports
import dqn


def train(args, params):

    repeats = np.arange(1, args.repeats + 1)

    parent_dir = os.path.abspath(args.parent_dir)
    if os.path.isdir(parent_dir):
        p = dqn.read_yaml(os.path.join(parent_dir, 'config.yaml'))
        if params == p:
            print('Found matching config in ' + parent_dir)
            print('Appending to existing repeats')
            repeats += len(dqn.validation._detect_output_dirs(parent_dir))
        else:
            raise IOError('Parent directory already exists: ' + parent_dir)
    else:
        os.makedirs(parent_dir)
        dqn.write_yaml(os.path.join(parent_dir, 'config.yaml'), params)

    env = dqn.get_env()
    env.seed(params['seed'])

    agent_type = getattr(dqn.agents, params['agent_type'])

    for repeat in repeats:

        dqn.set_seeds(params['seed'])
        dqn.new_session()

        agent = agent_type(
            env,
            fc_layers=params['fc_layers'],
            learning_rate=params['learning_rate'],
            replay_memory_size=params['replay_memory_size'],
            min_eps=params['min_eps'],
            max_eps=params['max_eps'],
            lmbda=params['lmbda'],
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            target_update_interval=params['target_update_interval']
        )

        output_dir = os.path.join(parent_dir, 'repeat%03i' % repeat)

        agent.fit(
            num_episodes=params['num_episodes'],
            num_consecutive_episodes=params['num_consecutive_episodes'],
            max_steps=params['max_steps'],
            min_score=params['min_score'],
            max_average_score=params['max_average_score'],
            output_dir=output_dir,
            save_weights_interval=params['save_weights_interval'],
            render=False,
            verbose=params['verbose']
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str
    )
    parser.add_argument(
        'parent_dir',
        type=str,
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        params = yaml.safe_load(stream)

    train(args, params)


if __name__ == '__main__':
    with dqn.Profiler(['time'], [10]):
        main()

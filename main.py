#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gym
import os
from shutil import copyfile
import yaml

# local imports
import dqn

os.environ['PYTHONHASHSEED'] = '0'


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
        params = yaml.load(stream)

    parent_dir = os.path.abspath(args.parent_dir)
    if os.path.isdir(parent_dir):
        raise IOError('Parent directory already exists: ' + parent_dir)
    os.makedirs(parent_dir)

    copyfile(args.config_file, os.path.join(parent_dir, args.config_file))

    env = gym.make('LunarLander-v2')
    env.seed(params['seed'])

    agent_type = getattr(dqn.agents, params['agent_type'])

    for repeat in range(1, args.repeats + 1):

        dqn.set_seeds(params['seed'])

        agent = agent_type(env, params)

        output_dir = os.path.join(parent_dir, 'repeat%i' % repeat)

        agent.fit(
            params['num_episodes'],
            params['num_consecutive_episodes'],
            params['max_steps'],
            params['min_score'],
            params['max_average_score'],
            output_dir=output_dir,
            save_weights_interval=params['save_weights_interval'],
            render=False,
            verbose=params['verbose']
        )


if __name__ == '__main__':
    main()

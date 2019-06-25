#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from shutil import copyfile
import yaml

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ['PYTHONHASHSEED'] = '0'

import dqn


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

    parent_dir = os.path.abspath(args.parent_dir)
    if os.path.isdir(parent_dir):
        raise IOError('Parent directory already exists: ' + parent_dir)
    os.makedirs(parent_dir)

    copyfile(args.config_file, os.path.join(parent_dir, args.config_file))

    env = dqn.get_env()
    env.seed(params['seed'])

    agent_type = getattr(dqn.agents, params['agent_type'])

    for repeat in range(1, args.repeats + 1):

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

        output_dir = os.path.join(parent_dir, 'repeat%i' % repeat)

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


if __name__ == '__main__':
    with dqn.Profiler(['time'], [10]):
        main()

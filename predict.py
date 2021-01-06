#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

# local imports
import dqn


def predict(args):

    agent = dqn.load_best_agent(args.parent_dir)

    scores = np.empty((args.repeats, args.num_iterations))
    for i in range(args.repeats):
        scores[i] = agent.predict(num_iterations=args.num_iterations,
                                  render=args.render, verbose=True)

    np.save(args.output_file, scores)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'parent_dir',
        type=str
    )
    parser.add_argument(
        'output_file',
        type=str,
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100
    )
    parser.add_argument(
        '--render',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1
    )
    args = parser.parse_args()

    predict(args)


if __name__ == '__main__':
    with dqn.Profiler(['time'], [10]):
        main()

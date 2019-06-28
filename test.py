#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

# local imports
import dqn


def test(args):

    agent = dqn.load_agent(args.parent_dir)

    s = np.empty((args.repeats, args.num_iterations))
    for i in range(args.repeats):
        scores = agent.test(num_iterations=args.num_iterations,
                            render=args.render, verbose=True)
        print('Mean score: %.2f, Min score: %.2f, Max score: %.2f' %
              (scores.mean(), scores.min(), scores.max()))
        s[i] = scores

    np.save(args.output_file, s)



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

    test(args)


if __name__ == '__main__':
    with dqn.Profiler(['time'], [10]):
        main()

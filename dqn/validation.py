# -*- coding: utf-8 -*-

import glob
import numpy as np
import os

__all__ = [
    'get_scores'
]


def get_scores(parent_dir):
    """Return lists of scores and average scores for each repeat run in a
    parent directory
    """
    output_dirs = sorted(glob.iglob(os.path.join(parent_dir, 'repeat*')))

    scores = []
    average_scores = []

    for output_dir in output_dirs:

        scores.append(np.load(os.path.join(output_dir, 'scores.npy')))
        average_scores.append(np.load(os.path.join(output_dir,
                                                   'average_scores.npy')))

    return scores, average_scores

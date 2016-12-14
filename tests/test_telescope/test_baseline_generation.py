from __future__ import print_function

import numpy as np

from fastimgproto.telescope.base import generate_baselines_and_labels


def test_baseline_generation():
    """
    Imagine a 4-antenna array, with co-ords in (East,North,Up).
    """
    ant_posns = np.array([
        [0,0,0], #Origin
        [1,0,0], #1-East
        [2,0,0], #2-East
        [0,1,0], #1-North
    ],dtype=np.float_
    )
    ant_labels = [
        'origin',
        '1-east',
        '2-east',
        '1-north',
    ]

    baselines, baseline_labels = generate_baselines_and_labels(
        ant_posns, ant_labels)

    assert len(baselines) == 6 # 4-choose-2
    assert len(baselines) == len(baseline_labels)
    # print()
    # for idx in range(len(baselines)):
    #     print(idx, baseline_labels[idx], baselines[idx])

    assert baseline_labels == [
        'origin,1-east',
        'origin,2-east',
        'origin,1-north',
        '1-east,2-east',
        '1-east,1-north',
        '2-east,1-north',
    ]
    assert (baselines == np.array(
        [[1., 0., 0.],
         [2., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.], # From 1-east to 2-east
         [-1., 1., 0.], # From 1-east to 1-north
         [-2., 1., 0.]] # From 2-east to 1-north
    )).all()
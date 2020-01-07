""" Collection of functions that are used to generate the dynamic map.
"""
import numpy as np


def get_static_obstacles_v1(d0=3, d1=4):
    """ Get static obstacles in the map, as well as map boundaries.

    :param d0: map height
    :param d1: map width
    :return:
    """
    obstacles = \
        [(-1, el) for el in range(d1)] + \
        [(d0, el) for el in range(d1)] + \
        [(el, -1) for el in range(d0)] + \
        [(el, d1) for el in range(d0)] + \
        [(1, 1)]

    return obstacles


def get_static_obstacles_v2(d0=3, d1=8):
    """ Get static obstacles in the map, as well as map boundaries.

    :param d0: map height
    :param d1: map width
    :return:
    """
    obstacles = \
        [(-1, el) for el in range(d1)] + \
        [(d0, el) for el in range(d1)] + \
        [(el, -1) for el in range(d0)] + \
        [(el, d1) for el in range(d0)] + \
        [(1, 4), (1, 6)]

    return obstacles


def get_dyn_obstacles_v1(t):
    return []


def get_dyn_obstacles_v2(t):
    if t % 3 == 0:
        return [(0, 1), (1, 1), (1, 3), (2, 3)]
    elif t % 3 == 1:
        return [(1, 1), (2, 1), (0, 3), (2, 3)]
    else:
        return [(0, 1), (2, 1), (0, 3), (1, 3)]


def get_reward_points_v1(t_max):
    return [{
        'pos': (0, 3),
        't': np.arange(0, t_max+1),
        'reward': np.ones(shape=(t_max+1,))
    },
        {
            'pos': (1, 3),
            't': np.arange(0, t_max+1),
            'reward': -6 * np.ones(shape=(t_max+1,))
        }
    ]


def get_reward_points_v2(t_max):
    return [{
        'pos': (0, 7),
        't': np.arange(0, t_max+1),
        'reward': np.ones(shape=(t_max+1,))
    },
        {
            'pos': (1, 7),
            't': np.arange(0, t_max+1),
            'reward': -6 * np.ones(shape=(t_max+1,))
        }
    ]

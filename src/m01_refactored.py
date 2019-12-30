import itertools
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple
from scipy.optimize import fmin_l_bfgs_b


def kd(a, b):
    """ Kronecker delta
    """
    return 1 if a == b else 0


def get_reward(x: Tuple[int, int], t: int, reward_points: List[dict]):
    """ Get the reward at time t and position x
    """

    rew = sum([el['reward'][t] for el in reward_points if el['pos'] == x])

    return rew


def get_mask(x: Tuple[int, int],
             impenetrables: List[Tuple[int, int]],
             impenetrables_next: List[Tuple[int, int]]):
    """ Mask that takes into account the impenetrable points

    :param x: current position
    :param impenetrables: set of impenetrable points
    :param impenetrables_next: set of impenetrable points at the next time step
    """

    if x in impenetrables:
        mask = np.zeros(5).astype(bool)
    else:
        mask = np.array([
                sum(kd((x[0], x[1]), d) for d in impenetrables_next),  # stay
                sum(kd((x[0] - 1, x[1]), d) for d in impenetrables_next),  # move up
                sum(kd((x[0] + 1, x[1]), d) for d in impenetrables_next),  # move down
                sum(kd((x[0], x[1] + 1), d) for d in impenetrables_next),  # move right
                sum(kd((x[0], x[1] - 1), d) for d in impenetrables_next)  # move left
        ]) == 0

    return mask


def get_rho(lam: np.ndarray, mask: np.ndarray):
    """ Get the probabilities for [stay, up, down, right, left] movement
    """

    r = np.zeros(shape=(len(mask),))

    if any(mask):
        r_ = np.exp(lam[mask])
        r_ = r_ / r_.sum()
        r[mask] = r_

    return r


def get_p_xf_xi(xf, xi, lam, mask):
    rho = get_rho(lam, mask)

    p_xf_xi = \
        kd(xf, (xi[0], xi[1])) * rho[0] + \
        kd(xf, (xi[0] - 1, xi[1])) * rho[1] + \
        kd(xf, (xi[0] + 1, xi[1])) * rho[2] + \
        kd(xf, (xi[0], xi[1] + 1)) * rho[3] + \
        kd(xf, (xi[0], xi[1] - 1)) * rho[4]

    return p_xf_xi


def get_grad_p_xf_xi(xf, xi, lam, mask):
    rho = get_rho(lam, mask)
    p_xf_xi = get_p_xf_xi(xf, xi, lam, mask)

    grad_p_xf_xi = - rho * p_xf_xi

    grad_p_xf_xi += np.array([
        kd(xf, (xi[0], xi[1])) * rho[0],
        kd(xf, (xi[0] - 1, xi[1])) * rho[1],
        kd(xf, (xi[0] + 1, xi[1])) * rho[2],
        kd(xf, (xi[0], xi[1] + 1)) * rho[3],
        kd(xf, (xi[0], xi[1] - 1)) * rho[4]
    ])

    return grad_p_xf_xi


def neg_grad_v_st_new(lam, v_star, mask, xi, reward, gamma=1, gamma_tau=1, d0=3, d1=8):
    dv = sum([
        (gamma * v_star[xf] + gamma_tau * reward[xf]) * get_grad_p_xf_xi(xf, xi, lam, mask)
        for xf in itertools.product(range(d0), range(d1))
    ])

    return -dv


def neg_v_st_new(lam, v_star, mask, xi, reward, gamma=1, gamma_tau=1, d0=3, d1=8):
    v = sum([
        (gamma * v_star[xf] + gamma_tau * reward[xf]) * get_p_xf_xi(xf, xi, lam, mask)
        for xf in itertools.product(range(d0), range(d1))
    ])

    return -v


def get_path_v2(params: dict,
                impenetrable_points: List[List[Tuple[int, int]]],
                reward_points: List[dict]):
    d0 = params.get('d0', 3)
    d1 = params.get('d1', 4)
    t_max = params.get('t_max', 6)
    # gamma = params.get('gamma', 1)

    lams = np.zeros(shape=(t_max, d0, d1, 5))
    rhos = np.zeros(shape=(t_max, d0, d1, 5))
    v_stars = np.zeros(shape=(t_max, d0, d1))

    rewards = np.zeros(shape=(t_max, d0, d1))
    masks = np.ones(shape=(t_max, d0, d1, 5)).astype(bool)

    for tau in range(t_max):
        for x in itertools.product(range(d0), range(d1)):
            rewards[tau][x] = get_reward(x, tau, reward_points)  # rewards[tau] refers to reward at tau+1
            masks[tau][x] = get_mask(x, impenetrable_points[tau], impenetrable_points[tau + 1])

    for tau in np.arange(0, t_max)[::-1]:
        print(f'tau = {tau}')

        # gamma_tau = gamma ** tau

        for xi in itertools.product(range(d0), range(d1)):

            reward = rewards[tau]  # rewards[tau] refers to reward collected at tau+1
            mask = masks[tau][xi]
            v_star = np.zeros(shape=(d0, d1)) if tau == t_max - 1 else v_stars[tau + 1]

            lam, min_val, info = fmin_l_bfgs_b(func=neg_v_st_new,
                                               x0=np.array([0., 0., 0., 0., 0.]),
                                               fprime=neg_grad_v_st_new,
                                               args=[np.round(v_star, 4), mask, xi, reward],  # small trick
                                               factr=1e5,  # 1e7
                                               pgtol=1e-9,  # 1e-5
                                               maxfun=1000000,  # 15000
                                               maxiter=1000000)  # 15000

            lams[tau][xi] = lam
            rhos[tau][xi] = get_rho(lam, mask)

            v_stars[tau][xi] = - min_val

    result = {
        'v_stars': v_stars,
        'rhos': rhos,
        'rewards': rewards,
        'masks': masks
    }

    return result


def plot_rho(V, probabilities, impenetrable_points, d0, d1):
    fig = plt.figure(figsize=(1.5 * (d1 + 1), 1.5 * d0))

    ax = fig.add_subplot(1, 1, 1, xticks=np.arange(d1), yticks=np.arange(d0))
    plt.imshow(V, interpolation='none', aspect='auto', cmap='RdYlGn')
    plt.colorbar()
    arrows = [u'\u26ab', '\u25b2', '\u25bc', '\u25b6', '\u25c0']

    for (i, j), v in np.ndenumerate(np.around(V, 2)):
        label = ''
        if (i, j) not in impenetrable_points:
            if v < 0.01:
                label = "x\n "
            else:
                probability_vec = probabilities[(i, j)]
                direction = arrows[np.argmax(probability_vec)]
                prob = np.max(np.round(probability_vec, 2))
                label = f"{direction}\n{prob}"

        plt.gca().text(j, i, label, ha='center', va='center')

    plt.show()

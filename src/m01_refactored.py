import copy
import itertools
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple


def kd(a, b):
    """ Kronecker delta
    """
    return 1 if a == b else 0


def get_reward(x: Tuple[int, int], t: int, reward_points: List[dict]):
    """ Get the reward at time t and position x
    """

    rew = sum([el['reward'][t] for el in reward_points if el['pos'] == x])

    return rew


def get_mask(x: Tuple[int, int], tau: int, impenetrables: List[Tuple[int, int]]):
    """ Mask that takes into account the impenetrable points

    :param x: current position
    :param tau: time
    :param impenetrables: set of impenetrable points
    """

    if x in impenetrables:
        mask = np.zeros(5).astype(bool)
    else:
        mask = np.logical_not(
            np.array([
                sum(kd((x[0], x[1]), d) for d in impenetrables),  # stay
                sum(kd((x[0] - 1, x[1]), d) for d in impenetrables),  # move up
                sum(kd((x[0] + 1, x[1]), d) for d in impenetrables),  # move down
                sum(kd((x[0], x[1] + 1), d) for d in impenetrables),  # move right
                sum(kd((x[0], x[1] - 1), d) for d in impenetrables)  # move left
            ]))

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


def get_all_grad_p_xf_xi(xi, lam, mask, d0, d1):
    all_grad_xf = np.zeros(shape=(d0, d1, 5))

    for xf in itertools.product(range(d0), range(d1)):
        all_grad_xf[xf] = get_grad_p_xf_xi(xf, xi, lam, mask)

    return all_grad_xf.transpose((2, 0, 1))  # shape = (5, d0, d1)


def get_path(params: dict,
             impenetrable_points: List[Tuple[int, int]],
             reward_points: List[dict]):
    d0 = params.get('d0', 3)
    d1 = params.get('d1', 4)
    t_max = params.get('t_max', 6)
    alpha = params.get('alpha', 0.1)
    gamma = params.get('gamma', 1)
    max_iterations = params.get('max_iterations', 10000)

    lams = np.zeros(shape=(t_max, d0, d1, 5))
    rhos = np.zeros(shape=(t_max, d0, d1, 5))
    v_stars = np.zeros(shape=(t_max, d0, d1))

    rewards = np.zeros(shape=(t_max, d0, d1))
    masks = np.ones(shape=(t_max, d0, d1, 5)).astype(bool)

    for tau in range(t_max):
        for x in itertools.product(range(d0), range(d1)):
            rewards[tau][x] = get_reward(x, tau, reward_points)
            masks[tau][x] = get_mask(x, tau, impenetrable_points)

    for tau in np.arange(0, t_max)[::-1]:
        print(f'tau = {tau}')

        gamma_tau = gamma ** tau

        for xi in itertools.product(range(d0), range(d1)):

            reward = rewards[tau]
            mask = masks[tau][xi]
            lam = lams[tau][xi]
            rho, rho_prev = get_rho(lam, mask), None
            v_star = np.zeros(shape=(d0, d1)) if tau == t_max - 1 else v_stars[tau + 1]

            for it in range(max_iterations):

                # all_grad_xf = get_all_grad_p_xf_xi(xi, lam, mask, d0, d1)
                #
                # d_lam = ((gamma * v_star + gamma_tau * rewards) * all_grad_xf).sum(axis=(1, 2))

                d_lam = sum([
                    (gamma * v_star[xf] + gamma_tau * reward[xf]) * get_grad_p_xf_xi(xf, xi, lam, mask)
                    for xf in itertools.product(range(d0), range(d1))
                ])

                lam += alpha * d_lam

                rho_prev = rho
                rho = get_rho(lam, mask)

                if np.linalg.norm((rho - rho_prev)) < 10 ** -6:
                    break

            lams[tau][xi] = lam
            rhos[tau][xi] = rho

            v_stars[tau][xi] = sum([
                (gamma * v_star[xf] + gamma_tau * reward[xf]) * get_p_xf_xi(xf, xi, lam, mask)
                for xf in itertools.product(range(d0), range(d1))
            ])

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

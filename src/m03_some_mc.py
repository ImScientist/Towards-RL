import numpy as np
import copy
from matplotlib import pyplot as plt


def CD(a,b):
    '''
    CD = kronecker delta
    '''
    if a==b:
        return 1
    else:
        return 0



def Reward(x):
    '''
    x=(x[0], x[1])
    '''

    return CD((0,3), x) - 5*CD((1,3), x)



def Mask_x0_x1(x, D):
    '''
    D set of impenetrable points.
    mask legend:
    [up, down, right, left, stay]

    d=(d[0],d[1])
    '''

    if x in D:
        M = np.zeros(5)
    else:
        M = 1 - np.array([\
        sum( CD((x[0]-1,x[1]  ), d) for d in D), \
        sum( CD((x[0]+1,x[1]  ), d) for d in D), \
        sum( CD((x[0]  ,x[1]+1), d) for d in D), \
        sum( CD((x[0]  ,x[1]-1), d) for d in D), \
        sum( CD((x[0]  ,x[1]  ), d) for d in D)
        ])

    return M



def P_xf_xi(xf, xi, lam, M):
    '''
    xf = (xf_0, xf_1)
    xi = (xi_0, xi_1)
    lam = lam[xi[0], xi[1]] 5 dim lambda
    M - mask that takes into account which probabilities can be different from zero
    ep_noise - noise that gives a minimum non-zero probability for every action
    '''
    value = \
        CD(xf, (xi[0]-1,xi[1]  ) ) * M[0] * lam[0] + \
        CD(xf, (xi[0]+1,xi[1]  ) ) * M[1] * lam[1] + \
        CD(xf, (xi[0],  xi[1]+1) ) * M[2] * lam[2] + \
        CD(xf, (xi[0],  xi[1]-1) ) * M[3] * lam[3] + \
        CD(xf, (xi[0],  xi[1]  ) ) * M[4] * lam[4]

    return value



def Add_noise_to_lam_vector(lam, M, ep_noise):
    '''
    lam - lam[xi[0], xi[1]]
    ep_noise - noise that gives a minimum non-zero probability for every action

    lam.shape = (5,)
    M.shape = (5,)
    ep_noise5.shape = (5,)
    '''

    prob_vec = (lam + ep_noise)*M
    norm = prob_vec.sum()
    if norm == 0:
        norm = 1
    prob_vec = prob_vec/norm

    return prob_vec



def Sample_lam_G_from_lam_vector(lam):
    '''
    sample a lam_G vector of the form [0, ... 0, 1, 0, ... 0] from lam vector
    '''

    l_bounds = np.array([0] + list(lam[:-1]))
    h_bounds = copy.deepcopy(lam)

    l_bounds = np.cumsum(l_bounds)
    h_bounds = np.cumsum(h_bounds)

    v = np.random.rand()

    return (v >= l_bounds)*(v < h_bounds).astype(int)



def Generate_sequence_of_lam_G(s_0, lam_noise, length, destination):
    '''
    s_0 = (s_0[0], s_0[1]) = initial state in the map

    lam_noise - current pdf that is influenced by the noise
    lam_noise.shape = (dim0, dim1, 5)
    '''

    lam_G = np.zeros((length, 5))
    positions = [s_0]

    for it in range(length):

        x0, x1 = positions[-1]

        lam_G_new = Sample_lam_G_from_lam_vector(lam_noise[x0,x1])

        lam_G[it, :] = copy.deepcopy(lam_G_new)

        new_pos =   (x0-1, x1  ) * lam_G_new[0] + \
                    (x0+1, x1  ) * lam_G_new[1] + \
                    (x0,   x1+1) * lam_G_new[2] + \
                    (x0,   x1-1) * lam_G_new[3] + \
                    (x0,   x1  ) * lam_G_new[4]


        positions.append(new_pos)

        if new_pos == destination:
            break

    return lam_G, positions



def Calculate_Reward(gam, positions, R):
    '''
    Calculate Reward for a single trajectory defined by a sequence of positions
    positions = [p0, p1, p2, ...]
    pj = (pj[0], pj[1])
    '''

    reward = sum( gam**idx * R[x0,x1] for idx, (x0,x1) in enumerate(positions[1:]) )

    return reward


def Calculate_V(dimensions, gam, lam, M, R, n_iterations):

    dim0, dim1  = dimensions

    V_new = np.zeros((dim0, dim1))


    for it in range(n_iterations):

        V = copy.deepcopy(V_new)

        for x0 in range(dim0):
            for x1 in range(dim1):
                V_new[x0,x1] = sum( (gam*V[s0,s1]+R[s0,s1])*P_xf_xi((s0,s1), (x0,x1), lam[x0,x1], M[x0,x1]) for s0 in range(dim0) for s1 in range(dim1) )

    return V_new



def Get_pdf(dimensions, hyperparameters, M, R):


    dim0, dim1  = dimensions

    seed        = hyperparameters['seed']
    destination = hyperparameters['destination']
    gam         = hyperparameters['gam']
    beta        = hyperparameters['beta']
    ep_noise    = hyperparameters['ep_noise']
    iterations  = hyperparameters['iterations']

    reward_best = 0
    np.random.seed(seed)



    # initialize the pdf for transitions in the map
    Lam = np.ones((dim0, dim1, 5))
    for d0 in range(dim0):
        for d1 in range(dim1):
            Lam[d0,d1] = Add_noise_to_lam_vector(Lam[d0,d1], M[d0,d1], np.zeros(5))


    # add some artificial noise in the Lam
    Lam_noise = copy.deepcopy(Lam)
    for d0 in range(dim0):
        for d1 in range(dim1):
            Lam_noise[d0,d1] = Add_noise_to_lam_vector(Lam[d0,d1], M[d0,d1], ep_noise*np.ones(5))


    for it in range(iterations):
        lam_G_arr, positions = Generate_sequence_of_lam_G(s_0=(2,0), lam_noise=Lam_noise, length=10, destination=destination)
        reward_ = Calculate_Reward(gam, positions, R)

        if reward_ >= reward_best and reward_ > 0:
            for (x0,x1), lam_G in zip(positions, lam_G_arr[:len(positions)-1]):
                Lam[x0,x1] += beta*(lam_G-Lam[x0,x1])
                Lam_noise[x0,x1] = Add_noise_to_lam_vector(Lam[x0,x1], M[x0,x1], ep_noise*np.ones(5))

    return Lam, Lam_noise




def Plot_V(V, probabilities, impenetrable_points):

    plt.imshow(V, interpolation='none', aspect='auto', cmap='RdYlGn')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2])
    plt.colorbar()

    arrows = ['\u25b2', '\u25bc', '\u25b6', '\u25c0', u'\u26ab']

    for (i, j), v in np.ndenumerate(np.around(V, 2)):
        label = ''
        if (i,j) not in impenetrable_points:
            probability_vec = np.around(probabilities[i][j],2)
            label += "\n" + str(np.max(probability_vec))
            label =  arrows[np.argmax(probability_vec)] + label

        plt.gca().text(j, i, label, ha='center', va='center')

    plt.show()

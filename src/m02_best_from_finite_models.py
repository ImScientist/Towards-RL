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



def P_xf_xi(xf, xi, lam, M, ep_noise = 0):
    '''
    xf = (xf_0, xf_1)
    xi = (xi_0, xi_1)
    lam = lam[xi[0], xi[1]] 5 dim lambda
    M - mask that takes into account which probabilities can be different from zero
    ep_noise - noise that gives a minimum non-zero probability for every action
    '''
    value = \
        CD(xf, (xi[0]-1,xi[1]  ) ) * M[0] * (lam[0]**2 + ep_noise) + \
        CD(xf, (xi[0]+1,xi[1]  ) ) * M[1] * (lam[1]**2 + ep_noise) + \
        CD(xf, (xi[0],  xi[1]+1) ) * M[2] * (lam[2]**2 + ep_noise) + \
        CD(xf, (xi[0],  xi[1]-1) ) * M[3] * (lam[3]**2 + ep_noise) + \
        CD(xf, (xi[0],  xi[1]  ) ) * M[4] * (lam[4]**2 + ep_noise)

    return value



def Normalize_the_Lambdas(lam, M, ep_noise):
    '''
    The constraints of the world are not taken into account in the definitions of the
    lambda-vectors.
    '''

    norm = np.sqrt( ((lam**2)*M).sum() / (1-(ep_noise*M).sum()) )
    if norm==0:
        norm = 1
    lam = lam*M/norm
    return lam



def Get_P_V_el_Iterative(coordinates, dimensions, hyperparameters, V, R, M, Lam_candidates=np.ones((1,5))):
    '''
    Obtain the best policy from a finite set of policies.

    hyperparameters = {
        ep_noise,
        gam
    }

    V - Value function
    R - reward function
    M - mask with ones and zeros to ensure that some probabilities are 0.
    Lam_candidates.shape() = (n_candidates, 5)
    '''

    gam      = hyperparameters['gam']                # keep
    ep_noise = hyperparameters['ep_noise']           # keep

    x0, x1     = coordinates
    dim0, dim1 = dimensions

    V_new_x0_x1_arr = np.array([sum( (gam*V[s0,s1]+R[s0,s1])*P_xf_xi((s0,s1), (x0,x1), lam, M, ep_noise) for s0 in range(dim0) for s1 in range(dim1) )\
                                     for lam in Lam_candidates])

    V_new_x0_x1 = np.max(V_new_x0_x1_arr)
    lam_new = Lam_candidates[np.argmax(V_new_x0_x1_arr)]

    return lam_new, V_new_x0_x1



def Get_P_V_single_timestep(dimensions, hyperparameters, V, R, M, Lam_candidates):
    '''
    dim0, dim1 = dimensions

    Lam_candidates.shape = (dim0, dim1, n_candidates, 5)
    Lam_new.shape = (dim0, dim1, 5)

    V.shape = (dim0, dim1)
    R.shape = (dim0, dim1)
    V_new.shape = (dim0, dim1)
    M.shape = (dim0, dim1, 5)
    '''

    dim0, dim1 = dimensions

    V_new = copy.deepcopy(V)
    Lam_new = np.zeros((dim0, dim1, 5))

    for x0 in range(dim0):
        for x1 in range(dim1):

            Lam_new[x0,x1], V_new[x0,x1] = Get_P_V_el_Iterative(coordinates = (x0,x1), \
                                                                dimensions = dimensions, \
                                                                hyperparameters = hyperparameters, \
                                                                V = V, \
                                                                R = R, \
                                                                M = M[x0,x1], \
                                                                Lam_candidates = Lam_candidates[x0,x1])

    return Lam_new, V_new



def Plot_V(V, probabilities, impenetrable_points):

    plt.imshow(V, interpolation='none', aspect='auto', cmap='RdYlGn')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2])
    plt.colorbar()

    arrows = ['\u25b2', '\u25bc', '\u25b6', '\u25c0', u'\u26ab']

    for (i, j), v in np.ndenumerate(np.around(V, 2)):
        label = ''
        if (i,j) not in impenetrable_points:
            probability_vec = probabilities[i][j]
            label += "\n" + str(np.max(probability_vec))
            label =  arrows[np.argmax(probability_vec)] + label

        plt.gca().text(j, i, label, ha='center', va='center')

    plt.show()

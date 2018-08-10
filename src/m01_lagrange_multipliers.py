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



def Grad_P_xf_xi(xf, xi, lam, M, ep_noise = 0):
    '''
    xf = (xf_1, xf_2)
    xi = (xi_1, xi_2)
    lam = lam[xi[0], xi[1]] 6 dim lambda
    ep_noise - noise that gives a minimum non-zero probability for every action.
    M - mask that takes into account which probabilities can be different from zero.
    The addition of noise does not change the function.
    '''

    value = np.array([
        2*lam[0]*CD(xf, (xi[0]-1,xi[1]  )),
        2*lam[1]*CD(xf, (xi[0]+1,xi[1]  )),
        2*lam[2]*CD(xf, (xi[0],  xi[1]+1)),
        2*lam[3]*CD(xf, (xi[0],  xi[1]-1)),
        2*lam[4]*CD(xf, (xi[0],  xi[1]  ))
    ])

    return value*M



def BCond(lam, M, ep_noise = 0):
    '''
    Implement the boundary condition that has to be equal to zero.
    M - mask that takes into account which probabilities can be different from zero.
    ep_noise - noise that gives a minimum non-zero probability for every action.
    '''

    return (lam[0]**2 + lam[1]**2 + lam[2]**2 + lam[3]**2 + lam[4]**2 + M.sum()*ep_noise - 1)



def Grad_BCond_square(lam, M, ep_noise = 0):
    '''
    Gradient applied to the boundary condition;
    In our case
        the boundary condition is
            (lam[0]**2 + lam[1]**2 + lam[2]**2 + lam[3]**2 + lam[4]**2 - 1)

    lam = lam[xi[0], xi[1]] 5 dim lambda
    ep_noise - noise that gives a minimum non-zero probability for every action.
    '''

    bc = BCond(lam, M, ep_noise)

    value = np.array([
        -2*lam[0]*bc,
        -2*lam[1]*bc,
        -2*lam[2]*bc,
        -2*lam[3]*bc,
        -2*lam[4]*bc
    ])

    return value*M



def Check_condition(tv, g, ep_min, g_min, beta, beta_2, n_test_interval, beta_mult, inter_mult):
    '''
    tv = test_values
    g - deviation from the boundary condition
    g_min - max deviation from the boubdary condition
    beta - strength of the update
    beta_2 - sterength of the update of the BC
    n_test_interval - length of the intervil to which we apply the criteria for reaching a steady state
    beta_mult, inter_mult - multimplacators for the increase of beta (? or beta_2) and of n_test_interval in case of ...

    ep_min - max threshold for determination if a trajectory has reached a steady state



    return (reset_tv, new_beta, new_beta_2, new_length_of_test_interval, continue_iterations)
    '''

    ep = (tv.max(axis=1)-tv.min(axis=1)).sum()
    m = beta_mult
    mm = inter_mult

    # print('ep={0:.6f}, g={0:.6f}'.format(ep,g))


    if ep < ep_min and g>g_min:
        return np.zeros((tv.shape[0], tv.shape[1]*mm)), beta/m, beta_2*m, n_test_interval*mm, True

    elif ep < ep_min and g<=g_min:
        return 0*tv, beta, beta_2, n_test_interval, False

    else:
        return 0*tv, beta, beta_2, n_test_interval, True



def Get_P_V_el_Iterative(coordinates, dimensions, hyperparameters, V, R, M, lam_0=np.ones(5)):
    '''
    Obtain Lambda after several iterations.

    M - mask with ones and zeros to ensure that some probabilities are 0.
    V - Value function
    R - reward function
    lam_0 initial values of the transition probabilities of P

    hyperparameters = {
        ep_noise,
        gam,
        beta,
        beta_2,
        n_iterations = 100,
        n_test_interval=100
    }
    '''

    gam =    hyperparameters['gam']
    beta =   hyperparameters['beta']
    beta_2 = hyperparameters['beta_2']
    ep_min = hyperparameters['ep_min']
    g_min =  hyperparameters['g_min']

    ep_noise        = hyperparameters['ep_noise']
    n_iterations    = hyperparameters['n_iterations']
    n_test_interval = hyperparameters['n_test_interval']
    beta_mult       = hyperparameters['beta_mult']
    inter_mult      = hyperparameters['inter_mult']
    continue_iterations = True


    x0, x1 = coordinates
    dim0, dim1 = dimensions
    lam_new = lam_0*M
    test_values = np.zeros((5, n_test_interval))



    for it in range(n_iterations):

        lam = lam_new
        test_values[:, it % n_test_interval] = lam


        if it>0 and it % n_test_interval==0:

            test_values, beta, beta_2, n_test_interval, continue_iterations = \
                Check_condition(tv=test_values, g=BCond(lam, M, ep_noise), ep_min=ep_min, g_min=g_min, beta=beta, \
                                beta_2=beta_2, n_test_interval=n_test_interval, \
                                beta_mult=beta_mult, inter_mult=inter_mult)

            if continue_iterations == False:
                break


        T1 = sum( (gam*V[s0,s1]+R[s0,s1])*Grad_P_xf_xi((s0,s1), (x0,x1), lam, M, ep_noise) for s0 in range(dim0) for s1 in range(dim1) )
        T2 = Grad_BCond_square(lam, M, ep_noise)

        lam_new = (lam + beta*(T1+beta_2*T2))#*M


    V_new_x0_x1 = sum( (gam*V[s0,s1]+R[s0,s1])*P_xf_xi((s0,s1), (x0,x1), lam_new, M, ep_noise) for s0 in range(dim0) for s1 in range(dim1) )

    return lam_new, V_new_x0_x1



def Get_P_V_single_timestep(dimensions, hyperparameters, V, R, M, Lam):
    '''
    dim0, dim1 = dimensions

    Lam.shape = (dim0, dim1, 6)

    M.shape = (dim0, dim1, 6)

    V.shape = (dim0, dim1)
    '''

    dim0, dim1 = dimensions

    V_new = copy.deepcopy(V)
    Lam_new = copy.deepcopy(Lam)

    for x0 in range(dim0):
        for x1 in range(dim1):
            Lam_new[x0,x1], V_new[x0,x1] = Get_P_V_el_Iterative(coordinates = (x0,x1), \
                                                                dimensions = dimensions, \
                                                                hyperparameters = hyperparameters, \
                                                                V = V, \
                                                                R = R, \
                                                                M = M[x0,x1], \
                                                                lam_0=Lam[x0,x1] )

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

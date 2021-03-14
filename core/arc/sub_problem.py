'''
Autho: Cooper Simpson
Date: March 12, 2021

ARC sub-problem solvers.

Here we implement methods for solving the regularized cubic minimization
problem for ARC. A number of methods are implemented to be chosen at the users
discretion.
'''

import numpy as np
import numpy.linalg as la
from numpy.random import default_rng
from scipy.optimize import minimize

'''
Sub-problem objective function
'''
def objective(s, g, H, sigma):
    return np.dot(g,s) + 0.5*np.dot(s, H@s) + (sigma/3)*(la.norm(s))**3

def objective_grad(s, g, H, sigma):
    return g + H@z + sigma*la.norm(s)*s

'''
Dispatcher for solution methods
'''
def arcSub(method, args, kw={}):
    solver_map = {
        'lanczos' : lanczos(),
        'gradient descent' : gradientDescent()
    }

    return solver_map['method'](*args, **kw)

'''
Solvers: All have the following structure.
Input:
    g -> Gradient
    H -> Hessian
    sigma -> Regularization parameter
    maxitr -> Maximum iterations (optional)
    tol -> Tolerance (optional)
'''

'''
Generalized Lanczos method as described in the Non-Convex Newton and Inexact
Newton papers.

Run Lanczos iterations and then solve low-dim cubic problem with tri-diagonal
matrix.
'''
def lanczos(g, H, sigma, maxitr, tol):
    d = g.shape[0]
    K = min(d, maxitr)
    Q = np.zeros(d, K)

    q = g + defualt_rng().standard_normal(g.shape)
    q = q/la.norm(q)

    T = np.zeros(K+1, K+1)

    b=0, q0=0, tol=min(tol, tol*la.norm(g))

    for i in range(K+1):
        Q[:,i] = q
        v = H*q
        alpha = np.dot(q, v)
        T[i,i] = alpha

        #Orthogonalize
        M = Q[:,:i]
        r = v
        for j in range(i):
            r = r - np.dot(v, M[:,j])*M[:,j]

        beta = la.norm(r)
        T[i,i+1] = beta
        T[i+1,i] = beta

        if beta < tol:
            break

        q0 = q
        q = r/beta

    T = T[:i, :i]
    Q = Q[:, :i]

    if la.norm(T) < tol and la.norm(g) < np.spacing(1):
        return zeros(d,1), 0

    gt = Q.T@g

    #Optimization inception
    z0 = np.zeros(i)
    out = minimize(objective, z0, jac=objective_grad,
                    args=(gt, T, sigma), method='BFGS', tol=tol)
    z = out['x'], m = out['fun']

    return Q@z, m

'''
Gradient Descent

Key Word Input:
    lipg -> Lipschitz constant of gradient (g)
'''
def gradientDescent(g, H, sigma, maxitr, tol, *, lipg):
    step = 1/(20*lipg)

    g_norm = la.norm(g)

    if g_norm >= (lipg**2)/sigma:
        tmp = np.dot(g, H@g) / ((g_norm**2)*sigma)
        RC = -tmp + sqrt(tmp**2 + 4*g_norm/sigma)
        x = -RC*(g/g_norm)*2
        m = objective(x, g, H, sigma)

    else:
        x = np.zeros(g.shape)

        for i in range(maxitr):
            x1 = x - step*objective_grad(x, g, H, sigma)

            if la.norm(x1-x) < tol:
                break

            x = x1

        m = objective(x, g, H, sigma)

    return x, m

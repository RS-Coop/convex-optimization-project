'''
Author: Cooper Simpson
Date: March 7, 2021

Adaptive Regularization with Cubics (ARC) implementation.

This implementation is meant to be general, but we expect the possibility
of using approximate hessian information and only approximately solving the
sub-problem.

Algorithms for solving the sub-problem are given in sub_problem.py
'''

import numpy.linalg.norm as norm

from .sub_problem import arcSub

'''
Approximate Adaptive Regularization with Cubics

Input:
    F -> Objective function
    gradF -> Gradient of F (callable)
    hessF -> Hessian of F (callable)
    x0 -> Starting point
    eps_g -> Gradient tolerance
    eps_h -> Hessian tolerance
    sigma -> Initial regularization parameter (0,inf) (optional)
    eta -> Step success (0,1] (optional)
    gamma -> Regularization update (1,inf) (optional)
    maxitr -> Maximum number of iterations (optional)
    sub_tol -> Sub-problem tolerance (optional)
Output:
    x -> Minimizer
'''
def arc(F, gradF, hessF, x0, eps_g, eps_h, sigma, eta, gamma, maxitr, sub_tol):
    xt = x0

    #Set current gradient and hessian
    gt = gradF(xt)
    Ht = hessF(xt)

    for i in range(maxitr):
        #Check termination conditions
        #Bounds on norm of Gradient and smallest eigenvalue of hessian
        if norm(gt)<=eps_g:
            norm(gt) = np.zeros(gt.shape)
            if eig_min(Ht)>=-eps_h:
                return xt

        #Solve sub-problem
        #Get step (s) and objective value at s (m)
        s, m = aarcSub(gt, Ht, sigma)

        #Evaluate how good our step was
        p = (F(xt) - F(xt+s))/(-m)

        #If step was good update
        #NOTE: We can also consider a very good step and introduce more params
        if p>=eta:
            xt = xt + s
            sigma = sigma/gamma

            #Update gradient and hessian accordingly
            gt = gradF(xt)
            Ht = hessF(xt)

        #If step wasn't good decrease regularization (sigma)
        #NOTE: Need to consider a fail count here so we dont loop forever
        else:
            sigma = sigma*gamma

    print RuntimeWarning('WARNING! Maximum iterations exceeded \
                    without achieving tolerance.')

    return xt

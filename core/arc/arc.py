'''
Author: Cooper Simpson
Date: March 7, 2021

Adaptive Regularization with Cubics (ARC) implementation.

This implementation is meant to be general, but we expect the possibility
of using approximate hessian information and only approximately solving the
sub-problem.

Algorithms for solving the sub-problem are given in sub_problem.py
'''

from warnings import warn
import numpy.linalg as la

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
    sub_method -> Sub-problem solver (optional)
    sub_tol -> Sub-problem tolerance (optional)
    sub_maxitr -> Maximum sub-problem solver iterations (optional)
Output:
    x -> Minimizer

NOTE: Can add multiple eta and gamma parameters for adaptiveness
NOTE: Look into stopping condtions, code seems to be different than paper
algorithm in sources.
'''
def arc(F, gradF, hessF, x0, eps_g, eps_h, sigma=1, eta=0.8, gamma=2,
        maxitr=1000, sub_method='lanczos', sub_tol=1e-6, sub_maxitr=500):

    fails = 0 #Keep track of failed updates

    xt = x0

    #Set current objective value, gradient, and hessian
    ft = F(xt)
    gt = gradF(xt)
    Ht = hessF(xt)

    #Check termination conditions
    #Bounds on norm of Gradient and smallest eigenvalue of hessian
    if la.norm(gt)<=eps_g:
        gt = np.zeros(gt.shape)
        if la.eigvals(Ht).min()>=-eps_h:
            return xt

    for i in range(maxitr):
        #Solve sub-problem
        #Get step (s) and objective value at s (m)
        s, m = arcSub(sub_method, (gt, Ht, sigma))

        #Evaluate how good our step was
        p = (ft - F(xt+s))/(-m)

        #If step was good update
        if p>=eta:
            xt = xt + s
            sigma = sigma/gamma

            #Update gradient and hessian accordingly
            ft = F(xt)
            gt = gradF(xt)
            Ht = hessF(xt)

            #Check termination conditions
            #Bounds on norm of Gradient and smallest eigenvalue of hessian
            if la.norm(gt)<=eps_g:
                gt = np.zeros(gt.shape)
                if la.eigvals(Ht).min()>=-eps_h:
                    return xt

        #If step wasn't good decrease regularization (sigma)
        else:
            sigma = sigma*gamma

            fails += 1
            if fails == 3:
                print('Failure, exiting.')
                return xt

    warn('WARNING! Maximum iterations exceeded \
            without achieving tolerance.', RuntimeWarning)

    return xt

'''
Custom PyTorch optimizer class that uses second order ARC.
'''
import torch
from torch.optim import Optimizer
from scipy.optimize import minimize
import numpy as np

import time

'''
Helper functions
'''

''' '''
def cubic_np(s, g, H, sigma):
    func = g.T@s + 0.5*s.T@H@s + (sigma/3)*(np.linalg.norm(s)**3)
    grad = g + H@s + sigma*np.linalg.norm(s)*s

    return func, grad


'''Stochastic Adaptive Regularization with Cubics'''
class SARC(Optimizer):
    '''
    Initialize the optimizer.

    params -> Model parameters
    kw -> Optimizer keywords
    '''
    def __init__(self, params, sigma=1, eta_1=0.1, eta_2=0.9, gamma_1=2,
                    gamma_2=2, sub_problem_fails=1):
        defaults = dict(sigma=sigma, eta_1=eta_1, eta_2=eta_2, gamma_1=gamma_1,
                        gamma_2=gamma_2, sub_problem_fails=sub_problem_fails)

        super().__init__(params, defaults)

        self.hvp_time, self.hvp_calls = 0.0, 0
        self.sub_time, self.sub_calls = 0.0, 0
        self.scipy_time, self.scipy_calls = 0.0, 0
        self.p = []

    '''
    Convert list of gradients to a single tensor.
    '''
    def _convertGrads(self, grads):
        grads_vec = []

        for g in grads:
            if g is not None: #Not sure if this is a check I need to make
                grads_vec.append(g.view(-1).data)

        return torch.cat(grads_vec, 0)


    '''
    Hessian vector product

    v -> The vector in question
    grads -> Gradients of the model
    '''
    def _hvp(self, v, gradsH):
        '''The problem here is that we need v to be a list of tensors like gradsH,
        but we have it as a single tensor. The alternative is to change how we
        do math with v and leave it as a list.'''
        self.hvp_calls += 1
        tic = time.perf_counter()

        vec = []
        start = 0
        for g in gradsH:
            vec.append(v[start:start+torch.numel(g)].view(g.shape))
            start += torch.numel(g)

        hvs = torch.autograd.grad(gradsH, self.param_groups[0]['params'],
                                    grad_outputs=vec, only_inputs=True,
                                    retain_graph=True)

        g_vec = torch.cat([hv.view(-1).data for hv in hvs])

        self.hvp_time += time.perf_counter()-tic

        return g_vec


    '''Evaluate cubic sub-problem and its gradient'''
    def _cubic(self, s, g, gH, sigma):
        Hs = self._hvp(s, gH)
        s_norm = torch.norm(s, 2)

        func = torch.dot(g, s) + 0.5*torch.dot(s, Hs) + (sigma/3)*s_norm.pow(3)
        grad = g + Hs + (sigma*s_norm)*s

        return func, grad


    '''Solve the cubic sub-problem using generalized Lanczos'''
    #NOTE: There are a few places in here where I am not sure if I should be
    #worried about GPU stuff e.g. tensor creation.
    def _subProbSolve(self, g, gH, sigma, maxitr=10, tol=1e-6):
        d = torch.numel(g)
        K = min(d, maxitr)
        Q = torch.zeros((d, K))

        q = g/torch.norm(g)

        T = torch.zeros((K+1, K+1))

        g_norm = torch.norm(g, 2)
        tol = min(tol, tol*g_norm)

        for i in range(K):
            Q[:,i] = q
            v = self._hvp(q, gH)
            T[i,i] = torch.dot(q, v)

            #Orthogonalize
            r = v - torch.matmul(Q[:,:i], torch.matmul(torch.transpose(Q[:,:i], 0, 1), v))

            b = torch.norm(r, 2)
            T[i,i+1] = b
            T[i+1,i] = b

            if b < tol:
                q = torch.zeros_like(q)
                break

            q = r/b

        #Compute last diagonal element
        T[i, i] = torch.dot(q, self._hvp(q, gH))

        '''EDIT: Check and see if these should be i+1 here and not i, I think
        there may have been a problem with how matlab vs python slicing works.'''
        T = T[:i, :i]
        Q = Q[:, :i]

        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        #Optimization
        #NOTE: This is the last thing I need to figure out, i.e. how to put
        #this in pytorch form
        z0 = np.zeros(i)

        self.scipy_calls += 1
        tic = time.perf_counter()

        out = minimize(cubic_np, z0, args=(gt.numpy(), T.numpy(), sigma),
                        method='L-BFGS-B', tol=tol, jac=True)

        self.scipy_time += time.perf_counter() - tic

        z = out['x']
        m = out['fun']

        return torch.matmul(Q, torch.from_numpy(z).float()), m


    '''Update model parameters'''
    def _update(self, s):
        start = 0

        for p in self.param_groups[0]['params']:
            n = torch.numel(p)
            p.data.add_(s[start:start+n].view(p.shape))

            start += n


    '''
    Step forward and update parameters using optimizer.

    grads -> Full list of gradients
    gradsH -> Partial list of gradients for sub-sampling Hessian
    loss -> Current loss value
    loss_fn -> Recomputes loss without retaining gradient information
    closure -> Clears gradients and re-evaluates model
    '''
    def step(self, grads, gradsH, loss, loss_fn, closure=None):
        sigma = self.param_groups[0]['sigma']
        eta_1 = self.param_groups[0]['eta_1']
        eta_2 = self.param_groups[0]['eta_2']
        gamma_1 = self.param_groups[0]['gamma_1']
        gamma_2 = self.param_groups[0]['gamma_2']
        sub_problem_fails = self.param_groups[0]['sub_problem_fails']

        fails = 0

        grad_vec = self._convertGrads(grads)

        while fails < sub_problem_fails:
            self.sub_calls += 1
            tic = time.perf_counter()
            s, m = self._subProbSolve(grad_vec, gradsH, sigma)
            self.sub_time += time.perf_counter() - tic

            #Need to figure out how its gonna work with updating parameters
            self._update(s)
            p = (loss - loss_fn())/(-m)
            self.p.append(p)

            #Not a good update
            if p<eta_1:
                #Undo update
                self._update(-s)

            #Okay update
            if p>=eta_2:
                sigma = max(sigma/gamma_2, 1e-16)

            #Bad update
            else:
                sigma = gamma_1*sigma
                fails += 1

        #Update sigma
        self.param_groups[0]['sigma'] = sigma

    '''
    Print performance details
    '''
    def getInfo(self):
        return (self.hvp_time/self.hvp_calls,
                self.scipy_time/self.scipy_calls,
                self.sub_time/self.sub_calls,
                self.p)

'''
Custom PyTorch optimizer class that uses second order ARC.
'''
import torch
from torch.optim import Optimizer
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np

import time

'''
Helper functions
'''
'''Add two lists of tensors'''
def group_add(a, b):
    return [torch.sum(av+bv) for (av,bv) in zip(a,b)]

'''Dot product between two lists of tensors'''
def group_dot(a, b):
    s = 0.0
    for (av, bv) in zip(a, b):
        s += torch.dot(av, bv)
    return s

'''Normalize a list of tensors'''
def group_normalize(a):
    norm = group_dot(a,a)**0.5

    return [ai/norm for ai in a]

'''Convert a list of tensors to a single 1D tensor'''
def list2vec(l):
    vec = [li.view(-1).data for li in l]

    return torch.cat(vec, 0)


'''Convert a 1D tensor to a list of tensors using template for the shapes'''
def vec2list(v, template):
    l = []
    start = 0
    for ti in template:
        l.append(v[start:start+torch.numel(ti)].view(ti.shape))
        start += torch.numel(ti)

    return l

'''Stochastic Adaptive Regularization with Cubics'''
class SARC(Optimizer):
    '''
    Initialize the optimizer.

    params -> Model parameters
    kw -> Optimizer keywords
    '''
    def __init__(self, params, sigma=1, eta_1=0.1, eta_2=0.9, gamma_1=2,
                    gamma_2=2, sub_prob_fails=1, sub_prob_max_iters=10,
                    sub_prob_tol=1e-2, sub_prob_method='lanczos'):

        defaults = dict(sigma=sigma, eta_1=eta_1, eta_2=eta_2, gamma_1=gamma_1,
                        gamma_2=gamma_2, sub_prob_fails=sub_prob_fails,
                        sub_prob_max_iters=sub_prob_max_iters, sub_prob_tol=sub_prob_tol,
                        sub_prob_method=sub_prob_method)

        self.hvp_time, self.hvp_calls = 0.0, 0
        self.sub_time, self.sub_calls = 0.0, 0
        self.scipy_time, self.scipy_calls = 0.0, 0
        self.p = []

        if sub_prob_method == 'lanczos':
            self.subProbSolve = self._subProbSolve_lanczos
        elif sub_prob_method == 'eigsh':
            self.subProbSolve = self._subProbSolve_eigsh
        else:
            raise ValueError('Sub-problem solution method not supported.')

        super().__init__(params, defaults)


    '''
    Hessian vector product

    v -> The vector in question
    grads -> Gradients of the model
    '''
    def _hvp(self, v, gradsH):
        self.hvp_calls += 1
        tic = time.perf_counter()

        v_list = vec2list(v, gradsH)

        hvp = torch.autograd.grad(gradsH, self.param_groups[0]['params'],
                                    grad_outputs=v_list, only_inputs=True,
                                    retain_graph=True)

        self.hvp_time += time.perf_counter()-tic

        return list2vec(hvp)


    '''Evaluate cubic sub-problem and its gradient'''
    def _cubic(self, s, g, gH, sigma):
        Hs = self._hvp(s, gH)
        s_norm = torch.norm(s, 2)

        func = torch.dot(g, s) + 0.5*torch.dot(s, Hs) + (sigma/3)*s_norm.pow(3)
        grad = g + Hs + (sigma*s_norm)*s

        return func, grad


    '''Same as above but using Numpy functionality'''
    def _cubic_np(self, s, g, H, sigma):
        func = g.T@s + 0.5*s.T@H@s + (sigma/3)*(np.linalg.norm(s)**3)
        grad = g + H@s + sigma*np.linalg.norm(s)*s

        return func, grad


    '''Solve the cubic sub-problem using generalized Lanczos'''
    #NOTE: There are a few places in here where I am not sure if I should be
    #worried about GPU stuff e.g. tensor creation.
    def _subProbSolve_lanczos(self, grads, gradsH):
        #Setup some parameters
        sigma = self.param_groups[0]['sigma']
        maxitr = self.param_groups[0]['sub_prob_max_iters']
        tol = self.param_groups[0]['sub_prob_tol']

        #Convert the gradients to a vector
        g = list2vec(grads)

        d = torch.numel(g)
        K = min(d, maxitr)
        Q = torch.zeros((d, K))

        q = g/torch.norm(g)

        T = torch.zeros((K, K))

        g_norm = torch.norm(g, 2)
        tol = min(tol, tol*g_norm)

        for i in range(K-1):
            Q[:,i] = q
            v = self._hvp(q, gradsH)
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
        T[i+1, i+1] = torch.dot(q, self._hvp(q, gradsH))

        T = T[:i+2, :i+2]
        Q = Q[:, :i+2]

        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        #Optimization
        z0 = np.zeros(i+2)

        self.scipy_calls += 1
        tic = time.perf_counter()

        out = minimize(self._cubic_np, z0, args=(gt.numpy(), T.numpy(), sigma),
                        method='L-BFGS-B', tol=tol, jac=True)

        self.scipy_time += time.perf_counter() - tic

        z = out['x']
        m = out['fun']

        return torch.matmul(Q, torch.from_numpy(z).float()), m


    def _subProbSolve_eigsh(self, grads, gradsH):
        #Setup some parameters
        sigma = self.param_groups[0]['sigma']
        maxitr = self.param_groups[0]['sub_prob_max_iters']
        tol = self.param_groups[0]['sub_prob_tol']

        #Convert the gradients to a vector
        g = list2vec(grads)
        d = torch.numel(g)

        #Define scipy linear operator
        def hvp(x):
            x = torch.from_numpy(x)
            return self._hvp(x, gradsH).numpy()

        hvp_lin_op = LinearOperator((d,d), matvec=hvp)

        #Calculate eigenvectors
        _, evec = eigsh(hvp_lin_op, k=1, which='SA', maxiter=maxitr, tol=tol)
        e = torch.from_numpy(evec.ravel())

        Q = torch.zeros((d, 2))
        T = torch.zeros((2, 2))

        g_norm = torch.norm(g, 2)
        tol = min(tol, tol*g_norm)
        q = g/g_norm

        Q[:,0] = q
        T[0,0] = g_norm
        a = torch.dot(q, e)
        T[0,1] = a
        T[1,0] = 0

        #Orthogonalize
        r = e - T[0,1]*q
        b = torch.norm(r, 2)
        T[1,1] = b

        if b < tol:
            q = torch.zeros_like(q)
        else:
            q = r/b
        Q[:,1] = q

        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        #Optimization
        z0 = np.zeros(2)

        self.scipy_calls += 1
        tic = time.perf_counter()

        out = minimize(self._cubic_np, z0, args=(gt.numpy(), T.numpy(), sigma),
                        method='L-BFGS-B', tol=tol, jac=True)

        self.scipy_time += time.perf_counter() - tic

        z = out['x']
        m = out['fun']
        return torch.matmul(Q, torch.from_numpy(z).float()), m


    '''Update model parameters inplace'''
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
        sub_prob_fails = self.param_groups[0]['sub_prob_fails']

        fails = 0

        while fails < sub_prob_fails:
            self.sub_calls += 1
            tic = time.perf_counter()
            s, m = self.subProbSolve(grads, gradsH)
            self.sub_time += time.perf_counter() - tic

            #Need to figure out how its gonna work with updating parameters
            self._update(s)
            p = (loss - loss_fn())/(-m)
            self.p.append(p)

            #bad update
            if p<eta_1:
                #Undo update
                self._update(-s)
                sigma = gamma_1*sigma
                fails += 1

            #great update
            elif p>=eta_2:
                sigma = max(sigma/gamma_2, 1e-16)
                break

            else:
                break

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

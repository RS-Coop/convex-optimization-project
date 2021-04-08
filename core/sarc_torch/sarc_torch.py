'''
Custom PyTorch optimizer class that uses second order ARC.
'''
import torch
from torch.optim import Optimizer
from scipy.optimize import minimize
import numpy as np

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

    '''Get all parameter gradients in one tensor'''
    def _getGradients(self):
        grads = []
        grads_vec = []
        params = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.view(-1))
                    grads_vec.append(p.grad.view(-1).data)

        self.grads = grads
        self.params = params

        return torch.cat(grads_vec, 0) #.detach()?

    '''
    This is neccessary to avoid a memory leak as per PyTorch documentation
    on backward. It is probably better to change things so that we take the
    gradients as an argument in step and then use autograd.grad in training.
    '''
    def _reset(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

        self.grads = None
        self.params = None

    '''
    Hessian vector product

    v -> The vector in question
    grads -> Gradients of the model
    '''
    def _hvp(self, v):
        #Note that this assumes a single group which at this point is all there
        #should be.
        return torch.autograd.grad(self.grads, self.params,
                                    grad_outputs=v, only_inputs=True,
                                    retain_graph=True)[0].view(-1).data#.detach()?


    '''Evaluate cubic sub-problem and its gradient'''
    def _cubic(self, s, g, sigma):
        Hs = self._hvp(s)
        s_norm = torch.norm(s, 2)

        func = torch.dot(g, s) + 0.5*torch.dot(s, Hs) + (sigma/3)*s_norm.pow(3)
        grad = g + Hs + (sigma*s_norm)*s

        return func, grad

    '''Solve the cubic sub-problem using generalized Lanczos'''
    #NOTE: There are a few places in here where I am not sure if I should be
    #worried about GPU stuff e.g. tensor creation.
    def _subProbSolve(self, g, sigma, maxitr=500, tol=1e-6):
        d = torch.numel(g)
        K = min(d, maxitr)
        Q = torch.zeros((d, K))

        q = g/torch.norm(g)

        T = torch.zeros((K+1, K+1))

        g_norm = torch.norm(g, 2)
        tol = min(tol, tol*g_norm)

        for i in range(K):
            Q[:,i] = q
            v = self._hvp(q)
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
        T[i, i] = torch.dot(q, self._hvp(q))

        T = T[:i, :i]
        Q = Q[:, :i]

        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        #Optimization
        #NOTE: This is the last thing I need to figure out, i.e. how to put
        #this in pytorch form
        z0 = np.zeros(i)
        out = minimize(cubic_np, z0, args=(gt.numpy(), T.numpy(), sigma),
                        method='BFGS', tol=tol, jac=True)

        z = out['x']
        m = out['fun']

        return torch.matmul(Q, torch.from_numpy(z).float()), m

    '''Update model parameters'''
    def _update(self, s):
        start = 0

        for p in self.params:
            n = torch.numel(p)
            p.data.add_(s[start:n])

            start += n

    '''
    Step forward and update parameters using optimizer.

    loss -> Current loss value
    loss_fn -> Recomputes loss without retaining gradient information
    closure -> Clears gradients and re-evaluates model
    '''
    def step(self, loss, loss_fn, closure=None):
        sigma = self.param_groups[0]['sigma']
        eta_1 = self.param_groups[0]['eta_1']
        eta_2 = self.param_groups[0]['eta_2']
        gamma_1 = self.param_groups[0]['gamma_1']
        gamma_2 = self.param_groups[0]['gamma_2']
        sub_problem_fails = self.param_groups[0]['sub_problem_fails']

        fails = 0

        grad_vec = self._getGradients()

        while fails < sub_problem_fails:
            s, m = self._subProbSolve(grad_vec, sigma)

            #Need to figure out how its gonna work with updating parameters
            self._update(s)
            p = (loss - loss_fn())/(-m)

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

        self._reset()

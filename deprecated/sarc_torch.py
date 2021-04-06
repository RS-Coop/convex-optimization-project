'''
Custom PyTorch optimizer class that uses second order ARC.
'''
import torch
from torch.optim import Optimizer

'''Stochastic Adaptive Regularization with Cubics'''
class SARC(Optimizer):
    '''
    Initialize the optimizer.
    Inputs:
        params -> Model parameters
        kw -> Optimizer keywords
    '''
    def __init__(self, params, sigma=1, eta_1=0.1, eta_2=0.9, gamma_1=2, gamma_2=2):
        defaults = dict(sigma=sigma, eta_1=eta_1, eta_2=eta_2,
                        gamma_1=gamma_1, gamma_2=gamma_2)

        super().__init__(params, defaults)

    '''
    Get all parameter gradients in one tensor
    '''
    def _getGradients(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    '''
    Step forward and update parameters using optimizer.
    Inputs:
        closure -> Closure function that clears gradients and
                returns newly computed loss
    '''
    def step(self, closure=None):
        for group in self.param_groups:
            #Not worrying about group specific params
            for p in group['params']:
                #Check if these params are learnable
                if p.grad is None:
                    continue

                #Get gradient and hessian
                grad = p.grad.data #Maybe not this

                #Do one step of ARC

                '''
                What we need here is the gradient of parameters
                and also a hessian vector product function
                '''
                '''
                def closure(x):
                    outputs = model(x)
                    loss = bce(outputs, labels)
                    loss.backward(retain_graph=True)
                    return loss

                vH F.vhp(closure, inputs, torch.ones(2))

                then Hv = vH.t()
                '''

import argparse

from core.arc.arc import arc
from core.arc import hessian
from problems.svd import svd

from torch.optim import SGD
from core.sarc_torch.sarc_torch import SARC
from problems.spambase_torch import spambase
from problems.mnist_torch import mnist
from problems.cifar_torch import cifar

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run ARC and SARC tests')
    parser.add_argument('--test', type=str, default='spambase')
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sub_method', type=str, default='lanczos')
    parser.add_argument('--sub_tol', type=float, default=1e-2)
    parser.add_argument('--sub_max_iters', type=int, default=10)

    args = parser.parse_args()

    if args.order == 2:
        method = SARC
        kw = {
            'sub_prob_method':args.sub_method,
            'sub_prob_tol':args.sub_tol,
            'sub_prob_max_iters':args.sub_max_iters
        }
    elif args.order == 1:
        method = SGD

    if args.test == 'svd':
        minimum = svd(arc, {'eps_g':1e-3, 'eps_h':1e-3})
        print(f'Minimum objective value: {minimum}')

    elif args.test == 'spambase':
        spambase(dataroot='problems/spambase', optim_method=method,
                    epochs=args.epochs, order=args.order, batch_size=args.batch_size, **kw)

    elif args.test == 'mnist':
        mnist(data_dir='problems/mnist', optim_method=method,
                epochs=args.epochs, order=args.order, batch_size=args.batch_size, **kw)

    elif args.test == 'cifar':
        cifar(data_dir='problems/cifar10', optim_method=method,
                epochs=args.epochs, order=args.order, batch_size=args.batch_size, **kw)

    else: print('Specified test not supported.')

import argparse

from core.arc.arc import arc
from core.arc import hessian
from problems.svd import svd

from torch.optim import SGD
from core.sarc_torch.sarc_torch import SARC
from problems.spambase_torch import spambase
from problems.mnist_torch import mnist

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run ARC and SARC tests')
    parser.add_argument('--test', type=str, default='spambase')
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()

    if args.order == 2:
        method = SARC
    elif args.order == 1:
        method = SGD

    if args.test == 'svd':
        minimum = svd(arc, {'eps_g':1e-3, 'eps_h':1e-3})
        print(f'Minimum objective value: {minimum}')

    elif args.test == 'spambase':
        spambase(dataroot='problems/spambase', optim_method=method,
                    epochs=args.epochs, order=args.order)

    elif args.test == 'mnist':
        mnist(data_dir='problems/mnist', optim_method=method,
                epochs=args.epochs, order=args.order)

    else: print('Specified test not supported.')

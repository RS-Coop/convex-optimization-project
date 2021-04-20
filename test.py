import argparse
from core.problems import imageClassification as imc

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run SARC tests')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--optim_method', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sub_method', type=str, default='eigsh')
    parser.add_argument('--sub_tol', type=float, default=1e-2)
    parser.add_argument('--sub_max_iters', type=int, default=10)

    args = parser.parse_args()

    if args.optim_method == 'sgd':
        kw = {
            'lr': 0.01,
            'momentum': 0.9
        }

    elif args.optim_method == 'sarc':
        kw = {
            'sub_prob_method': args.sub_method,
            'sub_prob_tol': args.sub_tol,
            'sub_prob_max_iters': args.sub_max_iters
        }

    else: raise ValueError('Optimization method not supported.')

    output = imc(dataset=args.dataset, optim_method=args.optim_method, epochs=args.epochs,
                    batch_size=args.batch_size, validate=False, **kw)

    print(f'Test Accuracy for {args.dataset.upper()}: {output["test_acc"]*100:0.2f}')

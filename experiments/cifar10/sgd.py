from core.problems import imageClassification as imc
import json
import torch
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 eigsh test')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = parser.parse_args()

    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'cifar10'
    epochs = 60
    batch_size = 100

    kw = {
        'lr': args.lr,
        'momentum': args.momentum
    }

    output, model = imc(dataset=dataset, optim_method='sgd', epochs=epochs,
                            batch_size=batch_size, return_model=True, validate=True, **kw)

    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, f'data/sgd_{str(args.lr).replace(".","")}_{str(args.momentum).replace(".","")}.json')

    with open(data_dir, 'w') as file:
        json.dump(output, file)

    data_dir = os.path.join(path, f'data/sgd_{str(args.lr).replace(".","")}_{str(args.momentum).replace(".","")}.pt')

    torch.save(model.state_dict(), data_dir)

    print('Test finished.')

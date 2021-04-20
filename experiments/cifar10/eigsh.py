from core.problems import imageClassification as imc
import json
import torch
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 eigsh test')
    parser.add_argument('--tol', type=float, default=1e-2)
    args = parser.parse_args()

    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'cifar10'
    epochs = 60
    batch_size = 100

    kw = {
        'sub_prob_tol': args.tol,
        'sub_prob_max_iters': 50,
        'sub_prob_method': 'eigsh'
    }

    output, model = imc(dataset=dataset, optim_method='sarc', epochs=epochs,
                            batch_size=batch_size, return_model=True, validate=True, **kw)

    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, f'data/eigsh_{str(args.tol).replace('.','')}.json')

    with open(data_dir, 'w') as file:
        json.dump(output, file)

    data_dir = os.path.join(path, f'./data/eigsh_{str(args.tol).replace('.','')}.pt')

    torch.save(model.state_dict(), data_dir)

    print('Test finished.')

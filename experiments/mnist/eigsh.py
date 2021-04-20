from core.problems import imageClassification as imc
import json
import os

if __name__=='__main__':
    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'mnist'
    epochs = 10
    batch_size = 100

    kw = {
        'sub_prob_tol': 1e-2,
        'sub_prob_max_iters': 50,
        'sub_prob_method': 'eigsh'
    }

    output = imc(dataset=dataset, optim_method='sarc', epochs=epochs,
                    batch_size=batch_size, return_model=False, validate=True, **kw)

    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, 'data/eigsh.json')

    with open(data_dir, 'w') as file:
        json.dump(output, file)

    print('Test finished.')

from core.problems import imageClassification as imc
import json
import os

if __name__=='__main__':
    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'mnist'
    epochs = 10
    batch_size = 100

    kw = {'sub_prob_method': 'lanczos', 'sub_prob_tol': 1e-6}

    path = os.path.dirname(__file__)

    for max_iters in [2,10,20]:
        kw['sub_prob_max_iters'] = max_iters

        output = imc(dataset=dataset, optim_method='sarc', epochs=epochs,
                        batch_size=batch_size, return_model=False, validate=True, **kw)

        data_dir = os.path.join(path, f'data/lanczos_{max_iters}.json')

        with open(data_dir, 'w') as file:
            json.dump(output, file)

    print('Test finished.')

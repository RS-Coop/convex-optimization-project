import matplotlib.pyplot as plt
import json
import numpy as np
import os

if __name__=='__main__':
    path = os.path.dirname(__file__)
    legend = []

    '''eigsh'''
    data_dir = os.path.join(path, 'data/eigsh.json')

    with open(data_dir, 'r') as file:
        info = json.load(file)

    acc = np.array(info['validation_acc'])

    plt.plot(range(len(acc)), acc)
    legend.append('eigsh')

    '''Lanczos'''
    for d in [2,10,20]:
        data_dir = os.path.join(path, f'data/lanczos_{d}.json')
        with open(data_dir, 'r') as file:
            info = json.load(file)

        acc = np.array(info['validation_acc'])
        plt.plot(range(len(acc)), acc)
        legend.append(f'lancos {d} dim')

    '''SGD'''
    data_dir = os.path.join(path, 'data/sgd.json')

    with open(data_dir, 'r') as file:
        info = json.load(file)

    acc = np.array(info['validation_acc'])

    plt.plot(range(len(acc)), acc)
    legend.append('sgd')

    '''Plot'''
    plt.legend(legend)
    plt.show()

    # props = np.cumsum(np.array(info['props_list']))
    # loss = np.array(info['train_loss'])
    #
    # plt.plot(props, loss)
    # plt.show()

import matplotlib.pyplot as plt
import json
import numpy as np
import os

if __name__=='__main__':
    path = os.path.dirname(__file__)
    legend = []

    '''eigsh'''
    data_dir = os.path.join(path, 'data/eigsh_001_v2.json')

    with open(data_dir, 'r') as file:
        info = json.load(file)

    acc = np.array(info['validation_acc'])

    plt.plot(range(10, 10+len(acc)), acc)
    legend.append('eigsh')

    # '''SGD'''
    # data_dir = os.path.join(path, 'data/sgd_001_09.json')
    #
    # with open(data_dir, 'r') as file:
    #     info = json.load(file)
    #
    # acc = np.array(info['validation_acc'])
    #
    # plt.plot(range(len(acc)), acc)
    # legend.append('sgd')

    '''Plot'''
    plt.legend(legend)
    plt.show()

import matplotlib.pyplot as plt
import json
import numpy as np
import os

if __name__=='__main__':
    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, 'data/eigsh.json')

    with open(data_dir, 'r') as file:
        info = json.load(file)

    props = np.cumsum(np.array(info['props_list']))
    loss = np.array(info['train_loss'])

    plt.plot(props, loss)
    plt.show()

    acc = np.array(info['validation_acc'])

    plt.plot(range(len(acc)), acc)
    plt.show()

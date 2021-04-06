'''
Casting the logistic regression problem as an MLP neural
network using PyTorch.
'''
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

'''
Dataset class for spambase dataset
'''
class SpamBase(Dataset):
    def __init__(self, split, device, data_map=None):
        assert split in ['train', 'test']

        self.data = pd.read_csv(f'./spambase/spambase_{split}.data', header=None)

        self.device = torch.device(device)

        if data_map is None:
            self.data_map = lambda x: np.log(x+0.1)
        else:
            self.data_map = data_map

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.data.iloc[idx, 0:-1].apply(self.data_map),
                            dtype=torch.float, device=self.device)
        y = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float,
                            device=self.device).reshape(-1)

        return {'features':X, 'labels':y}

'''
Simple 1 layer dense network for logistic regression
'''
class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        outputs = self.forward(x)
        return torch.round(torch.sigmoid(outputs))

'''
Build the network, train, and validate potentially using our
custom second order optimizer.
'''
def spambase(dataset='spambase', optim_method=None, learn_rate=0.001,
                batch_size=100, epochs=20):

    #Check for GPU
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('Using GPU acceleration.')
    else:
        device = 'cpu'
        print('Using CPU only.')

    #Check dataset
    assert dataset in ['spambase']
    input_dim = 57

    #Load the model, setup loss and optimizer
    model = Model(input_dim)
    model.to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    if optim_method is not None:
        optimizer = optim_method()
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    #Setup dataloader
    train = SpamBase('train', device=device)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    #Now train the model
    print('Starting to train...')

    for epoch in range(epochs):
        for data in trainloader:
            optimizer.zero_grad()

            outputs = model(data['features'])
            loss = bce_loss(outputs, data['labels'])

            loss.backward()
            optimizer.step()

    print('Training finished.')

    #Validate
    test = SpamBase('test', device=device)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    correct, total = 0, 0
    for features, labels in testloader:
        predictions = model.predict(data['features'])

        total += data['labels'].size(0)
        correct += (predictions == data['labels']).sum()

    print(f'Test Accuracy: {(correct/total).item()}')

if __name__=='__main__':
    spambase()

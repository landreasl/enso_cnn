
# %%
from os import path
from ColorMNIST import ColorMNIST
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, M, N, width, height, output_size, input_channels = 3):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, M, 3, padding=1)
        
        self.conv2 = nn.Conv2d(M, M, 3, padding=1)
        
        self.conv3 = nn.Conv2d(M, M, 3, padding=1)

        self.fc1 = nn.Linear(int((width/4*height/4)*M), N)   
        self.fc2 = nn.Linear(N,output_size)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        x = x.view(-1, int((width/4*height/4))*M)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__=="__main__":
        
    BATCH_SIZE = 32
    PATH = 'mnist/'
    M = 50
    N = 30
    output_size=10


    # Load Data
    train_data = ColorMNIST('both', 'train', PATH, randomcolor=True)
    test_data = ColorMNIST('both', 'test', PATH, randomcolor=True)
    train_length = train_data.__len__()
    train_data, valid_data = torch.utils.data.random_split(train_data, (int(train_length*0.7), int(train_length*0.3))) # Split Training data into Training and Validation Data
    print(valid_data.__len__()+ train_data.__len__()+ test_data.__len__())
    valid_data.__len__(), train_data.__len__(), test_data.__len__()


    # %%
    #Define Train Loader and Test Loader
    train_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    images = images.numpy()

    width = images.shape[3]
    height = images.shape[2]
    # %%
    for i, l in enumerate(train_loader):
        data, label = l
        break
    #%%


    #Show 20 Samples from the batch
    fig =  plt.figure(figsize=(20,5))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].transpose([1,2,0]))
        ax.set_title(str(labels[idx].item()))

    plt.show()
    
    

    # %%
    #Train the Network
    model = Net()
    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr) 
    model.train()
    # %%
    n_epochs = 10

    epochs_train_saver = np.zeros(n_epochs)
    epochs_valid_saver = np.zeros(n_epochs)

    for epochs in range(n_epochs):
        
        train_loss = 0.0
        valid_loss = 0.0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.sampler)

        epochs_train_saver[epochs] = train_loss

        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)

        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)
            
            valid_loss = valid_loss/len(valid_loader.sampler)
            epochs_valid_saver[epochs] = valid_loss


        print(f'Epoch: {epochs +1}\t Training Loss: {train_loss}\t Validation Loss: {valid_loss}')

    plt.plot(epochs_train_saver)
    plt.plot(epochs_valid_saver)
    plt.title(f"learnrate = {lr} Epochs = {n_epochs}")
    plt.show()


        
    # %%


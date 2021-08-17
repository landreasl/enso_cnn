from os import path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch



class Net(nn.Module):
    """
    Args:
    ---
    
    M (int):
        Number of feature maps (30 or 50 in the paper).
    
    N (int):
        Number of neurons (30 or 50).
    
    width/height (int):
        Width and height of input layer. 
            
    output_size (int):
        Size of output (Nino3.4 index)
        
    input_channels (int):
        Number of input layers, 
        
    """

    def __init__(self, M, N, width, height, output_size = 1, input_channels = 6):
        super(Net, self).__init__()
        self.width = width
        self.height = height
        self.M = M

        self.conv1 = nn.Conv2d(input_channels, M, 3, padding=1)
        
        self.conv2 = nn.Conv2d(M, M, 3, padding=1)
        
        self.conv3 = nn.Conv2d(M, M, 3, padding=1)

        self.fc1 = nn.Linear(int((self.width/4*self.height/4)*M/2), N)   
        self.fc2 = nn.Linear(N, output_size)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, int((self.width/4*self.height/4)*self.M/2)) # last factor of 0.5 to match target size (why?)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x
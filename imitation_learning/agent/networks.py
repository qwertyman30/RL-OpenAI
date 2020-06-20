import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super().__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(1+history_length, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(8820, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)

        shape = x.shape
        x = x.view(-1, shape[1] * shape[2] * shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.log_softmax(x, dim=1)
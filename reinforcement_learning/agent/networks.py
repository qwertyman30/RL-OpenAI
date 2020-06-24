import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super().__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(1+history_length, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv3(x)
        x = F.relu(x)

        shape = x.shape
        x = x.view(-1, shape[1] * shape[2] * shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
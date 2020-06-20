import torch
import torch.nn as nn
from agent.networks import CNN

class BCAgent:
    def __init__(self, lr=1e-4, history_length=1):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.net = CNN(history_length=history_length, n_classes=4).cuda()
        self.history_length = history_length
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        X_batch = torch.Tensor(X_batch).cuda()
        X_batch = X_batch.view((-1, self.history_length+1, 96, 96))
        y_batch = torch.LongTensor(y_batch).cuda()
        # TODO: forward + backward + optimize
        #forward
        preds = self.predict(X_batch)

        #backward
        loss = self.criterion(preds, y_batch)
        self.optimizer.zero_grad()
        loss.backward()

        #optimize
        self.optimizer.step()
        return loss, preds

    def predict(self, X):
        # TODO: forward pass
        outputs = self.net(X)
        return outputs

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
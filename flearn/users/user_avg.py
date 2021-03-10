import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
from torch.optim.lr_scheduler import StepLR


# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, L, local_epochs):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, L, local_epochs)

        if model[1] == "linear":
            self.loss = nn.MSELoss()
        elif model[1] == "cnn":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        if model[1] == "cnn":
            layers = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.fc1, self.model.fc2]
            self.optimizer = torch.optim.SGD([{'params': layer.weight} for layer in layers] +
                                             [{'params': layer.bias, 'lr': 2 * self.learning_rate} for layer in layers],
                                             lr=self.learning_rate, weight_decay=L)
            self.scheduler = StepLR(self.optimizer, step_size=8, gamma=0.1)
            self.lr_drop_rate = 0.95
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.csi = None

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        return loss

    def get_params_norm(self):
        params = []
        for delta in self.delta_model:
            params.append(torch.flatten(delta.data))
        # return torch.linalg.norm(torch.cat(params), 2)
        return torch.max(torch.cat(params))

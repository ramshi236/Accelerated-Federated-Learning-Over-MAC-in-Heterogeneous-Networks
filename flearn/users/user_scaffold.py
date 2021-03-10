import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
import math
from torch.optim.lr_scheduler import StepLR


# Implementation for SCAFFOLD clients

class UserSCAFFOLD(User):
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
            weights = [{'params': layer.weight} for layer in layers]
            biases = [{'params': layer.bias, 'lr': 2 * self.learning_rate} for layer in layers]
            param_groups = [None] * (len(weights) + len(biases))
            param_groups[::2] = weights
            param_groups[1::2] = biases
            self.optimizer = SCAFFOLDOptimizer(param_groups, lr=self.learning_rate, weight_decay=L)
            # self.optimizer = SCAFFOLDOptimizer([{'params': layer.weight} for layer in layers] +
            #                                    [{'params': layer.bias, 'lr': 2 * self.learning_rate} for layer in
            #                                     layers],
            #                                    lr=self.learning_rate, weight_decay=L)

            self.scheduler = StepLR(self.optimizer, step_size=8, gamma=0.1)
            self.lr_drop_rate = 0.95
        else:
            self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate, weight_decay=L)

        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.delta_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
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
        grads = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.get_grads(grads)
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.server_controls, self.controls)
            if self.scheduler:
                self.scheduler.step()

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get client new controls
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        opt = 2
        if opt == 1:
            for new_control, grad in zip(new_controls, grads):
                new_control.data = grad.grad
        if opt == 2:
            for server_control, control, new_control, delta in zip(self.server_controls, self.controls, new_controls,
                                                                   self.delta_model):
                a = 1 / (math.ceil(self.train_samples / self.batch_size) * self.learning_rate)
                new_control.data = control.data - server_control.data - delta.data * a

        # get controls differences
        for control, new_control, delta in zip(self.controls, new_controls, self.delta_controls):
            delta.data = new_control.data - control.data
            control.data = new_control.data

        return loss

    def get_params_norm(self):
        params = []
        controls = []

        for delta in self.delta_model:
            params.append(torch.flatten(delta.data))

        for delta in self.delta_controls:
            controls.append(torch.flatten(delta.data))

        # return torch.linalg.norm(torch.cat(params), 2)
        return torch.max(torch.cat(params)), torch.max(torch.cat(controls))

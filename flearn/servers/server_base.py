import torch
import os
import h5py

import numpy as np
from utils.model_utils import Metrics
import copy
from scipy.stats import rayleigh


class Server:
    def __init__(self, dataset, algorithm, model, batch_size, learning_rate, L,
                 num_glob_iters, local_epochs, users_per_round, similarity, noise, times):

        # Set up the main attributes
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.users_per_round = users_per_round
        self.L = L
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []

        self.times = times
        self.similarity = similarity
        self.noise = noise
        self.communication_thresh = None
        self.param_norms = []
        self.control_norms = None

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, users_per_round):
        if users_per_round in [len(self.users), 0]:
            return self.users

        users_per_round = min(users_per_round, len(self.users))
        # fix the list of user consistent
        np.random.seed(round * (self.times + 1))
        return np.random.choice(self.users, users_per_round, replace=False)  # , p=pk)

    def select_transmitting_users(self):
        transmitting_users = []
        for user in self.users:
            user.csi = rayleigh.rvs()
            if user.csi >= self.communication_thresh:
                transmitting_users.append(user)
        return transmitting_users

    def save_results(self):
        """ Save loss, accuracy to h5 file"""
        file_name = "./results/" + self.dataset + "_" + self.algorithm
        file_name += "_" + str(self.similarity) + "s"
        if self.noise:
            file_name += '_noisy'
        file_name += "_" + str(self.times) + ".h5"
        if len(self.rs_glob_acc) != 0 & len(self.rs_train_acc) & len(self.rs_train_loss):
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_norms(self):
        """ Save norms, to h5 file"""
        file_name = "./results/" + self.dataset + "_" + self.algorithm + '_norms'
        file_name += "_" + str(self.similarity) + "s"
        if self.noise:
            file_name += '_noisy'
        file_name += "_" + str(self.times) + ".h5"

        if len(self.param_norms):
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('rs_param_norms', data=self.param_norms)
                if self.algorithm == 'SCAFFOLD':
                    hf.create_dataset('rs_control_norms', data=self.control_norms)

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.user_id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.user_id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)

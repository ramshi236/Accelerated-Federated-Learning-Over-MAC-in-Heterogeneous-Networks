#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from flearn.servers.server_avg import FedAvg
from flearn.servers.server_scaffold import SCAFFOLD
from flearn.trainmodel.models import *
from utils.plot_utils import *
import torch

torch.manual_seed(0)


def simulate(dataset, algorithm, model, batch_size, learning_rate, L, num_glob_iters, local_epochs, users_per_round,
             similarity, noise, times):
    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm: {algorithm}")
    print(f"Batch size              : {batch_size}")
    print(f"Learing rate            : {learning_rate}")
    print(f"Subset of users         : {users_per_round if users_per_round else 'all users'}")
    print(f"Number of local rounds  : {local_epochs}")
    print(f"Number of global rounds : {num_glob_iters}")
    print(f"Dataset                 : {dataset}")
    print(f"Data Similarity         : {similarity}")
    print(f"Local Model             : {model}")
    print("=" * 80)

    for i in range(times):
        print("---------------Running time:------------", i)

        # Generate model
        if model == "mclr":  # for Mnist and Femnist datasets
            model = MclrLogistic(output_dim=47), model

        if model == "linear":  # For Linear dataset
            model = LinearRegression(40, 1), model

        if model == "dnn":  # for Mnist and Femnist datasets
            model = DNN(), model

        if model == "cnn":  # for Cifar-10 dataset
            model = CifarNet(), model

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(dataset, algorithm, model, batch_size, learning_rate, L, num_glob_iters, local_epochs,
                            users_per_round, similarity, noise, i)

        if algorithm == "SCAFFOLD":
            server = SCAFFOLD(dataset, algorithm, model, batch_size, learning_rate, L, num_glob_iters, local_epochs,
                              users_per_round, similarity, noise, i)
        server.train()
        server.test()

    # Average data
    average_data(num_glob_iters=num_glob_iters, algorithm=algorithm, dataset=dataset, similarity=similarity,
                 noise=noise, times=times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                        choices=["CIFAR-10", "Mnist", "Linear_synthetic", "Logistic_synthetic"])
    parser.add_argument("--similarity", type=int, default=1)
    parser.add_argument("--model", type=str, default="CIFAR-10", choices=["linear", "mclr", "dnn", "CIFAR-10"])
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=0.008, help="Local learning rate")
    parser.add_argument("--hyper_learning_rate", type=float, default=0.02, help=" Learning rate of FEDL")
    parser.add_argument("--L", type=int, default=0.004, help="Regularization term")
    parser.add_argument("--num_glob_iters", type=int, default=250)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--algorithm", type=str, default="FedAvg", choices=["FEDL", "FedAvg", "SCAFFOLD"])
    parser.add_argument("--clients_per_round", type=int, default=0, help="Number of Users per round")
    parser.add_argument("--rho", type=float, default=0, help="Condition Number")
    parser.add_argument("--noise", type=float, default=False, help="Applies noisy channel effect")
    parser.add_argument("--pre-coding", type=float, default=False, help="Applies pre-coding")
    parser.add_argument("--times", type=int, default=1, help="Running time")
    args = parser.parse_args()

    simulate(dataset=args.dataset, algorithm=args.algorithm, model=args.model,
             batch_size=args.batch_size, learning_rate=args.learning_rate,
             hyper_learning_rate=args.hyper_learning_rate, L=args.L, num_glob_iters=args.num_glob_iters,
             local_epochs=args.local_epochs, users_per_round=args.clients_per_round,
             rho=args.rho, similarity=args.similarity, noise=args.noise, times=args.times)

#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from flearn.servers.serveravg import FedAvg
from flearn.servers.serverfedl import FEDL
from flearn.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

algorithms_list = ["FEDL","FedAvg","FEDL","FedAvg","FEDL","FedAvg","FEDL","FEDL"]
rho = [0,0,0,0,0,0,0,0,0,0,0,0,0]
lamb_value = [0, 0, 0, 0, 0, 0,0, 0, 0, 0]
learning_rate = [0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003]
hyper_learning_rate = [0.2,0,0.2,0,0.2,0,2.0,4.0]
local_ep = [20, 20, 20, 20, 20, 20, 20, 20]
batch_size = [20,20,40,40,0,0,0,0]
DATA_SET = "Mnist"
number_users = 10
plot_summary_mnist(num_users=number_users, loc_ep1=local_ep, Numb_Glob_Iters=800, lamb=lamb_value,learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms_list=algorithms_list, batch_size=batch_size, rho = rho, dataset=DATA_SET)
print("-- FINISH -- :",)

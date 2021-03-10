#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_utils import *
import torch
torch.manual_seed(0)

algorithms_list = ["FEDL","FEDL","FEDL","FEDL","FEDL","FEDL","FEDL","FEDL","FEDL","FEDL","FEDL","FEDL"]
rho = [1.4, 1.4, 1.4, 1.4,  2 ,2 , 2, 2, 5, 5, 5, 5]
lamb_value = [0, 0, 0, 0,  0, 0, 0, 0 , 0, 0, 0 ,0]
learning_rate = [0.04,0.04,0.04,0.04, 0.04,0.04,0.04,0.04, 0.04,0.04,0.04,0.04]
hyper_learning_rate = [0.01,0.03,0.05,0.07, 0.01,0.03,0.05,0.07, 0.01,0.03,0.05,0.07]
local_ep = [20, 20, 20, 20,  20, 20, 20, 20,  20, 20, 20, 20]
batch_size = [0,0,0,0 ,0,0,0,0, 0,0,0,0]
DATA_SET = "Linear_synthetic"
number_users = 100

plot_summary_linear(num_users=number_users, loc_ep1=local_ep, Numb_Glob_Iters=200, lamb=lamb_value, learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms_list=algorithms_list, batch_size=batch_size, rho = rho, dataset=DATA_SET)
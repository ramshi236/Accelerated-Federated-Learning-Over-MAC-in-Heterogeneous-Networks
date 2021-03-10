#!/usr/bin/env python
import numpy as np
import json
import random
import os
np.random.seed(0)

NUM_USER = 100
Kappa = 1.4
Dim = 40 
Noise = 0.05

def generate_x(n_samples = 100, dim= 40, kappa= 10):
    '''Helper function to generate data''' 

    powers = - np.log(kappa) / np.log(dim) / 2

    S = np.power(np.arange(dim)+1, powers)
    X = np.random.randn(n_samples, dim) # Random standard Gaussian data
    X *= S
    covarient_matrix = np.cov(X)
    print("Covarient matrix:",covarient_matrix)                            # Conditioning
    print("np.diag(S)", np.diag(S))
    return X, 1, 1/kappa, np.diag(S)

def generate_linear_data(num_users=100, kappa=10, dim=40, noise_ratio=0.05):

    '''Helper function to generate data'''
    # generate power S
    powers = - np.log(kappa) / np.log(dim) / 2
    DIM = np.arange(dim)

    # Covariance matrix for X
    S = np.power(DIM+1, powers)

    # Creat list data for all users 
    X_split = [[] for _ in range(num_users)]  # X for each user
    y_split = [[] for _ in range(num_users)]  # y for each user
    samples_per_user = np.random.lognormal(4, 2, num_users).astype(int) + 500
    indices_per_user = np.insert(samples_per_user.cumsum(), 0, 0, 0)
    num_total_samples = indices_per_user[-1]

    # Create mean of data for each user, each user will have different distribution
    sig = np.random.uniform(0.1, 10)
    mean = np.random.uniform(low=-0.1, high=0.1)
    cov = np.random.uniform(low=0.0, high=0.01)
    #print("mean -cov", mean,cov)
    mean_X = np.random.normal(mean, cov, dim)

    X_total = np.zeros((num_total_samples, dim))
    y_total = np.zeros(num_total_samples)

    for n in range(num_users):
        # Generate data
        X_n = np.random.multivariate_normal(mean_X, sig * np.diag(S), samples_per_user[n])
        X_total[indices_per_user[n]:indices_per_user[n+1], :] = X_n

    # Normalize all X's using LAMBDA
    norm = np.sqrt(np.linalg.norm(X_total.T.dot(X_total), 2) / num_total_samples)
    X_total /= norm

    # Generate weights and labels
    W = np.random.rand(dim)
    y_total = X_total.dot(W)
    noise_variance = 0.01
    y_total = y_total + np.sqrt(noise_ratio) * np.random.randn(num_total_samples)

    for n in range(num_users):
        X_n = X_total[indices_per_user[n]:indices_per_user[n+1], :]
        y_n = y_total[indices_per_user[n]:indices_per_user[n+1]]
        X_split[n] = X_n.tolist()
        y_split[n] = y_n.tolist()

        # print("User {} has {} samples.".format(n, samples_per_user[n]))

    print("=" * 80)
    print("Generated synthetic data for logistic regression successfully.")
    print("Summary of the generated data:".format(kappa))
    print("    Total # users       : {}".format(num_users))
    print("    Input dimension     : {}".format(dim))
    print("    rho                 : {}".format(kappa))
    print("    Total # of samples  : {}".format(num_total_samples))
    print("    Minimum # of samples: {}".format(np.min(samples_per_user)))
    print("    Maximum # of samples: {}".format(np.max(samples_per_user)))
    print("=" * 80)

    return X_split, y_split


def save_total_data():
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    train_path = os.path.join("data", "train", "mytrain.json")
    test_path = os.path.join("data", "test", "mytest.json")
    for path in [os.path.join("data", "train"), os.path.join("data", "test")]:
        if not os.path.exists(path):
            os.makedirs(path)

    X, y = generate_linear_data(NUM_USER, Kappa, Dim, Noise)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(NUM_USER):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len
        print("User: ",uname, " Num Sample: ", num_samples )
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    
    print("=" * 80)
    print("Saved all users' data sucessfully.")
    print("    Train path:", os.path.join(os.curdir, train_path))
    print("    Test path :", os.path.join(os.curdir, test_path))
    print("=" * 80)


def main():
    #generate_x()
    save_total_data()


if __name__ == '__main__':
    main()

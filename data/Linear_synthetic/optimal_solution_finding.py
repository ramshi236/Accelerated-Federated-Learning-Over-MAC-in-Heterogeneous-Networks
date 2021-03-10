import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn as sk
np.random.seed(0)

NUM_USER = 100

def normalize_data(X):

    #nomarlize all feature of data between (-1 and 1)
    normX = X - X.min()
    normX = normX / (X.max() - X.min())

    # nomarlize data with respect to -1 < X.X^T < 1.
    temp = normX.dot(normX.T)
    return normX/np.sqrt(temp.max())


def finding_optimal_synthetic(num_users=100, kappa=10, dim = 40, noise_ratio=0.05):
    
    powers = - np.log(kappa) / np.log(dim) / 2
    DIM = np.arange(dim)
    S = np.power(DIM+1, powers)

    # Creat list data for all users 
    X_split = [[] for _ in range(num_users)]  # X for each user
    y_split = [[] for _ in range(num_users)]  # y for each user
    samples_per_user = np.random.lognormal(4, 2, num_users).astype(int) + 500
    indices_per_user = np.insert(samples_per_user.cumsum(), 0, 0, 0)
    num_total_samples = indices_per_user[-1]

    # Create mean of data for each user, each user will have different distribution
    mean_X = np.array([np.random.randn(dim) for _ in range(num_users)])

    # Covariance matrix for X
    X_total = np.zeros((num_total_samples, dim))
    y_total = np.zeros(num_total_samples)

    for n in range(num_users):
        # Generate data
        X_n = np.random.multivariate_normal(mean_X[n], np.diag(S), samples_per_user[n])
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
        X_n = X_total[indices_per_user[n]:indices_per_user[n+1],:]
        y_n = y_total[indices_per_user[n]:indices_per_user[n+1]]
        X_split[n] = X_n.tolist()
        y_split[n] = y_n.tolist()
    
    # split data to get training data 
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(NUM_USER):
        num_samples = len(X_split[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len
        train_x.append(X_split[i][:train_len])
        train_y.append(y_split[i][:train_len])
        test_x.append(X_split[i][train_len:])
        test_y.append(y_split[i][train_len:])

    train_xc = np.concatenate(train_x)
    train_yc = np.concatenate(train_y)
    test_xc = np.concatenate(test_x)
    test_yc = np.concatenate(test_y)
    
    # # finding optimal
    X_X_T = np.zeros(shape=(dim+1,dim+1))
    X_Y = np.zeros(shape=(dim+1,1))

    for n in range(num_users):
        X = np.array(train_x[i])
        y = np.array(train_y[i])
        one = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((one, X), axis = 1)
        X_X_T += Xbar.T.dot(Xbar)*len(y)/len(train_yc)
        X_Y += np.array(Xbar).T.dot(y).reshape((dim+1, 1))*len(y)/len(train_yc)
    
    # get optimal point.
    w = np.linalg.inv(X_X_T).dot(X_Y)

    # caculate loss over all devices
    loss = 0
    for n in range(num_users):
        X = np.array(train_x[i])
        y = np.array(train_y[i])
        one = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((one, X), axis = 1)
        y_predict = Xbar.dot(w)
        loss += sk.metrics.mean_squared_error(y,y_predict)*len(y)/len(train_yc)

    return loss

def main():
    loss = 0
    loss = finding_optimal_synthetic()
    print("loss for train data", loss)

if __name__ == "__main__":
    main()


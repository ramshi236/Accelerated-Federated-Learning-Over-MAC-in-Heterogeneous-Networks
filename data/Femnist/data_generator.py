import emnist
import numpy as np
from tqdm import trange
import random
import json
import os
import argparse
from os.path import dirname


def generate_data(similarity, num_of_users=100, samples_num=20):
    root_path = os.path.dirname(__file__)
    train_path = root_path + '/data/train/mytrain.json'
    test_path = root_path + '/data/test/mytest.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dataset = 'balanced'
    images, train_labels = emnist.extract_training_samples(dataset)  # TODO: add test samples
    images = np.reshape(images, (images.shape[0], -1))
    images = images.astype(np.float32)
    train_labels = train_labels.astype(np.int)
    num_of_labels = len(set(train_labels))

    emnist_data = []
    for i in range(min(train_labels), num_of_labels + min(train_labels)):
        idx = train_labels == i
        emnist_data.append(images[idx])

    iid_samples = int(similarity * samples_num)
    X = [[] for _ in range(num_of_users)]
    y = [[] for _ in range(num_of_users)]
    idx = np.zeros(num_of_labels, dtype=np.int64)

    # create %similarity of iid data
    for user in range(num_of_users):
        labels = np.random.randint(0, num_of_labels, iid_samples)
        for label in labels:
            X[user].append(emnist_data[label][idx[label]].tolist())
            y[user] += (label * np.ones(1)).tolist()
            idx[label] += 1

    print(idx)

    # fill remaining data
    for user in range(num_of_users):
        label = user % num_of_labels
        X[user] += emnist_data[label][idx[label]:idx[label] + samples_num - iid_samples].tolist()
        y[user] += (label * np.ones(samples_num - iid_samples)).tolist()
        idx[label] += samples_num - iid_samples

    print(idx)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(num_of_users, ncols=120):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    print(train_data['num_samples'])
    print(sum(train_data['num_samples']))

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=float, default=0)
    parser.add_argument("--num_of_users", type=int, default=100)
    parser.add_argument("--samples_num", type=int, default=20)
    args = parser.parse_args()
    generate_data(similarity=args.similarity, num_of_users=args.num_of_users, samples_num=args.samples_num)

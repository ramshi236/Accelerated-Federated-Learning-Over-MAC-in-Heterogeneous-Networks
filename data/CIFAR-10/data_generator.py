import emnist
import numpy as np
from tqdm import trange
import random
import json
import os
from functools import reduce


def generate_data(similarity: int,  num_of_users=10, samples_num=5000):
    """
    generate CIFAR-10 data among 10 users with different similarities
    :param similarity: portion of similar data between users. number between 0 to 1
    :param num_of_users: number of users data distributed among
    :param samples_num: number of samples distributed to each user
    """
    root_path = os.path.dirname(__file__)
    train_path = root_path + '/data/train/mytrain.json'
    test_path = root_path + '/data/test/mytest.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifar_dicts = []
    for i in range(1, 6):
        cifar_dicts.append(unpickle(root_path + '/cifar-10-batches-py/data_batch_' + f"{i}"))

    train_images = np.concatenate([cifar_dict['data'] for cifar_dict in cifar_dicts])
    # train_labels = reduce((lambda x, y: x + y), [cifar_dict['labels'] for cifar_dict in cifar_dicts])
    train_labels = np.concatenate([cifar_dict['labels'] for cifar_dict in cifar_dicts])
    train_images = train_images.astype(np.float32)
    train_labels = train_labels.astype(np.int)
    num_of_labels = len(set(train_labels))

    cifar_dict = unpickle(root_path + '/cifar-10-batches-py/test_batch')
    test_images = cifar_dict['data']
    test_labels = np.array(cifar_dict['labels'])
    test_images = test_images.astype(np.float32)
    test_labels = test_labels.astype(np.int)

    cifar_data = []
    for i in range(min(train_labels), num_of_labels + min(train_labels)):
        idx = train_labels == i
        cifar_data.append(train_images[idx])

    iid_samples = int(similarity * samples_num)
    X_train = [[] for _ in range(num_of_users)]
    y_train = [[] for _ in range(num_of_users)]
    idx = np.zeros(num_of_labels, dtype=np.int64)

    # fill users data by labels
    for user in range(num_of_users):
        label = user % num_of_labels
        X_train[user] += cifar_data[label][idx[label]:idx[label] + samples_num - iid_samples].tolist()
        y_train[user] += (label * np.ones(samples_num - iid_samples)).tolist()
        idx[label] += samples_num - iid_samples

    print(idx)

    # create %similarity of iid data
    for user in range(num_of_users):
        labels = np.random.randint(0, num_of_labels, iid_samples)
        for label in labels:
            while idx[label] >= len(cifar_data[label]):
                label = (label + 1) % num_of_labels
            X_train[user].append(cifar_data[label][idx[label]].tolist())
            y_train[user] += (label * np.ones(1)).tolist()
            idx[label] += 1

    print(idx)

    # create test data
    X_test = test_images.tolist()
    y_test = test_labels.tolist()

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(num_of_users):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X_train[i], y_train[i]))
        random.shuffle(combined)
        X_train[i][:], y_train[i][:] = zip(*combined)
        train_len = len(X_train[i])
        test_len = int(len(test_images) / num_of_users)

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[test_len * i:test_len * (i + 1)],
                                         'y': y_test[test_len * i:test_len * (i + 1)]}
        test_data['num_samples'].append(test_len)

    print(train_data['num_samples'])
    print(sum(train_data['num_samples']))
    print(sum(test_data['num_samples']))

    print("Saving data, please wait")
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    print("Saving completed")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
    return data_dict


if __name__ == '__main__':
    generate_data(similarity=1)

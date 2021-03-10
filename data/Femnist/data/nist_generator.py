from __future__ import division
import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange

from PIL import Image

NUM_USER = 50
CLASS_PER_USER = 50
FEMNIST = True  # True: generate data will full 62 label, False: only 26 labels for lowercase
SAMPLE_NUM_MEAN = 400
SAMPLE_NUM_STD = 110


def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers : total 10
    - 10 through 35 for classes representing respective uppercase letters : 26
    - 36 through 61 for classes representing respective lowercase letters : 26 
    - in total we have 10 + 26 + 26 = 62 class for FEMIST   tiwand only 36-61 for FEMIST*  
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90:  # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)


def load_image(file_name):
    '''read in a png
    Return: a flatted list representing the image
    '''
    size = (28, 28)
    img = Image.open(file_name)
    gray = img.convert('L')
    gray.thumbnail(size, Image.ANTIALIAS)
    arr = np.asarray(gray).copy()
    vec = arr.flatten()
    vec = vec / 255  # scale all pixel values to between 0 and 1
    vec = vec.tolist()

    return vec


def main():
    file_dir = "raw_data/by_class"

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    if(FEMNIST):
        train_path = "train/nisttrain.json"
        test_path = "test/nisttest.json"
    else:
        train_path = "train/femnisttrain.json"
        test_path = "test/femnisttest.json"

    X = [[] for _ in range(NUM_USER)]
    y = [[] for _ in range(NUM_USER)]

    nist_data = {}

    for class_ in os.listdir(file_dir):

        real_class = relabel_class(class_)

        if(FEMNIST):
            full_img_path = file_dir + "/" + class_ + "/train_" + class_
            all_files_this_class = os.listdir(full_img_path)
            random.shuffle(all_files_this_class)
            sampled_files_this_class = all_files_this_class[:7000]
            imgs = []
            for img in sampled_files_this_class:
                imgs.append(load_image(full_img_path + "/" + img))
            class_ = relabel_class(class_)
            print("Class:", class_)
            nist_data[class_] = imgs  # a list of list, key is (0, 25)
            print("Image len:", len(imgs))

        else:
            if real_class >= 36 and real_class <= 61:
                full_img_path = file_dir + "/" + class_ + "/train_" + class_
                all_files_this_class = os.listdir(full_img_path)
                random.shuffle(all_files_this_class)
                sampled_files_this_class = all_files_this_class[:7000]
                imgs = []
                for img in sampled_files_this_class:
                    imgs.append(load_image(full_img_path + "/" + img))
                class_ = relabel_class(class_)
                print(class_)
                nist_data[class_-36] = imgs  # a list of list, key is (0, 25)
                print(len(imgs))

    # assign samples to users by power law
    normal_std = np.sqrt(np.log(1 + (lognormal_std/lognormal_mean)**2))
    normal_mean = np.log(lognormal_mean) - normal_std**2 / 2

    num_samples = np.random.lognormal(normal_mean, normal_std, (NUM_USER)) + 5
    #num_samples = np.random.normal(SAMPLE_NUM_MEAN,SAMPLE_NUM_STD,(NUM_USER))

    if(FEMNIST):
        idx = np.zeros(62, dtype=np.int64)
    else:
        idx = np.zeros(26, dtype=np.int64)

    for user in range(NUM_USER):
        num_sample_per_class = int(num_samples[user]/CLASS_PER_USER)
        if num_sample_per_class < 2:
            num_sample_per_class = 2

        for j in range(CLASS_PER_USER):
            if(FEMNIST):
                class_id = (user + j) % 62
            else:
                class_id = (user + j) % 26

            if idx[class_id] + num_sample_per_class < len(nist_data[class_id]):
                idx[class_id] = 0
            X[user] += nist_data[class_id][idx[class_id]
                : (idx[class_id] + num_sample_per_class)]
            y[user] += (class_id * np.ones(num_sample_per_class)).tolist()
            idx[class_id] += num_sample_per_class

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {
            'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {
            'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()

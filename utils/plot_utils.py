import matplotlib.pyplot as plt
import matplotlib
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os
from pathlib import Path

plt.rcParams.update({'font.size': 14})


def read_from_results(file_name):
    hf = h5py.File(file_name, 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc


# TODO: replace all args with input_dict
def get_all_training_data_value(num_glob_iters, algorithm, dataset, times, similarity, noise):
    train_acc = np.zeros((times, num_glob_iters))
    train_loss = np.zeros((times, num_glob_iters))
    glob_acc = np.zeros((times, num_glob_iters))

    file_name = "./results/" + dataset + "_" + algorithm
    file_name += "_" + str(similarity) + "s"
    if noise:
        file_name += '_noisy'

    for i in range(times):
        f = file_name + "_" + str(i) + ".h5"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(read_from_results(f))[:, :num_glob_iters]
    return glob_acc, train_acc, train_loss


def average_smooth(data, window_len=10, window='hanning'):
    results = []
    if window_len < 3:
        return data
    for i in range(len(data)):
        x = data[i]
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('numpy.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        results.append(y[window_len - 1:])
    return np.array(results)


def average_data(num_glob_iters, algorithm, dataset, times, similarity, noise):
    glob_acc, train_acc, train_loss = get_all_training_data_value(num_glob_iters, algorithm, dataset, times, similarity,
                                                                  noise)

    glob_acc_data = np.average(glob_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())
    print("std:", np.std(max_accurancy))
    print("Mean:", np.mean(max_accurancy))

    # store average value to h5 file
    file_name = "./results/" + dataset + "_" + algorithm
    file_name += "_" + str(similarity) + "s"
    if noise:
        file_name += '_noisy'
    file_name += "_avg.h5"

    if len(glob_acc) != 0 & len(train_acc) & len(train_loss):
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()
    return 0


def get_plot_dict(input_dict, algorithms, local_epochs):
    keys = ["dataset", "learning_rate", "num_glob_iters", "users_per_round", "batch_size", "local_epochs",
            "similarity", "noise"]
    plot_dict = {x: input_dict[x] for x in keys}
    plot_dict["local_epochs"] = local_epochs
    plot_dict["algorithms"] = algorithms
    return plot_dict


def plot_by_epochs(dataset, algorithms, num_glob_iters, learning_rate, users_per_round, batch_size, local_epochs,
                   similarity, noise):
    # TODO: check if i can take all this args from the results file
    """take the Monta Carlo simulation and present it SCAFFOLD vs FedAvg"""
    colours = ['r', 'g', 'b']
    fig, axs = plt.subplots(1, len(local_epochs), constrained_layout=True)
    if len(algorithms) == 2:
        fig.suptitle(f"{algorithms[0]} vs {algorithms[1]} - {dataset}")
    elif len(algorithms) == 1:
        fig.suptitle(f"{algorithms[0]} - {dataset}")

    if len(local_epochs) == 1:
        axs = [axs]

    for k, epochs in enumerate(local_epochs):
        axs[k].set_xlabel("Global Iterations")
        axs[k].set_ylabel("Accuracy")
        axs[k].set_title("number of local epochs =" + str(epochs))

        for j, algorithm in enumerate(algorithms):
            file_name = "./results/" + dataset
            file_name += "_" + algorithm
            file_name += "_" + str(learning_rate) + "lr"
            file_name += "_" + str(users_per_round) + "u"
            file_name += "_" + str(batch_size) + "b"
            file_name += "_" + str(epochs) + "e"
            file_name += "_" + str(similarity) + "s"
            if noise:
                file_name += '_noisy'
            file_name += "_avg.h5"
            train_acc, train_loss, glob_acc = np.array(read_from_results(file_name))[:, :num_glob_iters]
            axs[k].plot(glob_acc, color=colours[j], label=algorithm)
            axs[k].legend(loc="lower right")
    plt.show()


def plot_by_similarities(dataset, algorithms, noises, similarities, num_glob_iters):
    # TODO: check if i can take all this args from the results file
    fig, axs = plt.subplots(1, len(similarities), constrained_layout=True)
    if len(similarities) == 1:
        axs = [axs]

    for k, similarity in enumerate(similarities):
        axs[k].set_xlabel("Global Iterations")
        axs[k].set_ylabel("Accuracy")
        axs[k].set_title(str(100 * similarity) + "% Similarity")

        for noise in noises:
            for j, algorithm in enumerate(algorithms):
                file_name = "./results/" + dataset
                file_name += "_" + algorithm
                file_name += "_" + str(similarity) + "s"
                label = algorithm
                if noise:
                    file_name += '_noisy'
                    label += ' with noise'
                file_name += "_avg.h5"
                train_acc, train_loss, glob_acc = np.array(read_from_results(file_name))[:, :num_glob_iters]
                axs[k].plot(glob_acc, label=label)
                axs[k].legend(loc="lower right")
    plt.show()

import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

plt.rcParams.update({'font.size': 14})


def simple_read_data(loc_ep, alg):
    hf = h5py.File("./results/" + '{}_{}.h5'.format(alg, loc_ep), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc


def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],
                            hyper_learning_rate=[], algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if (lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"

        string_learning_rate = str(learning_rate[i])

        if (algorithms_list[i] == "FEDL"):
            string_learning_rate = string_learning_rate + "_" + str(hyper_learning_rate[i])
        algorithms_list[i] = algorithms_list[i] + \
                             "_" + string_learning_rate + "_" + str(num_users) + \
                             "u" + "_" + str(batch_size[i]) + "b" + "_" + str(loc_ep1[i])
        if (rho[i] > 0):
            algorithms_list[i] += "_" + str(rho[i]) + "p"

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data("avg", dataset + "_" + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    return glob_acc, train_acc, train_loss


def get_data_label_style(input_data=[], linestyles=[], algs_lbl=[], lamb=[], loc_ep1=0, batch_size=0):
    data, lstyles, labels = [], [], []
    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i] + str(lamb[i]) + "_" +
                      str(loc_ep1[i]) + "e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels


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


def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],
                            hyper_learning_rate=[], algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    # glob_acc, train_acc, train_loss = get_training_data_value(
    #    users_per_round, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,hyper_learning_rate, algorithms_list, batch_size, dataset)

    glob_acc_, train_acc_, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, hyper_learning_rate, algorithms_list,
                                                                 batch_size, rho, dataset)
    glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    plt.figure(1)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', ':']
    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i],
                 label=algorithms_list[i] + str(lamb[i]) + "_" + str(loc_ep1[i]) + "e" + "_" + str(batch_size[i]) + "b")
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper())
    # plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.png', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')
    plt.figure(2)

    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, start:], linestyle=linestyles[i], label=algorithms_list[i] + str(lamb[i]) +
                                                                       "_" + str(loc_ep1[i]) + "e" + "_" + str(
            batch_size[i]) + "b")
        # plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    # plt.ylim([train_loss.min(), 0.5])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, start:], linestyle=linestyles[i],
                 label=algorithms_list[i] + str(lamb[i]) + "_" + str(loc_ep1[i]) + "e" + "_" + str(batch_size[i]) + "b")
        # plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    # plt.ylim([0.6, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')


def get_max_value_index(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[],
                        batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)
    for i in range(Numb_Algs):
        print("Algorithm: ", algorithms_list[i], "Max testing Accurancy: ", glob_acc[i].max(
        ), "Index: ", np.argmax(glob_acc[i]), "local update:", loc_ep1[i])


def plot_summary_mnist(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], hyper_learning_rate=[],
                       algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)

    # glob_acc, train_acc, train_loss = get_training_data_value(users_per_round, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, hyper_learning_rate, algorithms_list, batch_size, rho, dataset)

    glob_acc_, train_acc_, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, hyper_learning_rate, algorithms_list,
                                                                 batch_size, rho, dataset)
    glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    plt.figure(1)

    linestyles = ['-', '--', '-.', ':']
    algs_lbl = ["FEDL", "FedAvg",
                "FEDL", "FedAvg",
                "FEDL", "FedAvg",
                "FEDL", "FEDL"]

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)  # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    # min = train_loss.min()
    min = train_loss.min() - 0.001
    max = 0.46
    # max = train_loss.max() + 0.01
    num_al = 2
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right', prop={'size': 10})

    for i in range(num_al):
        stringbatch = str(batch_size[i + 2])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i + num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right', prop={'size': 10})

    for i in range(4):
        stringbatch = str(batch_size[i + 4])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(train_loss[i + num_al * 2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al * 2] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al * 2]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right', prop={'size': 10})

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)  # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    # min = train_loss.min()
    min = 0.82
    max = glob_acc.max() + 0.001  # train_loss.max() + 0.01
    num_al = 2
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right', prop={'size': 10})

    for i in range(num_al):
        stringbatch = str(batch_size[i + 2])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i + num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al * 1]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right', prop={'size': 10})

    for i in range(4):
        stringbatch = str(batch_size[i + 4])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(glob_acc[i + num_al * 2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al * 2] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al * 2]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right', prop={'size': 10})

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'test_accu.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) + 'test_accu.png', bbox_inches='tight')


def plot_summary_nist(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], hyper_learning_rate=[],
                      algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    # glob_acc, train_acc, train_loss = get_training_data_value( users_per_round, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, hyper_learning_rate, algorithms_list, batch_size, rho, dataset)
    glob_acc_, train_acc_, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, hyper_learning_rate, algorithms_list,
                                                                 batch_size, rho, dataset)
    glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].max())
    plt.figure(1)

    linestyles = ['-', '--', '-.', ':']
    algs_lbl = ["FEDL", "FedAvg", "FEDL",
                "FEDL", "FedAvg", "FEDL",
                "FEDL", "FedAvg", "FEDL"]
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(111)  # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    # min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 3  # train_loss.max() + 0.01
    num_al = 3
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i]) + ', $K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right', prop={'size': 10})

    for i in range(num_al):
        stringbatch = str(batch_size[i + num_al])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i + num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al]) + ', $K_l = $' + str(loc_ep1[i + num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right', prop={'size': 10})

    for i in range(num_al):
        stringbatch = str(batch_size[i + num_al * 2])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(train_loss[i + num_al * 2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al * 2] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al * 2]) + ', $K_l = $' + str(loc_ep1[i + num_al * 2]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right', prop={'size': 10})

    ax.set_title('FEMNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)  # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    # min = train_loss.min()
    num_al = 3
    min = 0.3
    max = glob_acc.max() + 0.01  # train_loss.max() + 0.01
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i]) + ', $K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right', prop={'size': 10})

    for i in range(num_al):
        stringbatch = str(batch_size[i + num_al])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i + num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al * 1]) + ', $K_l = $' + str(loc_ep1[i + num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right', prop={'size': 10})

    for i in range(num_al):
        stringbatch = str(batch_size[i + num_al * 2])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(glob_acc[i + num_al * 2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al * 2] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(
                     hyper_learning_rate[i + num_al * 2]) + ', $K_l = $' + str(loc_ep1[i + 2 * num_al]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right', prop={'size': 10})

    ax.set_title('FEMNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'test_accu.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) + 'test_accu.png', bbox_inches='tight')


def plot_summary_linear(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], hyper_learning_rate=[],
                        algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,
                                                              hyper_learning_rate, algorithms_list, batch_size, rho,
                                                              dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    plt.figure(1)

    linestyles = ['-', '-', '-', '-']
    markers = ["o", "v", "s", "*", "x", "P"]
    algs_lbl = ["FEDL", "FEDL", "FEDL", "FEDL",
                "FEDL", "FEDL", "FEDL", "FEDL",
                "FEDL", "FEDL", "FEDL", "FEDL"]
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)  # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    # min = train_loss.min()
    num_al = 4
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + ": " + '$\eta = $' + str(hyper_learning_rate[i]), marker=markers[i], markevery=0.4,
                 markersize=5)

    ax1.hlines(y=0.035, xmin=0, xmax=200, linestyle='--', label="optimal solution", color="m")
    ax1.legend(loc='upper right', prop={'size': 10})
    ax1.set_ylim([0.02, 0.5])
    ax1.set_title('$\\rho = $' + str(rho[0]))
    ax1.grid(True)
    for i in range(num_al):
        str_rho = ', $\eta  = $' + str(rho[i])
        ax2.plot(train_loss[i + num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + ": " + '$\eta = $' + str(hyper_learning_rate[i + num_al]),
                 marker=markers[i], markevery=0.4, markersize=5)

    ax2.hlines(y=0.035, xmin=0, xmax=200, linestyle='--', label="optimal solution", color="m")
    ax2.set_ylim([0.02, 0.5])
    # ax2.legend(loc='upper right')
    ax2.set_title('$\\rho = $' + str(rho[0 + num_al]))
    ax2.grid(True)
    for i in range(num_al):
        str_rho = ', $\rho  = $' + str(rho[i])
        ax3.plot(train_loss[i + num_al * 2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al * 2] + ": " + '$\eta = $' + str(hyper_learning_rate[i + num_al * 2]),
                 marker=markers[i], markevery=0.4, markersize=5)

    ax3.hlines(y=0.035, xmin=0, xmax=200, linestyle='--',
               label="optimal solution", color="m")
    ax3.set_ylim([0.02, 0.5])
    # ax3.legend(loc='upper right')
    ax3.set_title('$\\rho = $' + str(rho[0 + 2 * num_al]))
    ax3.grid(True)
    ax.set_title('Synthetic dataset', y=1.1)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.png', bbox_inches='tight')


def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0,
                                hyper_learning_rate=0, algorithms="", batch_size=0, dataset="", rho=0, times=5):
    train_acc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list = [algorithms] * times

    for i in range(times):
        if (lamb > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb)

        string_learning_rate = str(learning_rate)

        if (algorithms_list[i] == "FEDL"):
            string_learning_rate = string_learning_rate + "_" + str(hyper_learning_rate)

        algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(
            batch_size) + "b" + "_" + str(loc_ep1)

        if (rho > 0):
            algorithms_list[i] += "_" + str(rho) + "p"

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(str(i), dataset + "_" + algorithms_list[i]))[:, :Numb_Glob_Iters]

    return glob_acc, train_acc, train_loss


def average_data(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, hyper_learning_rate, algorithms, batch_size,
                 dataset, rho, times):
    glob_acc, train_acc, train_loss = get_all_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                  learning_rate, hyper_learning_rate, algorithms,
                                                                  batch_size, dataset, rho, times)
    # store average value to h5 file
    glob_acc_data = np.average(glob_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())
    print("std:", np.std(max_accurancy))
    print("Mean:", np.mean(max_accurancy))

    alg = dataset + "_" + algorithms
    alg += "_" + str(learning_rate)

    if (algorithms == "FEDL"):
        alg += "_" + str(hyper_learning_rate)

    alg += "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)

    if (lamb > 0):
        alg += "_" + str(lamb) + "L"

    if (rho > 0):
        alg += "_" + str(rho) + "p"

    # alg = alg + "_" + str(learning_rate) + "_" + str(hyper_learning_rate) + "_" + str(lamb) + "_" + str(users_per_round) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
    alg = alg + "_" + "avg"
    if (len(glob_acc) != 0 & len(train_acc) & len(train_loss)):
        with h5py.File("./results/" + '{}.h5'.format(alg, loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()
    return 0


def plot_summary_one_mnist(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],
                           hyper_learning_rate=[], algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    # glob_acc, train_acc, train_loss = get_training_data_value(
    #    users_per_round, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,hyper_learning_rate, algorithms_list, batch_size, dataset)

    glob_acc_, train_acc_, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, hyper_learning_rate, algorithms_list,
                                                                 batch_size, rho, dataset)
    glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    plt.figure(1)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':']
    markers = ["o", "v", "s", "*", "x", "P"]
    algs_lbl = ["FEDL", "FedAvg", "FEDL", "FedAvg"]
    plt.grid(True)
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], marker=markers[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch, markevery=0.4, markersize=5)

    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper())
    plt.ylim([0.85, train_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.png', bbox_inches="tight")
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.pdf', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')
    plt.figure(2)

    plt.grid(True)
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], marker=markers[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch, markevery=0.4, markersize=5)

        # plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    plt.ylim([train_loss.min() - 0.01, 0.7])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png', bbox_inches="tight")
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.pdf', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    plt.grid(True)
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], marker=markers[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch, markevery=0.4, markersize=5)
        # plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    plt.ylim([0.8, glob_acc.max() + 0.005])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png', bbox_inches="tight")
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.pdf', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')


def plot_summary_one_nist(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],
                          hyper_learning_rate=[], algorithms_list=[], batch_size=0, rho=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    # glob_acc, train_acc, train_loss = get_training_data_value(
    #    users_per_round, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,hyper_learning_rate, algorithms_list, batch_size, dataset)

    glob_acc_, train_acc_, train_loss_ = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, hyper_learning_rate, algorithms_list,
                                                                 batch_size, rho, dataset)
    glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    plt.figure(1)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':']
    markers = ["o", "v", "s", "*", "x", "P"]
    algs_lbl = ["FEDL", "FedAvg", "FedAvg"]
    plt.grid(True)
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], marker=markers[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch, markevery=0.4, markersize=5)

    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title('FEMNIST')
    # plt.ylim([0.85, train_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.png', bbox_inches="tight")
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.pdf', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')
    plt.figure(2)

    plt.grid(True)
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], marker=markers[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch, markevery=0.4, markersize=5)

        # plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title('FEMNIST')
    # plt.ylim([train_loss.min(), 0.7])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png', bbox_inches="tight")
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.pdf', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    plt.grid(True)
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if (stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], marker=markers[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch, markevery=0.4, markersize=5)
        # plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    # plt.ylim([0.8, glob_acc.max() + 0.005])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title('FEMNIST')
    # ax.set_title('FEMNIST', y=1.02)
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png', bbox_inches="tight")
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.pdf', bbox_inches="tight")
    # plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')



from utils.plot_utils import *
from simulate import simulate
from data.Femnist.data_generator import generate_data as generate_femnist_data
from data.CIFAR.data_generator import generate_data as generate_cifar10_data

cifar_dict = {"model": "cnn",
              "batch_size": 60,
              "learning_rate": 0.08,
              "local_epochs": 1,
              "L": 0.004,
              "users_per_round": 8}

femnist_dict = {"model": "mclr",
                "batch_size": 4,
                "learning_rate": 0.001,
                "local_epochs": 1,
                "L": 0,
                "users_per_round": 20}

input_dict = {}

dataset = 'CIFAR'
if dataset == 'CIFAR':
    input_dict = cifar_dict
elif dataset == 'Femnist':
    input_dict = femnist_dict

num_glob_iters = 200
times = 1
algorithms = ["SCAFFOLD", "FedAvg"]
noises = [True, False]
similarities = [1]


for similarity in similarities:
    # print("Downloading dataset")
    # generate_cifar10_data(similarity)
    for noise in noises:
        for algorithm in algorithms:
            simulate(**input_dict, dataset=dataset, algorithm=algorithm, similarity=similarity, noise=noise,
                     num_glob_iters=num_glob_iters, times=times)

plot_by_similarities(dataset, algorithms, noises, similarities, num_glob_iters)

# plot_dict = get_plot_dict(input_dict, algorithms, epochs)
# plot_by_epochs(**plot_dict)

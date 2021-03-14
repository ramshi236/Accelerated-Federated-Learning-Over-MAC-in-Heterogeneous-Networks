from utils.plot_utils import *
from simulate import simulate
from data.Femnist.data_generator import generate_data as generate_femnist_data
from data.CIFAR.data_generator import generate_data as generate_cifar10_data


def generate_data(dataset, similarity):
    if dataset == 'CIFAR':
        generate_cifar10_data(similarity)
    elif dataset == 'Femnist':
        generate_femnist_data(similarity)


cifar_dict = {"model": "cnn",
              "batch_size": 60,
              "learning_rate": 0.008,
              "local_epochs": 1,
              "L": 0.04,
              "users_per_round": 8}

femnist_dict = {"model": "mclr",
                "batch_size": 4,
                "learning_rate": 0.001,
                "local_epochs": 1,
                "L": 0,
                "users_per_round": 20}

input_dict = {}

dataset = 'Femnist'
if dataset == 'CIFAR':
    input_dict = cifar_dict
elif dataset == 'Femnist':
    input_dict = femnist_dict

num_glob_iters = 300
times = 15
algorithms = ["SCAFFOLD", "FedAvg"]
noises = [True, False]
similarities = [1, 0.1, 0]


# for similarity in similarities:
#     generate_data(dataset, similarity)
#     for noise in noises:
#         for algorithm in algorithms:
#             simulate(**input_dict, dataset=dataset, algorithm=algorithm, similarity=similarity, noise=noise,
#                      num_glob_iters=num_glob_iters, times=times)

plot_accuracy(dataset, algorithms, noises, similarities, num_glob_iters)
plot_norms(dataset, algorithms, noises, similarities, num_glob_iters)

# plot_dict = get_plot_dict(input_dict, algorithms, epochs)
# plot_norms(**plot_dict)




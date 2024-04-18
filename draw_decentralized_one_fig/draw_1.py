import numpy as np
import matplotlib.pyplot as plt
import pickle

interval = 100
rounds = 200

set_iteration = [r * interval for r in range(rounds+1)]
colors = ['orange', 'blue', 'green', 'red']
markers = ['s', '^', 'v', 'o']

FONTSIZE = 25

def draw(task, dataset):
    """
    Draw the curve of experimental results
    """

    labels = ['Heterogeneity', 'Bound of A under SLF', 'Bound of A under DLF']
    path_list = [
        'record/'+ task + '_' + dataset + '/Centralized_n=10_b=0/LabelSeperation/CSGD_baseline_mean_hetero_list',
        'record/'+ task + '_' + dataset + '/Centralized_n=10_b=1/LabelSeperation/CSGD_label_flipping_mean_Bound_A',
        'record/'+ task + '_' + dataset + '/Centralized_n=10_b=1/LabelSeperation/CSGD_furthest_label_flipping_mean_Bound_A',
    ]
    value_list = []

    for i in range(len(path_list)):
        with open(path_list[i], 'rb') as f:
            value = pickle.load(f)
            value_list.append(value)

    plt.figure(1)
    for i in range(len(path_list)):
        plt.plot(set_iteration, value_list[i], color=colors[i], marker=markers[i], label=labels[i], markevery=20)
    plt.xticks(range(0, interval * rounds+1, 5000), fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Magnitude', fontsize=20)
    plt.xlabel('iterations', fontsize=20)
    plt.title('Noniid (' + dataset.upper() +')', fontsize=20)
    plt.grid('on')
    plt.legend(fontsize=16)
    plt.ylim(0, 100)

    plt.savefig('draw_decentralized_one_fig/' + task + '_' + dataset + '_A_hetero.pdf', bbox_inches='tight')
    plt.show()


def draw_neural_network():
    """
    Draw the curve of experimental results
    """
    datasets = ['mnist', 'cifar10']
    labels = ['Heterogeneity', 'Bound of A under SLF', 'Bound of A under DLF']
    path_list = [
        'record/NeuralNetwork_mnist/Centralized_n=10_b=0/LabelSeperation/CSGD_baseline_mean_hetero_list',
        'record/NeuralNetwork_mnist/Centralized_n=10_b=1/LabelSeperation/CSGD_label_flipping_mean_Bound_A',
        'record/NeuralNetwork_mnist/Centralized_n=10_b=1/LabelSeperation/CSGD_furthest_label_flipping_mean_Bound_A',
        # 'record/NeuralNetwork_cifar10/Centralized_n=10_b=0/LabelSeperation/CSGD_baseline_mean_ladderLR_hetero_list',
        'record/NeuralNetwork_cifar10/Centralized_n=10_b=0/LabelSeperation/CSGD_baseline_mean_hetero_list',
        'record/NeuralNetwork_cifar10/Centralized_n=10_b=1/LabelSeperation/CSGD_label_flipping_mean_Bound_A',
        'record/NeuralNetwork_cifar10/Centralized_n=10_b=1/LabelSeperation/CSGD_furthest_label_flipping_mean_Bound_A',
    ]
    value_list = []

    for i in range(len(path_list)):
        with open(path_list[i], 'rb') as f:
            value = pickle.load(f)
            value_list.append(value)

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 7), sharex=True, sharey=True)
    axes[0].set_ylabel('Magnitude', fontsize=FONTSIZE)
    axes[0].set_ylim(0, 100)
    for i in range(len(datasets)):
        for j in range(len(labels)):
            axes[i].set_title('Noniid (' + datasets[i].upper() + ')', fontsize=FONTSIZE)
            axes[i].plot(set_iteration, value_list[i * len(labels) + j], color=colors[j], marker=markers[j], label=labels[j], markevery=20)
            axes[i].set_xticks(range(0, interval * rounds+1, 5000), labelsize=15)
            axes[i].tick_params(labelsize=15)
            axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
            axes[i].grid('on')

    # axes[0].legend(fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=20)

    plt.subplots_adjust(top=0.91, bottom=0.22, left=0.125, right=1, hspace=0.27, wspace=0.18)


    plt.savefig('draw_decentralized_one_fig/pdf_A_hetero/NeuralNetwork_A_hetero.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    draw('SR', 'mnist')
    # draw('cifar10')
    draw_neural_network()
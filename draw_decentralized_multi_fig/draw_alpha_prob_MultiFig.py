import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

FONTSIZE = 25

graph_name = 'Centralized_n=10_b=1'

threshold = 0

def draw(task, dataset):
    task_name = task + '_' + dataset
    alpha_list = [100, 10, 1, 0.1, 0.01, 0.001]
    prob_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    attack_list = [
        ('label_flipping', 'SLF'),
        # ('furthest_label_flipping', 'DLF')
    ]
    data = [[[] for _ in range(len(alpha_list))] for _ in range(len(attack_list))]

    aggregations = [
        ('mean', 'Mean'),
        ('CC_tau=0.3', 'CC'),
        ('faba', 'IOS/FABA'),
        ('LFighter', 'LFighter'),
        ('trimmed_mean', 'TriMean'),
    ]

    for i, (attack_name, attack_label) in enumerate(attack_list):
        for j, alpha in enumerate(alpha_list):
            for prob in prob_list:
                file_path = [task_name, graph_name, f'DirichletPartition_alpha={alpha}']
                file_name = f'CSGD_p={prob}_' + attack_name + '_mean'
                record = load_file_in_cache(file_name, path_list=file_path)
                acc_max = max(record['acc_path'])
                # acc_max = record['acc_path'][-1]
                agg_name = 'Mean'

                for (agg_code_name, agg_show_name) in aggregations:
                    file_path = [task_name, graph_name, f'DirichletPartition_alpha={alpha}']
                    file_name = f'CSGD_p={prob}_' + attack_name + '_' + agg_code_name
                    record = load_file_in_cache(file_name, path_list=file_path)
                    acc_path = record['acc_path']
                    # if acc_path[-1] - acc_max > threshold:
                    if  max(acc_path)- acc_max > threshold:
                        # acc_max = acc_path[-1]
                        acc_max = max(acc_path)
                        agg_name = agg_show_name
                data[i][j].append(rf'{acc_max:.2f}  ({agg_name})')

    len_a = len(attack_list)
    len_x = len(alpha_list)
    len_y = len(prob_list)
    fig, ax = plt.subplots(1, len_a, figsize=(4 * len_x + 3, 2 * len_y))

    for a in range(len_a):
        ax[a].set_title(attack_list[a][1] + ' (' + dataset.upper() + ')', fontsize=FONTSIZE)
        for i in range(len_x):
            for j in range(len_y):
                cell_data = data[a][i][j]
                if '(Mean)' in cell_data:
                    color = 'orange'
                else:
                    # color = 'paleturquoise'
                    color = 'lightcyan'
                ax[a].add_patch(plt.Rectangle((i, j), 1, 1, fill=True, facecolor=color,  edgecolor='black'))
                # ax[a].text(i + 0.5, j + 0.5, cell_data, color='black', ha='center', va='center', fontsize=FONTSIZE)
                ax[a].text(i + 0.5, j + 0.6, cell_data.split()[0], color='black', ha='center', fontsize=FONTSIZE-5)
                ax[a].text(i + 0.5, j + 0.4, cell_data.split()[1], color='black', ha='center', fontsize=FONTSIZE-7)



        # 设置坐标轴刻度
        ax[a].set_xticks(np.arange(len_x) + 0.5, minor=False)
        ax[a].set_yticks(np.arange(len_y) + 0.5, minor=False)
        ax[a].set_xlim(0, len_x)
        ax[a].set_ylim(0, len_y)
        ax[a].set_xlabel(r'Dirichlet distribution ($\alpha$)', fontsize=FONTSIZE)

        # 隐藏坐标轴
        ax[a].set_xticklabels(alpha_list, fontsize=FONTSIZE)
        ax[a].set_yticklabels(prob_list, fontsize=FONTSIZE)
        ax[a].tick_params(which='both', width=0)
        ax[a].grid(False)

    # ax[0].set_yticklabels(prob_list, fontsize=FONTSIZE)
    # ax[1].set_yticklabels([])
    ax[0].set_ylabel(r'Flipping probability ($p$)', fontsize=FONTSIZE)

    plt.savefig('pdf_alpha_prob_MultiFig/' + task_name +  '_alpha_prob_threshold='+ str(threshold) +'.pdf', bbox_inches='tight')  
            

if __name__ == '__main__':
    draw('NeuralNetwork', 'mnist')
    # draw('NeuralNetwork', 'cifar10')
    # draw('SR', 'mnist')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


# task_name = 'NeuralNetwork_cifar10'
# task_name = 'SR_mnist'
task_name = 'NeuralNetwork_mnist'


attack_name = 'label_flipping'
# attack_name = 'furthest_label_flipping'

graph_name = 'Centralized_n=10_b=1'

threshold = 0

def draw():
    alpha_list = [100,  1, 0.1,  0.001]
    prob_list = [0.0,  0.4, 0.7,  1.0]
    data = [[] for _ in range(len(alpha_list))]

    aggregations = [
        ('mean', 'Mean'),
        ('CC_tau=0.3', 'CC'),
        ('faba', 'IOS/FABA'),
        ('LFighter', 'LFighter'),
        ('trimmed_mean', 'TriMean'),
    ]

    for i, alpha in enumerate(alpha_list):
        for prob in prob_list:
            file_path = [task_name, graph_name, f'DirichletPartition_alpha={alpha}']
            file_name = f'CSGD_p={prob}_' + attack_name + '_mean'
            record = load_file_in_cache(file_name, path_list=file_path)
            acc_max = max(record['acc_path'])
            # acc_max = record['acc_path'][-1]
            agg_name = 'Mean'
            # if alpha == 100 and prob == 1:
            #     print(f'alpha={alpha}, prob={prob}, acc of Mean: {acc_max}')
            # elif alpha == 10 and prob == 1:
            #     print(f'alpha={alpha}, prob={prob}, acc of Mean: {acc_max}')
            # elif alpha == 100 and prob == 0.8:
            #     print(f'alpha={alpha}, prob={prob}, acc of Mean: {acc_max}')
            # elif alpha == 0.01 and prob == 0.8:
            #     print(f'alpha={alpha}, prob={prob}, acc of Mean: {acc_max}')
            # elif alpha == 0.001 and prob == 0.8:
            #     print(f'alpha={alpha}, prob={prob}, acc of Mean: {acc_max}')
            # elif alpha == 0.01 and prob == 0.6:
            #     print(f'alpha={alpha}, prob={prob}, acc of Mean: {acc_max}')



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
            data[i].append(f'{acc_max:.2f} ({agg_name})')
            # if alpha == 100 and prob == 1:
            #     print(f'alpha={alpha}, prob={prob}, acc of {agg_name}: {acc_max}')
            # elif alpha == 10 and prob == 1:
            #     print(f'alpha={alpha}, prob={prob}, acc of {agg_name}: {acc_max}')
            # elif alpha == 100 and prob == 0.8:
            #     print(f'alpha={alpha}, prob={prob}, acc of {agg_name}: {acc_max}')
            # elif alpha == 0.01 and prob == 0.8:
            #     print(f'alpha={alpha}, prob={prob}, acc of {agg_name}: {acc_max}')
            # elif alpha == 0.001 and prob == 0.8:
            #     print(f'alpha={alpha}, prob={prob}, acc of {agg_name}: {acc_max}')
            # elif alpha == 0.01 and prob == 0.6:
            #     print(f'alpha={alpha}, prob={prob}, acc of {agg_name}: {acc_max}')


    len_x = len(alpha_list)
    len_y = len(prob_list)
    fig, ax = plt.subplots(figsize=(2 * len_x, 2 * len_y))

    for i in range(len_x):
        for j in range(len_y):
            cell_data = data[i][j]
            if '(Mean)' in cell_data:
                color = 'orange'
            else:
                # color = 'paleturquoise'
                color = 'lightcyan'
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=True, facecolor=color,  edgecolor='black'))
            # ax.text(i + 0.5, j + 0.5, cell_data, color='black', ha='center', va='center', fontsize=FONTSIZE)
            ax.text(i + 0.5, j + 0.6, cell_data.split()[0], color='black', ha='center', fontsize=FONTSIZE-5)
            ax.text(i + 0.5, j + 0.4, cell_data.split()[1], color='black', ha='center', fontsize=FONTSIZE-7)



    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len_x) + 0.5, minor=False)
    ax.set_yticks(np.arange(len_y) + 0.5, minor=False)
    ax.set_xlim(0, len_x)
    ax.set_ylim(0, len_y)
    ax.set_xlabel(r'Dirichlet distribution ($\alpha$)', fontsize=FONTSIZE)
    ax.set_ylabel(r'Flipping probability ($p$)', fontsize=FONTSIZE)
    
    # 隐藏坐标轴
    ax.set_xticklabels(alpha_list, fontsize=FONTSIZE)
    ax.set_yticklabels(prob_list, fontsize=FONTSIZE)
    ax.tick_params(which='both', width=0)
    ax.grid(False)

    # axins = inset_axes(ax, width="40%", height="20%", loc='upper right',
    #                    bbox_to_anchor=(0.66, 0.66, 1, 1),
    #                    bbox_transform=ax.transAxes)
    # axins.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='lightcyan',  edgecolor='black'))
    # axins.text(0.5, 0.6, data[4][4].split()[0], color='black', ha='center', fontsize=FONTSIZE-5)
    # axins.text(0.5, 0.4, data[4][4].split()[1], color='black', ha='center', fontsize=FONTSIZE-7)
    # axins.set_xticks([0, 1.0], minor=False)
    # axins.set_yticks([0, 1.0], minor=False)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    plt.savefig('pdf_alpha_prob/' + task_name + '_' + attack_name + '_alpha_prob_threshold='+ str(threshold) +'.pdf', bbox_inches='tight')  
    plt.savefig('pdf_alpha_prob/' + task_name + '_' + attack_name + '_alpha_prob_threshold='+ str(threshold) +'.png', bbox_inches='tight')  
                   

if __name__ == '__main__':
    draw()

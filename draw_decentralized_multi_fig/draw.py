import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

colors = [ 'green', 'red',  'orange', 'blue', 'purple', 'olive']
markers = ['h', '+', 'v',  's', 'x', 'o']

# task_name = 'NeuralNetwork'
# task_name = 'SR'
graph_name = 'Centralized_n=10_b=1'
# attack_name = 'label_flipping'
attack_name = 'furthest_label_flipping'


FONTSIZE = 50

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

def draw(task_name):
    datasets = ['mnist', 'cifar10']
    # datasets = ['cifar10']
    aggregations = [
        ('mean', 'Baseline'), 
        ('mean', 'Mean'), 
        # ('median', 'CooMed'),
        # ('geometric_median', 'GeoMed'), 
        # ('Krum', 'Krum'), 
        ('trimmed_mean', 'TriMean'),
        # ('SCClip', 'SCC'),
        # ('SCClip_T', 'SCC-T'),
        ('faba', 'FABA'), 
        ('CC', 'CC'),
        # ('IOS', r'\textbf{IOS (ours)}'), 
        # ('bulyan', 'Bulyan'),
        # ('remove_outliers', 'Cutter'),
        ('LFighter', 'LFighter'),
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name = 'centralized_' + task_name + '_' + graph_name + '_' + attack_name

    # fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 13), sharex=True, sharey=True)
    # fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 13), sharex=True, sharey=True)
    # fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 19), sharex=True, sharey='row')
    fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 14), sharex=True, sharey='row')
    axes[0][0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    axes[1][0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    axes[0][0].set_ylim(0.45, 0.98)
    axes[1][0].set_ylim(0.2, 0.7)

    

    for l in range(len(datasets)):
        taskname = task_name + '_' + datasets[l]
        for i in range(len(partition_names)):
            axes[l][i].set_title(partition_names[i][1] + f' ({datasets[l]})'.upper(), fontsize=FONTSIZE)
            axes[1][i].set_xlabel('iterations', fontsize=FONTSIZE)
            axes[l][i].tick_params(labelsize=FONTSIZE)
            axes[l][i].grid('on')
            for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
                color = colors[agg_index]
                marker = markers[agg_index]
                if partition_names[i][0] == 'iidPartition' and agg_code_name == 'CC':
                    agg_code_name += '_tau=0.1'
                elif agg_code_name == 'CC':
                    agg_code_name += '_tau=0.3'

                if agg_show_name == 'Baseline':
                    file_name = 'CSGD_baseline_mean'
                    file_path = [taskname, 'Centralized_n=10_b=0', partition_names[i][0]]
                else:
                    file_name = 'CSGD_' + attack_name + '_' + agg_code_name + ''
                    file_path = [taskname, graph_name, partition_names[i][0]]

                record = load_file_in_cache(file_name, path_list=file_path)
                acc_path = record['acc_path']

                x_axis = [r*record['display_interval']
                            for r in range(record['rounds']+1)]

                axes[l][i].plot(x_axis, acc_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20)

    handles, labels = axes[0][0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE)

    leg = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE, markerscale=3)
    leg_lines = leg.get_lines()
    for i in range(len(leg_lines)):
        plt.setp(leg_lines[i], linewidth=5.0)

    # plt.subplots_adjust(top=1, bottom=0.25, left=0, right=1, hspace=0.13, wspace=0.13)
    plt.subplots_adjust(top=1, bottom=0.33, left=0, right=1, hspace=0.18, wspace=0.13)


    file_dir = os.path.dirname(os.path.abspath(__file__))
    dir_png_path = os.path.join(file_dir, 'pic', 'png')
    dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)

    suffix = ''
    pic_png_path = os.path.join(dir_png_path, pic_name + suffix + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + suffix + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    plt.show()


def draw_mnist(task_name):
    # datasets = ['mnist', 'cifar10']
    dataset = 'mnist'


    aggregations = [
        ('mean', 'Baseline'), 
        ('mean', 'Mean'), 
        # ('median', 'CooMed'),
        # ('geometric_median', 'GeoMed'), 
        # ('Krum', 'Krum'), 
        ('trimmed_mean', 'TriMean'),
        # ('SCClip', 'SCC'),
        # ('SCClip_T', 'SCC-T'),
        ('faba', 'FABA'), 
        ('CC', 'CC'),
        # ('IOS', r'\textbf{IOS (ours)}'), 
        # ('bulyan', 'Bulyan'),
        # ('remove_outliers', 'Cutter'),
        ('LFighter', 'LFighter'),
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name = 'centralized_' + task_name + '_' + dataset + '_' + graph_name + '_' + attack_name

    fig, axes = plt.subplots(1, len(partition_names), figsize=(21, 11), sharex=True, sharey=True)
    axes[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    # axes[0].set_ylim(0.45, 0.96)
    axes[0].set_ylim(0.7, 0.93)


    taskname = task_name + '_' + dataset
    for i in range(len(partition_names)):
        axes[i].set_title(partition_names[i][1] + ' (MNIST)', fontsize=FONTSIZE)
        axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
        axes[i].tick_params(labelsize=FONTSIZE)
        axes[i].grid('on')
        for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
            color = colors[agg_index]
            marker = markers[agg_index]
            if partition_names[i][0] == 'iidPartition' and agg_code_name == 'CC':
                agg_code_name += '_tau=0.1'
            elif agg_code_name == 'CC':
                agg_code_name += '_tau=0.3'
            if agg_show_name == 'Baseline':
                file_name = 'CSGD_baseline_mean'
                file_path = [taskname, 'Centralized_n=10_b=0', partition_names[i][0]]
            else:
                file_name = 'CSGD_' + attack_name + '_' + agg_code_name + ''
                file_path = [taskname, graph_name, partition_names[i][0]]
            record = load_file_in_cache(file_name, path_list=file_path)
            acc_path = record['acc_path']
            x_axis = [r*record['display_interval']
                        for r in range(record['rounds']+1)]
            axes[i].plot(x_axis, acc_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE)
    leg = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE, markerscale=3)
    leg_lines = leg.get_lines()
    for i in range(len(leg_lines)):
        plt.setp(leg_lines[i], linewidth=5.0)

    plt.subplots_adjust(top=1, bottom=0.42, left=0, right=1, hspace=0.1, wspace=0.13)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dir_png_path = os.path.join(file_dir, 'pic', 'png')
    dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)

    suffix = ''
    pic_png_path = os.path.join(dir_png_path, pic_name + suffix + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + suffix + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    draw('NeuralNetwork')
    draw_mnist('SR')

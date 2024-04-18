import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

colors = [ 'green', 'red',  'orange', 'blue', 'purple']
markers = ['h', '+', 'v',  's', 'x', 'o']



interval = 100
rounds = 200

FONTSIZE = 50

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)


def draw(task_name):
    datasets = ['mnist', 'cifar10']
    # dataset = 'mnist'

    suffix_list = [
        # ('_hetero_list', 'Heterogeneity'),
        # ('_Bound_A', 'Norm of sample gradient under Static Label Flipping'),
        # ('_Bound_A', 'Norm of sample gradient under Dynamic Label FLipping'),
        # ('_Bound_A_full_batch', 'Disturbance of static label flipping'),
        # ('_Bound_A_full_batch', 'Disturbance of dynamic label flipping'),
        ('_Bound_A', 'Disturbance of static label flipping'),
        ('_Bound_A', 'Disturbance of dynamic label flipping'),
        ('_hetero_list', 'Heterogeneity'),
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name = task_name +  '_A_hetero'

    # fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 19), sharex=True, sharey='row')
    fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 14), sharex=True, sharey='row')
    axes[0][0].set_ylabel('Magnitude', fontsize=FONTSIZE)
    axes[1][0].set_ylabel('Magnitude', fontsize=FONTSIZE)
    axes[0][0].set_ylim(-2, 22)
    axes[1][0].set_ylim(-2, 52)


    for l in range(len(datasets)):
        taskname = task_name + '_' + datasets[l]
        for i in range(len(partition_names)):
            axes[l][i].set_title(partition_names[i][1] + ' (' + datasets[l].upper() + ')', fontsize=FONTSIZE)
            axes[1][i].set_xlabel('iterations', fontsize=FONTSIZE)
            axes[l][i].tick_params(labelsize=FONTSIZE)
            axes[l][i].grid('on')
            for index, (suffix, label) in enumerate(suffix_list):
                color = colors[index]
                marker = markers[index]

                if label == 'Heterogeneity':
                    file_name = 'CSGD_baseline_mean' + suffix
                    file_path = [taskname, 'Centralized_n=10_b=0', partition_names[i][0]]
                elif label == 'Disturbance of static label flipping':
                    file_name = 'CSGD_label_flipping_mean' + suffix
                    file_path = [taskname, 'Centralized_n=10_b=1', partition_names[i][0]]
                elif label == 'Disturbance of dynamic label flipping':
                    file_name = 'CSGD_furthest_label_flipping_mean' + suffix
                    file_path = [taskname, 'Centralized_n=10_b=1', partition_names[i][0]]
                record = load_file_in_cache(file_name, path_list=file_path)
                x_axis = [r*interval for r in range(rounds+1)]
                axes[l][i].plot(x_axis, record, '-', color=color, marker=marker, label=label, markevery=20)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=FONTSIZE)
    leg = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=FONTSIZE, markerscale=3)
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

    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    plt.show()


def draw_mnist(task_name):
    # datasets = ['mnist', 'cifar10']
    dataset = 'mnist'


    suffix_list = [
        # ('_Bound_A', 'Norm of sample gradient under Static Label Flipping'),
        # ('_Bound_A', 'Norm of sample gradient under Dynamic Label FLipping'),
        # ('_Bound_A_full_batch', 'Disturbance of static label flipping'),
        # ('_Bound_A_full_batch', 'Disturbance of dynamic label flipping'),
        ('_Bound_A', 'Disturbance of static label flipping'),
        ('_Bound_A', 'Disturbance of dynamic label flipping'),
        ('_hetero_list', 'Heterogeneity'),
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name = task_name + '_' + dataset + '_A_hetero'

    fig, axes = plt.subplots(1, len(partition_names), figsize=(21, 11), sharex=True, sharey=True)
    axes[0].set_ylabel('Magnitude', fontsize=FONTSIZE)
    axes[0].set_ylim(-2, 22)

    taskname = task_name + '_' + dataset
    for i in range(len(partition_names)):
        axes[i].set_title(partition_names[i][1] + ' (MNIST)', fontsize=FONTSIZE)
        axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
        axes[i].tick_params(labelsize=FONTSIZE)
        axes[i].grid('on')
        for index, (suffix, label) in enumerate(suffix_list):
            color = colors[index]
            marker = markers[index]
            
            if label == 'Heterogeneity':
                file_name = 'CSGD_baseline_mean' + suffix
                file_path = [taskname, 'Centralized_n=10_b=0', partition_names[i][0]]
            elif label == 'Disturbance of static label flipping':
                file_name = 'CSGD_label_flipping_mean' + suffix
                # file_path = [taskname, 'Centralized_n=10_b=1', partition_names[2][0]]
                file_path = [taskname, 'Centralized_n=10_b=1', partition_names[i][0]]
            elif label == 'Disturbance of dynamic label flipping':
                file_name = 'CSGD_furthest_label_flipping_mean' + suffix
                # file_path = [taskname, 'Centralized_n=10_b=1', partition_names[2][0]]
                file_path = [taskname, 'Centralized_n=10_b=1', partition_names[i][0]]
            record = load_file_in_cache(file_name, path_list=file_path)
            x_axis = [r*interval for r in range(rounds+1)]
            axes[i].plot(x_axis, record, '-', color=color, marker=marker, label=label, markevery=20)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=FONTSIZE)
    leg = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=FONTSIZE, markerscale=3)
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

    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_mnist('SR')
    draw('NeuralNetwork')

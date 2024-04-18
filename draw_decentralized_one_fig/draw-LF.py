# python draw_decentralized_one_fig.py --task SR_mnist --graph ER_n=12_b=2_p=0.7_seed=300 --partition LabelSeperation --portrait

import matplotlib.pyplot as plt
import os

import sys
sys.path.append('..')

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path
from matplotlib import rcParams

import argparse
parser = argparse.ArgumentParser(description='Plotter for robust TD')

parser.add_argument('--task', type=str, default='LR_ijcnn1',
    help='task performed (LR_ijcnn1, SR_mnist, etc)')
parser.add_argument('--graph', type=str, default='ER',
    help='name of the graph (ER, TwoCastle)')
parser.add_argument('--partition', type=str, default='TrivalDist',
    help='name of the partition (TrivalDist)')
parser.add_argument('--workspace', type=str, nargs='+', default=[],
    help='workspace')
parser.add_argument('--mark_on_title', type=str, default='',
    help='mark_on_title')
parser.add_argument('--portrait', action='store_true',
    help='use portrait layout or landscape layout')

args = parser.parse_args()

args.task = 'SR_mnist_sorted_by_label'
args.graph = 'Complete_n=10_b=1'
args.partition = 'LabelSeperation'
# args.partition = 'iidPartition'
args.portrait = True

task_name = args.task
partition_name = args.partition
graph_name = args.graph
workspace = args.workspace
mark_on_title = args.mark_on_title
if mark_on_title != '':
    mark_on_title = '_' + mark_on_title

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

def optimal_gap(path, Fmin):
    return [p-Fmin for p in path]

SCALE = 2.5
FONT_SIZE = 14
LINE_STYLE = '-'
# LINE_STYLE = 'v-'

pic_name = 'dec_' + task_name + '_' + graph_name + '_' + partition_name + '_adversariallabelflipping'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

aggregations = [
    # ('meanW', 'Mean (without attack)'),
    # ('geometric_median', 'GeoMed'), 
    # ('median', 'CooMed'),
    # ('Krum', 'Krum'), 
    ('trimmed_mean', 'Tr-Mean'),
    # ('bulyan', 'Bulyan'),
    # ('CC_tau=0.3', 'CC'),
    ('SCClip_tau=0.3', 'SCC'),
    # ('SCClip_tau=0.1', 'SCC'),
    # ('RSA', 'RSA'),
    # ('FABA', 'FABA'), 
    ('IOS', 'IOS'), 
    # ('remove_outliers', 'Cutter'),
    # ('mean', 'mean'), 
    ('meanW', 'Mean'), 
]
attackNames = [
    # ('baseline', 'without attack'), 
    # ('gaussian', 'Gaussian attack'), 
    # ('sign_flipping', 'sign flipping attack'), 
    # ('isolation', 'isolation attack'),
    # ('label_flipping', 'label flipping attack'),
    ('adversarial_label_flipping', 'adversarial label flipping attack')
]

# ====== create axis ======
ax_obj = {
    'loss': {
        attack_code_name: None for attack_code_name, _ in attackNames
    },
    'accuracy': {
        attack_code_name: None for attack_code_name, _ in attackNames
    },
    'consensus': {
        attack_code_name: None for attack_code_name, _ in attackNames
    },
}
if args.portrait:
    fig, ax_matrix = plt.subplots(len(attackNames), 2)
    if len(attackNames) == 1:
        ax_matrix = [ax_matrix]
    for i, (attack_code_name, _) in enumerate(attackNames):
        # ax_obj['loss'][attack_code_name] = ax_matrix[i][0]
        ax_obj['accuracy'][attack_code_name] = ax_matrix[i][0]
        ax_obj['consensus'][attack_code_name] = ax_matrix[i][1]
        
    ax_matrix[-1][0].set_xlabel(r'iteration k', fontsize=FONT_SIZE)
    ax_matrix[-1][1].set_xlabel(r'iteration k', fontsize=FONT_SIZE)
    # 4 attacks: (SCALE*3.5, SCALE*(1*len(attackNames)+1))
    fig.set_size_inches((SCALE*3, SCALE*(1*len(attackNames))))
    plt.subplots_adjust(hspace=0.35, wspace=0.38, top=0.9)
    
    # ====== set title ======
    for attack_code_name, attack_show_name in attackNames:
        # axes_loss = ax_obj['loss'][attack_code_name]
        axes_acc = ax_obj['accuracy'][attack_code_name]
        axes_ce = ax_obj['consensus'][attack_code_name]
        # loss
        # axes_loss.set_title(f'loss-vs-iteration ({attack_code_name})')
        # axes_loss.set_ylabel(r'$f(x^k)$', fontsize=FONT_SIZE)
        # axes_acc.set_title(f'accuracy ({attack_show_name})', fontsize=FONT_SIZE)
        # axes_acc.set_ylabel(r'accuracy', fontsize=FONT_SIZE)
        # consensus error
        # axes_ce.set_title(f'ce-vs-iteration ({attack_code_name})')
        # axes_ce.set_ylabel(r'consensus error', fontsize=FONT_SIZE)
        
        axes_acc.set_title(f'Accuracy', fontsize=FONT_SIZE)
        # axes_acc.set_ylabel(r'accuracy', fontsize=FONT_SIZE)
        # consensus error
        axes_ce.set_title(f'Consensus Error', fontsize=FONT_SIZE)
else:
    fig, ax_matrix = plt.subplots(2, len(attackNames))
    if len(attackNames) == 1:
        ax_matrix = [
            [ax_matrix[0]], [ax_matrix[1]]
        ]        
    for i, (attack_code_name, _) in enumerate(attackNames):
        # ax_obj['loss'][attack_code_name] = ax_matrix[0][i]
        ax_obj['accuracy'][attack_code_name] = ax_matrix[0][i]
        ax_obj['consensus'][attack_code_name] = ax_matrix[1][i]
        
    # ax_matrix[0][0].set_ylabel(r'loss', fontsize=FONT_SIZE)
    ax_matrix[0][0].set_ylabel(r'accuracy', fontsize=FONT_SIZE)
    ax_matrix[1][0].set_ylabel(r'consensus error', fontsize=FONT_SIZE)
    fig.set_size_inches((SCALE*1, SCALE*2.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    
    # ====== set title ======
    # ax_matrix[0][0].set_ylabel(r'$f(x^k)$', fontsize=FONT_SIZE)
    ax_matrix[0][0].set_ylabel(r'accuracy', fontsize=FONT_SIZE)
    ax_matrix[1][0].set_ylabel(r'consensus error', fontsize=FONT_SIZE)
    
    for attack_code_name, attack_show_name in attackNames:
        # axes_loss = ax_obj['loss'][attack_code_name]
        axes_acc = ax_obj['accuracy'][attack_code_name]
        axes_ce = ax_obj['consensus'][attack_code_name]
        # loss
        # axes_loss.set_title(f'{attack_code_name}', fontsize=FONT_SIZE)
        axes_acc.set_title(f'{attack_show_name}', fontsize=FONT_SIZE)
        axes_ce.set_title(f'{attack_show_name}', fontsize=FONT_SIZE)

# ====== set tickets of consensus axes =======
for axes in ax_obj['accuracy'].values():
    axes.grid('on')
    axes.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    # axes.set_xticklabels([])
for axes in ax_obj['consensus'].values():
    # axes.xaxis.set_ticks(fontsize = FONT_SIZE)
    # axes.yaxis.set_ticks(fontsize = FONT_SIZE)
    axes.grid('on')
    axes.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    axes.set_yscale('log')

# fig.suptitle(pic_name, y=0.96, fontsize=FONT_SIZE)
# fig.subplots_adjust(top=1)
    
# ====== plot ======
for attack_code_name, attack_show_name in attackNames:
    for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
        color = colors[agg_index]
        # file_name = 'DSGD_MSG_' + attack_code_name + '_' + agg_code_name \
        #     + '_invSqrtLR' + mark_on_title
        if agg_show_name == 'Mean (without attack)':
            file_name = 'DSGD_MSG_baseline_' + agg_code_name + '_invSqrtLR' + mark_on_title
        else: 
            file_name = 'DSGD_MSG_' + attack_code_name + '_' + agg_code_name \
                        + '_invSqrtLR' + mark_on_title
        file_path = [task_name, graph_name, partition_name] + workspace
        record = load_file_in_cache(file_name, path_list=file_path)
        loss_path = record['loss_path']
        acc_path = record['acc_path']
        # loss_path = optimal_gap(loss_path, Fmin)
        consensus_error_path = record['consensus_error_path']
        # smooth consensus error path
        smoothed_ce_path = [consensus_error_path[0]]
        smooth_p = 0.1
        for ce in consensus_error_path[1:]:
            smoothed_ce_path.append(smoothed_ce_path[-1]*smooth_p
                                    +ce*(1-smooth_p))
        
        x_axis = [r*record['display_interval']
                        for r in range(record['rounds']+1)]
        
        # axes_loss = ax_obj['loss'][attack_code_name]
        axes_acc = ax_obj['accuracy'][attack_code_name]
        axes_ce = ax_obj['consensus'][attack_code_name]
        
        # axes_loss.plot(x_axis, loss_path, LINE_STYLE, color=color, label=agg_show_name)
        axes_acc.plot(x_axis, acc_path, LINE_STYLE, color=color, label=agg_show_name)
        axes_ce.plot(x_axis, smoothed_ce_path, LINE_STYLE, color=color, label=agg_show_name)
        # ce_x_sqrt_k = [ce * k**2 for k, ce in enumerate(consensus_error_path)]
        # axes_ce.plot(x_axis[1:], ce_x_sqrt_k, LINE_STYLE, color=color, label=show_name)

if args.portrait:
    # ax_matrix[-1][0].legend(loc='lower left', bbox_to_anchor=(-0.0,-0.8),
    #         borderaxespad=0., ncol=4, fontsize=FONT_SIZE)
    # ax_matrix[-1][-1].legend(loc='lower left', bbox_to_anchor=(1.1,-0.3),
    #         borderaxespad=0., ncol=1, fontsize=FONT_SIZE)
    ax_matrix[-1][-1].legend(loc='lower left', bbox_to_anchor=(1.1,0.1),
            borderaxespad=0., ncol=1, fontsize=FONT_SIZE)
else:
    ax_matrix[-1][0].legend(loc='lower left', bbox_to_anchor=(0.1,-0.4),
                           borderaxespad=0., ncol=1, fontsize=FONT_SIZE)
# fig.tight_layout()
    
file_dir = os.path.dirname(os.path.abspath(__file__))
dir_png_path = os.path.join(file_dir, 'pic', 'png')
dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

if not os.path.isdir(dir_pdf_path):
    os.makedirs(dir_pdf_path)
if not os.path.isdir(dir_png_path):
    os.makedirs(dir_png_path)

suffix = '_portrait' if args.portrait else ''
pic_png_path = os.path.join(dir_png_path, pic_name + suffix + '.png')
pic_pdf_path = os.path.join(dir_pdf_path, pic_name + suffix + '.pdf')
plt.savefig(pic_png_path, format='png', bbox_inches='tight')
plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
# plt.show()
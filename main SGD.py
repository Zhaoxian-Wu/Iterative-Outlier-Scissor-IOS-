from argsParser import args
import numpy as np

from ByrdLab import FEATURE_TYPE, DEVICE
from ByrdLab.attack import (D_alie, D_gaussian, D_isolation_weight,
                            D_sample_duplicate, D_sign_flipping,
                            D_zero_sum, D_zero_value, D_label_flipping, D_label_random, D_feature_label_random, D_furthest_label_flipping)
from ByrdLab.SingleMachineAlgorithm import SGD
from ByrdLab.graph import CompleteGraph, ErdosRenyi, OctopusGraph, TwoCastle
from ByrdLab.library.cache_io import dump_file_in_cache, dump_model_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist, cifar10, mnist_sorted_by_labels
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition)
from ByrdLab.library.tool import log
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask, softmax_regression_loss
from ByrdLab.tasks.leastSquare import LeastSquareToySet, LeastSquareToyTask
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask

args.graph = 'CompleteGraph'
# args.attack = 'label_flipping'
# args.lr_ctrl = 'ladder'
# args.data_partition = 'noniid'
# args.aggregation = 'ios' 

# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
if args.graph == 'CompleteGraph':
    graph = CompleteGraph(node_size=1, byzantine_size=0)
else:
    assert False, 'unknown graph'
    
# If the Byzantine nodes present, they will do the attack even they do not change their model 
# over time. The presence of Byzantine nodes is an attack
if args.attack == 'none':
    graph = graph.honest_subgraph()

# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------
# data_package = ijcnn()
# task = LogisticRegressionTask(data_package)

# dataset = ToySet(set_size=500, dimension=5, fix_seed=True)

# data_package = mnist()
data_package = mnist_sorted_by_labels()
task = softmaxRegressionTask(data_package)

# data_package = fashionmnist()
# task = softmaxRegressionTask(data_package)

# data_package = cifar10()
# task = NeuralNetworkTask(data_package, batch_size=32)

# w_star = torch.tensor([1], dtype=FEATURE_TYPE)
# data_package = LeastSquareToySet(set_size=2000, dimension=1, w_star=w_star, noise=0, fix_seed=True)
# data_package = LeastSquareToySet(set_size=100, dimension=1, noise=0, fix_seed=True)
# task = LeastSquareToyTask(data_package)

# task.super_params['display_interval'] = 20000
# task.super_params['rounds'] = 10
# task.super_params['lr'] = 0.004
# task.super_params['lr'] = 0.1
# task.initialize_fn = None
# ===========================================


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
    # super_params = task.super_params
    # total_iterations = super_params['rounds']*super_params['display_interval']
    # lr_ctrl = one_over_sqrt_k_lr(total_iteration=total_iterations,
    #                              a=math.sqrt(1001), b=1000)
elif args.lr_ctrl == 'ladder':
    decreasing_iter_ls = [5000, 10000, 15000]
    proportion_ls = [0.3, 0.2, 0.1]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'

# ===========================================
    
    
# -------------------------------------------
# define data partition
# -------------------------------------------
partition_cls = TrivalPartition

# ===========================================
    

# -------------------------------------------
# define aggregation
# -------------------------------------------

# ===========================================
    
# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'label_flipping':
    attack = D_label_flipping(graph)
elif args.attack == 'label_random':
    attack = D_label_random(graph)
elif args.attack == 'feature_label_random':
    attack = D_feature_label_random(graph)
elif args.attack == 'furthest_label_flipping':
    attack = D_furthest_label_flipping(graph)
elif args.attack == 'sign_flipping':
    attack = D_sign_flipping(graph)
elif args.attack == 'gaussian':
    attack = D_gaussian(graph)
elif args.attack == 'isolation':
    # D_isolation(graph),
    attack = D_isolation_weight(graph)
elif args.attack == 'sample_duplicate':
    attack = D_sample_duplicate(graph)
elif args.attack == 'zero_sum':
    attack = D_zero_sum(graph)
elif args.attack == 'zero_value':
    attack = D_zero_value(graph)
elif args.attack == 'alie':
    attack = D_alie(graph)
    # D_alie(graph, scale=2)
else:
    assert False, 'unknown attack'

if args.attack == 'none':
    attack_name = 'baseline'
else:
    attack_name = attack.name

# ===========================================

# workspace = []
workspace = ['best']
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record
step_agg = args.step_agg

# initilize optimizer
env = SGD(graph=graph, attack=attack, step_agg = step_agg,
           weight_decay=task.weight_decay, data_package=task.data_package,
           model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
           initialize_fn=task.initialize_fn,
           get_train_iter=task.get_train_iter,
           get_test_iter=task.get_test_iter,
           partition_cls=partition_cls, lr_ctrl=lr_ctrl,
           fix_seed=fix_seed, seed=seed,
           **task.super_params)


title = '{}_{}'.format(env.name, attack_name)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title
title = title + '.pt'

# data_package = task.data_package
super_params = task.super_params

# print the running information
print('=========================================================')
print('[Task] ' + task.name + ': ' + title)
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', task.data_package.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, 'None', attack_name))
print('{:12s} lr={} lr_ctrl={}, weight_decay={}'.format(
    '[Optimizer]', super_params['lr'], env.lr_ctrl.name, task.weight_decay))
print('{:12s} graph={}, honest_size={}, byzantine_size={}'.format(
    '[Graph]', graph.name, graph.honest_size, graph.byzantine_size))
print('{:12s} rounds={}, display_interval={}, total iterations={}'.format(
    '[Running]', env.rounds, env.display_interval, env.total_iterations))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', seed, fix_seed))
print('{:12s} record_in_file={}'.format('[System]', record_in_file))
print('-------------------------------------------')

log('[Start Running]')
data_size = len(data_package.train_set)
num_classes = data_package.num_classes
len_U = data_size * num_classes 
loss_clean_model = np.zeros(len_U)

model = env.run()
path_best_list = [task.name, graph.name, 'TrivalPartition'] + workspace
dump_file_in_cache('SGD_momentum_invSqrtLR.pt', model, path_list=path_best_list)

loss_fn = softmax_regression_loss
for i in range(data_size):
    feature = data_package.train_set.features[i]
    target = data_package.train_set.targets[i]
    feature = feature.to(DEVICE)
    target = target.to(DEVICE)
    prediction = model(feature)
    for k in range(num_classes):
        target_flipped = (target + k) % num_classes
        loss_clean_model[i + k * data_size] = loss_fn(prediction, target_flipped)

path_best_list = [task.name, graph.name, 'TrivalPartition'] + workspace
dump_file_in_cache('loss_best_model_on_each_sample', loss_clean_model, path_list=path_best_list)

# record = {
#     'dataset': data_package.name,
#     'dataset_size': len(data_package.train_set),
#     'dataset_feature_dimension': data_package.feature_dimension,
#     'lr': super_params['lr'],
#     'weight_decay': task.weight_decay,
#     'honest_size': graph.honest_size,
#     'byzantine_size': graph.byzantine_size,
#     'rounds': env.rounds,
#     'display_interval': env.display_interval,
#     'total_iterations': env.total_iterations,
#     'loss_path': loss_path,
#     'acc_path': acc_path,
#     'consensus_error_path': consensus_error_path,
#     'fix_seed': fix_seed,
#     'seed': seed,
#     'graph': graph,
# }

# if record_in_file:
#     path_list = [task.name, graph.name, env.partition_name] + workspace
#     dump_file_in_cache(title, record, path_list=path_list)
print('-------------------------------------------')


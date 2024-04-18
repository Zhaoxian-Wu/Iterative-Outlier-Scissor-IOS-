import copy
import itertools
import math
import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans
import numpy as np
import torch
from ByrdLab import FEATURE_TYPE, DEVICE
from scipy import stats

from ByrdLab.library.tool import MH_rule, flatten_list, unflatten_vector

class CentraliedAggregation():
    def __init__(self, name, honest_nodes, byzantine_nodes):
        self.name = name
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
    def run(self, messages):
        raise NotImplementedError


def mean(wList):
    return torch.mean(wList, dim=0)


def geometric_median(wList, max_iter=80, err=1e-5):
    guess = torch.mean(wList, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack(
            [w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= err:
            break
    return guess


def medoid_index(wList):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    dist_sum = dist.sum(dim=1)
    return dist_sum.argmax()


def medoid(wList):
    return wList[medoid_index(wList)]


def Krum_index(wList, byzantine_size):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()**2
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    # The distance from any node to itself must be 0.00, so we add 1 here
    k = node_size - byzantine_size - 2 + 1
    topv, _ = dist.topk(k=k, dim=1)
    scores = topv.sum(dim=1)
    return scores.argmax()


def Krum(wList, byzantine_size):
    index = Krum_index(wList, byzantine_size)
    return wList[index]


def mKrum(wList, byzantine_size, m=1):
    remain = wList
    result = torch.zeros_like(wList[0], dtype=FEATURE_TYPE)
    for _ in range(m):
        res_index = Krum_index(remain, byzantine_size)
        result += remain[res_index]
        remain = remain[torch.arange(remain.size(0)) != res_index]
    return result / m


def median(wList):
    return wList.median(dim=0)[0]


def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
      data Indexable (including ability to query length) containing the elements
    Returns:
      Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (data[i], data[j])


def brute_selection(gradients, f, **kwargs):
    """ Brute rule. 
    brute is also called minimum diameter averaging (MDA)
    The code comes from:
    https://github.com/LPD-EPFL/Garfield/blob/master/pytorch_impl/libs/aggregators/brute.py#L32

    Args:
      gradients Non-empty list of gradients to aggregate
      f         Number of Byzantine gradients to tolerate
      ...       Ignored keyword-arguments
    Returns:
      Selection index set
    """
    n = len(gradients)
    # Compute all pairwise distances
    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
        distances[i] = gradients[x].sub(gradients[y]).norm().item()
    # Select the set of smallest diameter
    sel_iset = None
    sel_diam = None
    for cur_iset in itertools.combinations(range(n), n - f):
        # Compute the current diameter (max of pairwise distances)
        cur_diam = 0.
        for x, y in pairwise(cur_iset):
            # Get distance between these two gradients ("magic" formula valid since x < y)
            cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
            # Check finite distance (non-Byzantine gradient must only contain finite coordinates), drop set if non-finite
            if not math.isfinite(cur_dist):
                break
            # Check if new maximum
            if cur_dist > cur_diam:
                cur_diam = cur_dist
        else:
            # Check if new selected diameter
            if sel_iset is None or cur_diam < sel_diam:
                sel_iset = cur_iset
                sel_diam = cur_diam
    # Return the selected gradients
    assert sel_iset is not None, "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
    return sel_iset


def brute(gradients, byzantine_size, **kwargs):
    """ Brute rule.
    Args:
      gradients Non-empty list of gradients to aggregate
      f         Number of Byzantine gradients to tolerate
      ...       Ignored keyword-arguments
    Returns:
      Aggregated gradient
    """
    sel_iset = brute_selection(gradients, byzantine_size, **kwargs)
    return sum(gradients[i] for i in sel_iset).div_(len(gradients) - byzantine_size)


# def trimmed_mean(wList, byzantine_size):
#     node_size = wList.size(0)
#     proportion_to_cut = byzantine_size / node_size
#     tm_np = stats.trim_mean(wList, proportion_to_cut, axis=0)
#     return torch.from_numpy(tm_np)


def trimmed_mean(wList, byzantine_size):
    # 将张量按某个维度排序
    sorted_wList, _ = torch.sort(wList, dim=0)
    
    # 对排序后的张量进行修剪
    if byzantine_size == 0:
        trimmed_data = sorted_wList
    elif byzantine_size > 0:
        trimmed_data = sorted_wList[byzantine_size:-byzantine_size, :]
    else:
        assert False, 'Byzantine size should be equal or larger than 0!'
    
    # 计算修剪后的均值
    if trimmed_data.nelement() > 0:
        tm = torch.mean(trimmed_data, dim=0)
    else:
        tm = 0

    return tm


def remove_outliers(wList, byzantine_size):
    mean = torch.mean(wList, dim=0)
    # remove the largest 'byzantine_size' model
    distances = torch.tensor([
        -torch.norm(model - mean) for model in wList
    ])
    node_size = wList.size(0)
    remain_cnt = node_size - byzantine_size
    (_, remove_index) = torch.topk(distances, k=remain_cnt)
    return wList[remove_index].mean(dim=0)


def faba(wList, byzantine_size):
    remain = wList
    for _ in range(byzantine_size):
        mean = remain.mean(dim=0)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
            torch.norm(model - mean) for model in remain
        ])
        remove_index = distances.argmax()
        remain = remain[torch.arange(remain.size(0)) != remove_index]
    return remain.mean(dim=0)


def bulyan(wList, byzantine_size):
    remain = wList
    selected_ls = []
    node_size = wList.size(0)
    selection_size = node_size-2*byzantine_size
    for _ in range(selection_size):
        res_index = Krum_index(remain, byzantine_size)
        selected_ls.append(remain[res_index])
        remain = remain[torch.arange(remain.size(0)) != res_index]
    selection = torch.stack(selected_ls)
    m = median(selection)
    dist = -(selection - m).abs()
    indices = dist.topk(k=selection_size-2*byzantine_size, dim=0)[1]
    if len(wList.size()) == 1:
        result = selection[indices].mean()
    else:
        result = torch.stack([
            selection[indices[:, d], d].mean() for d in range(wList.size(1))])
    return result


class C_mean(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='mean', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)

    def run(self, messages):
        return torch.mean(messages, dim=0)


class C_trimmed_mean(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='trimmed_mean', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)

    def run(self, messages):
        # 将张量按某个维度排序
        sorted_wList, _ = torch.sort(messages, dim=0)
        byzantine_size = len(self.byzantine_nodes)

        # 对排序后的张量进行修剪
        if byzantine_size == 0:
            trimmed_data = sorted_wList
        elif byzantine_size > 0:
            trimmed_data = sorted_wList[byzantine_size:-byzantine_size, :]
        else:
            assert False, 'Byzantine size should be equal or larger than 0!'
        
        # 计算修剪后的均值
        if trimmed_data.nelement() > 0:
            tm = torch.mean(trimmed_data, dim=0)
        else:
            tm = 0
    
        return tm


class C_faba(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='faba', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)

    def run(self, messages):
        remain = messages
        byzantine_size = len(self.byzantine_nodes)

        for _ in range(byzantine_size):
            mean = torch.mean(remain, dim=0)
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain
            ])
            remove_index = distances.argmax()
            remain = remain[torch.arange(remain.size(0)) != remove_index]
        return remain.mean(dim=0)


class C_centered_clipping(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes, threshold=10):
        super().__init__(name=f'CC_tau={threshold}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
        self.memory = None
        self.threshold = threshold

    def run(self, messages):
        if self.memory == None:
            self.memory = torch.zeros_like(messages[0])
        
        diff = torch.zeros_like(self.memory)
        for n in self.honest_nodes + self.byzantine_nodes:
            grad = messages[n]
            norm = (grad - self.memory).norm()
            if norm > self.threshold:
                diff += self.threshold * (grad - self.memory) / norm
            else:
                diff += grad - self.memory
        diff /= (len(self.honest_nodes) + len(self.byzantine_nodes))
        self.memory = self.memory + diff
        return self.memory
    

class C_LFighter(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__('LFighter', honest_nodes, byzantine_nodes)

    def clusters_dissimilarity(self, clusters):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def run(self, messages):
        m = len(messages)
        dw = [[] for _ in range(m)]
        for i in range(m):
            dw[i].append(messages[i][-2].cpu().data.numpy())
        dw = np.asarray(dw)
        dw = np.squeeze(dw)
        norms = np.linalg.norm(dw, axis=-1)
        memory = np.sum(norms, axis=0)
        max_two_freq_classes = memory.argsort()[-2:]
        # print('Potential source and target classes:', max_two_freq_classes)
        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1
        
        remain_worker_grad = []
        for i, l in enumerate(labels):
            if l == good_cl:
                remain_worker_grad.append(messages[i])
        
        mean = [torch.zeros_like(para, requires_grad=False) for para in messages[0]]
        for grad in remain_worker_grad:
            for i, g in enumerate(grad):
                mean[i].add_(g, alpha=1 / len(remain_worker_grad))

        return mean


class DecentralizedAggregation():
    def __init__(self, name, graph, superparameter={}):
        self.name = name
        self.graph = graph
        # some aggregation may need global state of the system
        # this global state can be store in the global state dictionary
        self.global_state = {}
        self.required_info = set()
        self.superparam = superparameter

    def run(self, local_models, node):
        raise NotImplementedError

    def all_neighbor_models(self, local_models, node):
        return local_models[self.graph.neighbors[node]]

    def neighbor_models_and_itself(self, local_model, node):
        li = list(self.graph.neighbors[node]) + [node]
        return local_model[li]


class D_mean(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='mean', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return neighbor_models.mean(axis=0)



class D_no_communication(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='no_communication',
                                                 graph=graph)

    def run(self, local_models, node):
        return local_models[node]


class D_meanW(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='meanW', graph=graph)
        self.W = MH_rule(graph)
        self.W = self.W.to(DEVICE)

    def run(self, local_models, node):
        return torch.tensordot(self.W[node], local_models, dims=1)


class D_median(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='median', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return median(neighbor_models)


class D_geometric_median(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='geometric_median',
                                                 graph=graph)

    def run(self,  local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return geometric_median(neighbor_models)


class D_Krum(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='Krum', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return Krum(neighbor_models, byzantine_size=self.graph.byzantine_sizes[node])


class D_mKrum(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='mKrum', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        m = self.graph.neighbor_sizes[node] - \
            2*self.graph.byzantine_sizes[node]-3
        return mKrum(neighbor_models,
                     byzantine_size=self.graph.byzantine_sizes[node],
                     m=m)


class D_trimmed_mean(DecentralizedAggregation):
    def __init__(self, graph, exact_byz_cnt=True, byz_cnt=-1):
        if exact_byz_cnt:
            name = 'trimmed_mean'
        else:
            if byz_cnt < 0:
                name = 'trimmed_mean_max'
            else:
                name = f'trimmed_mean_{byz_cnt}'
        super().__init__(name=name, graph=graph)

        self.exact_byz_cnt = exact_byz_cnt
        self.Byz_cnt = byz_cnt

    def run(self, local_models, node):
        if self.exact_byz_cnt:
            estimate_byz_cnt = self.graph.byzantine_sizes[node]
        else:
            if self.Byz_cnt < 0:
                estimate_byz_cnt = max(self.graph.byzantine_sizes)
            else:
                estimate_byz_cnt = self.Byz_cnt
        neighbor_models = self.all_neighbor_models(local_models, node)
        tm = trimmed_mean(neighbor_models, byzantine_size=estimate_byz_cnt)
        trimmed_neighbor_size = len(neighbor_models) - 2 * estimate_byz_cnt
        local_model = local_models[node]
        return (tm * trimmed_neighbor_size + local_model) / (trimmed_neighbor_size + 1)


class D_remove_outliers(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='remove_outliers',
                                                graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        local_model = local_models[node]
        rm = remove_outliers(neighbor_models,
                             byzantine_size=self.graph.byzantine_sizes[node])
        neighbor_size = len(neighbor_models)
        res = (rm * neighbor_size + local_model) / (neighbor_size + 1)
        return res


class D_faba(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='FABA', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        local_model = local_models[node]
        agg = faba(neighbor_models,
                   byzantine_size=self.graph.byzantine_sizes[node])
        hoenst_neighbor_size = self.graph.honest_sizes[node]
        res = (agg * hoenst_neighbor_size + local_model) / \
            (hoenst_neighbor_size + 1)
        return res


class D_ios(DecentralizedAggregation):
    def __init__(self, graph, exact_byz_cnt=True, byz_cnt=-1):
        if exact_byz_cnt:
            name = 'IOS'
        else:
            if byz_cnt < 0:
                name = 'IOS_max'
            else:
                name = f'IOS_{byz_cnt}'
        super().__init__(name=name, graph=graph)
        node_size = graph.number_of_nodes()
        self.W = torch.eye(node_size, dtype=FEATURE_TYPE).to(DEVICE)
        self.exact_byz_cnt = exact_byz_cnt
        self.Byz_cnt = byz_cnt
        for i in range(node_size):
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i):
                    continue
                i_n = self.graph.neighbor_sizes[i] + 1
                j_n = self.graph.neighbor_sizes[j] + 1
                self.W[i][j] = 1 / max(i_n, j_n)
                # self.W[i][j] = 1 / i_n
                self.W[i][i] -= self.W[i][j]

    def run(self, local_models, node):
        remain_models = local_models[self.graph.neighbors[node]]
        remain_weight = self.W[node][self.graph.neighbors[node]]
        if self.exact_byz_cnt:
            estimate_byz_cnt = self.graph.byzantine_sizes[node]
        else:
            if self.Byz_cnt < 0:
                estimate_byz_cnt = max(self.graph.byzantine_sizes)
            else:
                estimate_byz_cnt = self.Byz_cnt
        for _ in range(estimate_byz_cnt):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean += self.W[node][node]*local_models[node]
            mean /= remain_weight.sum() + self.W[node][node]
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res += self.W[node][node]*local_models[node]
        res /= remain_weight.sum() + self.W[node][node]
        return res


class D_ios_equal_neigbor_weight(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='IOS_equal_neigbor_weight', graph=graph)
        node_size = graph.number_of_nodes()
        self.W = torch.eye(node_size, dtype=FEATURE_TYPE)
        max_degree = -1
        for i in range(node_size):
            if self.graph.neighbor_sizes[i] > max_degree:
                max_degree = self.graph.neighbor_sizes[i] + 1
        for i in range(node_size):
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i):
                    continue
                self.W[i][j] = 1 / max_degree
                # self.W[i][j] = 1 / i_n
                self.W[i][i] -= self.W[i][j]

    def run(self, local_models, node):
        remain_models = local_models[self.graph.neighbors[node]]
        remain_weight = self.W[node][self.graph.neighbors[node]]
        for _ in range(self.graph.byzantine_sizes[node]):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean += self.W[node][node]*local_models[node]
            mean /= remain_weight.sum() + self.W[node][node]
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res += self.W[node][node]*local_models[node]
        res /= remain_weight.sum() + self.W[node][node]

        # FABA
        # neighbor_models = self.all_neighbor_models(local_models, node)
        # local_model = local_models[node]
        # remain = neighbor_models
        # for _ in range(self.graph.byzantine_sizes[node]):
        #     mean = remain.mean(dim=0)
        #     # remove the largest 'byzantine_size' model
        #     distances = torch.tensor([
        #         torch.norm(model - mean) for model in remain
        #     ])
        #     remove_index = distances.argmax()
        #     remain = remain[torch.arange(remain.size(0)) != remove_index]
        # agg = remain.mean(dim=0)
        # hoenst_neighbor_size = self.graph.honest_sizes[node]
        # res2 = (agg * hoenst_neighbor_size + local_model) / (hoenst_neighbor_size + 1)
        return res


class D_brute(DecentralizedAggregation):
    def __init__(self, graph):
        self.byzantine_sizes = graph.byzantine_sizes
        super().__init__(name='Brute', graph=graph)
    def run(self, local_models, node):
        local_model = local_models[node]
        agg = brute(local_model, byzantine_size=self.byzantine_sizes[node])
        return agg
        

class D_bulyan(DecentralizedAggregation):
    def __init__(self, graph):
        self.byzantine_sizes = graph.byzantine_sizes
        super().__init__(name='Bulyan', graph=graph)

    def run(self, local_models, node):
        local_model = local_models[node]
        agg = bulyan(local_model, byzantine_size=self.byzantine_sizes[node])
        return agg


class D_centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold=10):
        super().__init__(name=f'CC_tau={threshold}', graph=graph)
        self.memory = None
        self.threshold = threshold

    def run(self, local_models, node):
        if self.memory == None:
            self.memory = torch.zeros_like(local_models)
        
        diff = torch.zeros_like(self.memory[node])
        for n in self.graph.neighbors[node] + [node]:
            model = local_models[n]
            norm = (model - self.memory[node]).norm()
            if norm > self.threshold:
                diff += self.threshold * (model - self.memory[node]) / norm
            else:
                diff += model - self.memory[node]
        diff /= (self.graph.neighbor_sizes[node] + 1)
        self.memory[node] = self.memory[node] + diff
        return self.memory[node]


class D_self_centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold_selection='estimation', threshold=10):
        if threshold_selection == 'estimation':
            name = 'SCClip'
        elif threshold_selection == 'true':
            name = 'SCClip_T'
        elif threshold_selection == 'parameter':
            name = f'SCClip_tau={threshold}'
        else:
            raise ValueError('invalid threshold setting')
        super().__init__(name=name, graph=graph)
        self.W = MH_rule(graph)
        self.threshold = threshold
        self.threshold_selection = threshold_selection

    def get_threshold_estimate(self, local_models, node):
        # find the bottom-(honest-size) weights as the estimated threshold
        local_model = local_models[node]
        node_size = local_models.size(0)
        norm_list = torch.tensor([
            -(local_models[n]-local_model).norm()
            if n in self.graph.neighbors[node] and n != node else 1
            for n in range(node_size)
        ])

        honest_size = self.graph.honest_sizes[node]
        _, bottom_index = norm_list.topk(k=honest_size)
        top_index = [
            n for n in self.graph.neighbors[node]
            if n not in bottom_index and n != node
        ]
        weighted_avg_norm = sum([
            self.W[node][n]*norm_list[n] for n in bottom_index
        ])
        cum_weight = sum([
            self.W[node][n] for n in top_index
        ])
        return torch.sqrt(weighted_avg_norm/cum_weight)

    def get_true_threshold(self, local_models, node):
        # find the bottom-(honest-size) weights as the estimated threshold
        local_model = local_models[node]

        weighted_avg_norm = sum([
            self.W[node][n]*(local_models[n]-local_model).norm()**2
            for n in self.graph.honest_neighbors[node]
        ])
        cum_weight = sum([
            self.W[node][n] for n in self.graph.byzantine_neighbors[node]
        ])
        return torch.sqrt(weighted_avg_norm/cum_weight)

    def run(self, local_models, node):
        if self.threshold_selection == 'estimation':
            threshold = self.get_threshold(local_models, node)
        elif self.threshold_selection == 'true':
            threshold = self.get_true_threshold(local_models, node)
        elif self.threshold_selection == 'parameter':
            threshold = self.threshold
        else:
            raise ValueError('invalid threshold setting')
        local_model = local_models[node]
        cum_diff = torch.zeros_like(local_model)
        for n in self.graph.neighbors[node]:
            model = local_models[n]
            diff = model - local_model
            norm = diff.norm()
            weight = self.W[node][n]
            if norm > threshold:
                cum_diff += weight * threshold * diff / norm
            else:
                cum_diff += weight * diff
        return local_model + cum_diff
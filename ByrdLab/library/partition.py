from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.dataset import DataPackage, StackedDataSet
import numpy as np
import matplotlib.pyplot as plt

class Partition():
    def __init__(self, name, partition, rng_pack: RngPackage=RngPackage()):
        self.name = name
        self.partition = partition
        self.rng_pack = rng_pack
    def get_subsets(self, dataset):
        '''
        return all subsets of dataset
        ---------------------------------------
        TODO: the partition of data depends on the specific structure
              of dataset.
              In the version, dataset has the structure that all features
              and targets are stacked in tensors. For other datasets with
              different structures, another type of `get_subsets` shoule
              be implemented.
        '''
        raise NotImplementedError
    def __getitem__(self, i):
        return self.partition[i]
    def __len__(self):
        return len(self.partition)
    
    
class HorizotalPartition(Partition):
    def __init__(self, name, partition):
        self.partition = partition
        super().__init__(name, partition)
    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset[p][0], targets=dataset[p][1])
            for i, p in enumerate(self.partition)
        ]
        
    
class EmptyPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        partition = [[] for _ in range(node_cnt)]
        super().__init__('EmptyPartition', partition)
    
    
class TrivalPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(node*len(dataset)) // node_cnt 
                      for node in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [list(range(seperation[i], seperation[i+1]))
                                for i in range(node_cnt)]
        super().__init__('TrivalDist', partition)

class DirichletPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage(), alpha=0.1, min_size=10):
        self.dataset = dataset
        self.node_cnt = node_cnt
        self.alpha = alpha
        self.min_size = min_size
        # self.min_size = len(dataset) // node_cnt
        self.rng_pack = rng_pack

        # Get the label set and label number
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)

        # deal with the situation that class idx don't
        # locate in consecutive integers starting from zeros
        self.class_idx_dict = {
            label: idx for idx, label in enumerate(self.class_set)}

        partition = self.non_iid_dirichlet()
        super(DirichletPartition, self).__init__(f'DirichletPartition_alpha={alpha}', partition)
        self.data_distribution = self.get_data_distribution()

    def non_iid_dirichlet(self):
        """Partition dataset into multiple clients following the Dirichlet process.

        Key parameters:
            alpha (float): The parameter for Dirichlet process simulation.
            min_size (int): The minimum number of data size of a client.

        Return:
            list[list[]]: The partitioned data.
        """
        np.random.seed(seed=1)

        current_min_size = 0
        data_size = len(self.dataset)

        all_index = [[] for _ in range(self.class_cnt)]
        for i, (_, label) in enumerate(self.dataset):
            # get indexes for all data with current label i at index i in all_index
            label = self.class_idx_dict[label.item()]
            all_index[label].append(i)

        partition = [[] for _ in range(self.node_cnt)]
        while current_min_size < self.min_size:
            partition = [[] for _ in range(self.node_cnt)]
            for k in range(self.class_cnt):
                idx_k = all_index[k]
                self.rng_pack.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.node_cnt))
                # using the proportions from dirichlet, only select those nodes having data amount less than average
                proportions = np.array(
                    [p * (len(idx_j) < data_size / self.node_cnt) for p, idx_j in zip(proportions, partition)])
                # scale proportions
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                partition = [idx_j + idx.tolist() for idx_j, idx in zip(partition, np.split(idx_k, proportions))]
                current_min_size = min([len(idx_j) for idx_j in partition])
        return partition
    
    def get_data_distribution(self):
        data_distribution = []
        for x in range(self.node_cnt):
            class_cnt_dict = {label: 0 for label in self.class_set}
            for j in self.partition[x]:
                class_cnt_dict[self.dataset[j][1].item()] += 1
            data_distribution.append(class_cnt_dict)
        return data_distribution
    
    def draw_data_distribution(self):
        """
        Draw data distributions for all nodes,
        showing the distribution of data categories for each node through cumulative bar charts.
        """
        labels = [i for i in range(1, len(self.partition) + 1)]
        class_list = sorted(list(set([label.item() for _, label in self.dataset])))
        class_cnt = len(class_list)
        data = [[] for _ in range(class_cnt)]

        for j in class_list:
            data[j] = np.array([x[j] if x.get(j) else 0 for x in self.data_distribution])
        sum_data = sum(data)
        y_max = max(sum_data)
        x = range(len(labels))
        width = 0.35

        # Initialize bottom_y element 0
        bottom_y = np.array([0] * len(labels))

        fig, ax = plt.subplots()
        for i, y in enumerate(data):
            ax.bar(x, y, width, bottom=bottom_y, label=class_list[i])
            bottom_y = bottom_y + y

        # Add Legend
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

        for a, b, i in zip(x, sum_data, range(len(x))):
            plt.text(a, int(b * 1.03), "%d" % sum_data[i], ha='center')

        # Setting the title and axis labels
        ax.set_title(self.name + ' data distribution')
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Sample number')
        plt.xticks(x)

        # Adjust chart layout to prevent annotations from obscuring chart content
        plt.tight_layout()

        plt.grid(True, linestyle=':', alpha=0.6)
        # Adjust the length of the vertical coordinate
        plt.ylim(0, int(y_max * 1.1))

        # show picture
        plt.savefig(f'dirichlet_alpha={self.alpha}.pdf')
        plt.show()


class DirichletIiiPartition(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=100)


class DirichletMildPartition(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=1)


class DirichletNoniidPartition(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=0.01)

class DirichletPartition_0(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=1000)

class DirichletPartition_a(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=100)


class DirichletPartition_b(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=10)


class DirichletPartition_c(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=1)

class DirichletPartition_d(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=1e-1, min_size=3000)


class DirichletPartition_e(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=1e-2, min_size=4000)

class DirichletPartition_f(DirichletPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        super().__init__(dataset, node_cnt, rng_pack, alpha=1e-3, min_size=4000)

class iidPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        indexes = list(range(len(dataset)))
        rng_pack.random.shuffle(indexes)
        sep = [(i*len(dataset)) // node_cnt for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [[indexes[i] for i in range(sep[node], sep[node+1])]
                                for node in range(node_cnt)]
        super().__init__('iidPartition', partition)

class SharedData(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        partition = [list(range(len(dataset)))] * node_cnt
        super().__init__('SharedData', partition)
        
class LabelSeperation(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw):
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)
        self.node_cnt = node_cnt
        self.dataset = dataset
        # deal with the situation that class idx don't
        # locate in consecutive integers starting from zeros
        self.class_idx_dict = {
            label: idx for idx, label in enumerate(self.class_set)}
        
        if self.class_cnt < node_cnt:
            partition = self.partition_with_adaquate_nodes()
        else:
            partition = self.partition_with_adaquate_classes()
        super().__init__('LabelSeperation', partition)
        
    def partition_with_adaquate_classes(self):
        '''
        class_cnt >= node_cnt
        some nodes possess several classes
        '''
        partition = [[] for _ in range(self.node_cnt)]
        for data_idx, (_, label) in enumerate(self.dataset):
            node_idx = self.class_idx_dict[label.item()] % self.node_cnt
            partition[node_idx].append(data_idx)
        return partition
    
    def partition_with_adaquate_nodes(self):
        '''
        class_cnt < node_cnt
        some classes are allocated on different workers
        '''
        class_cnt = self.class_cnt
        node_cnt = self.node_cnt
        dataset = self.dataset
        
        # divide the nodes into `class_cnt` groups
        group_boundary = [(group_idx*node_cnt) // class_cnt 
                            for group_idx in range(class_cnt)]
        # when a data is going to be allocated to `group_idx`-th groups,
        # it'll be allocated to `insert_node_ptrs[group_idx]`-th node
        # then `insert_node_ptrs[group_idx]` increases by 1
        insert_node_ptrs = group_boundary.copy()
        group_boundary.append(node_cnt)
        # [e.g] 
        # class_cnt = 5
        # node_cnt = 8
        # group_boundary = [0, 1, 3, 4, 6, 8]
        # divide 8 nodes into 5 groups by
        # 0 | 1 | 2 3 | 4 5 | 6 7 |
        # where the vertical line represent the corresponding `group_boundary`
        # this means
        # class 0 on worker 0
        # class 1 on worker 1
        # class 2 on worker 2, 3
        # class 3 on worker 4, 5
        # class 4 on worker 6, 7
        
        partition = [[] for _ in range(node_cnt)]
        for data_idx, (_, label) in enumerate(dataset):
            # determine which group the data belongs to
            group_idx = self.class_idx_dict[label.item()]
            node_idx = insert_node_ptrs[group_idx]
            partition[node_idx].append(data_idx)
            # `insert_node_ptrs[group_idx]` increases by 1
            if insert_node_ptrs[group_idx] + 1 < group_boundary[group_idx+1]:
                insert_node_ptrs[group_idx] += 1
            else:
                insert_node_ptrs[group_idx] = group_boundary[group_idx]
        return partition

# class LabelSeperation(HorizotalPartition):
#     def __init__(self, dataset, node_cnt, non_iid_degree=2, *args, **kw):
#         partition = [[] for _ in range(node_cnt)]
#         flags = [0]*10
#         # aux = [[] for _ in range(node_cnt)]
#         for i, (_, label) in enumerate(dataset):
#             # if flags[label] != (non_iid_degree-1):
#             if flags[label] != 1:
#                 partition[(label+flags[label]) % node_cnt].append(i)
#                 # aux[(label + flags[label]) % node_cnt].append(label.cpu().numpy().tolist())
#                 flags[label] += 1
#             else:
#                 # partition[(label+non_iid_degree-1) % node_cnt].append(i)
#                 partition[(label+ 1 ) % node_cnt].append(i)
#                 # aux[(label + non_iid_degree - 1) % node_cnt].append(label.cpu().numpy().tolist())
#                 flags[label] = 0
#         super(LabelSeperation, self).__init__('LabelSeperation', partition)
        
# class LabelSeperation(HorizotalPartition):
#     def __init__(self, dataset, node_cnt, *args, **kw):
#         partition = [[] for _ in range(node_cnt)]
#         for i, (_, label) in enumerate(dataset):
#             partition[label % node_cnt].append(i)
#         super(LabelSeperation, self).__init__('LabelSeperation', partition)

        
class VerticalPartition(Partition):
    def __init__(self, dataset: StackedDataSet, 
                 node_cnt: int, *args, **kw) -> None:
        feature_dimension = dataset.feature_dimension
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(i*feature_dimension) // node_cnt
                      for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [list(range(seperation[i], seperation[i+1]))
                                for i in range(node_cnt)]
        super().__init__('TrivalDist', partition)
    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset.features[:, p],
                           targets=dataset.targets)
            for i, p in enumerate(self.partition)
        ]
    
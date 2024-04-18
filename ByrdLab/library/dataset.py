import ByrdLab
from ByrdLab.library.RandomNumberGenerator import RngPackage, torch_rng
import random
import re
import os
import torch

from ByrdLab import FEATURE_TYPE, TARGET_TYPE
from .cache_io import isfile_in_cache, load_file_in_cache, dump_file_in_cache

def load_SVM_dataset(dataset_file, set_size, dimension,
                     finding_type, multi_class=False):
    X = torch.zeros((set_size, dimension), dtype=FEATURE_TYPE)
    Y = torch.zeros((set_size))

    with open(dataset_file, 'r') as f:
        for line, vector in enumerate(f):
            cat, data = vector.split(' ', 1)
            if multi_class:
                Y[line] = cat
            else:
                Y[line] = 1 if cat == finding_type else 0
            for piece in data.strip().split(' '):
                match = re.search(r'(\S+):(\S+)', piece)
                # feature in dataset starts from 1, so we substract 1 here
                feature = int(match.group(1)) - 1
                value = float(match.group(2))
                # record features
                X[line][feature] = value
    return X, Y
    
class StackedDataSet():
    '''
    All data samples are stacked into a tensor `self.features` 
    and `self.targets`
    '''
    def __init__(self, features, targets):
        assert len(features) == len(targets)
        set_size = len(features)
        self.features = features
        self.targets = targets

        # random shuffling
        self.__RR = False
        self.__order = list(range(set_size))

        self.set_size = set_size
        self.__COUNT = None
        
        ARBITRARY_DATA_SAMPLE = features[0]
        self.feature_dimension = ARBITRARY_DATA_SAMPLE.nelement()
        self.feature_size = ARBITRARY_DATA_SAMPLE.size()
        
    def get_count(self):
        if self.__COUNT is None:
            count = {}
            for target_tensor in self.targets:
                target = target_tensor.item()
                if target in count.keys():
                    count[target] += 1
                else:
                    count[target] = 1
            self.__COUNT = count
        return self.__COUNT
        
    def randomReshuffle(self, on):
        self.__RR = on
        if on:
            random.shuffle(self.__order)
    def __getitem__(self, index):
        if self.__RR:
            i = self.__order[index]
            return self.features[i], self.targets[i]
        else:
            return self.features[index], self.targets[index]
    def __len__(self):
        return self.set_size
    def subset(self, indexes, name=''):
        if name == '':
            name = self.name + '_subset'
        return StackedDataSet(self.features[indexes], self.targets[indexes])
    def get_test_set(self):
        if self.test_features is None or self.test_targets is None:
            self.test_features = self.features
            self.test_targets = self.targets
        return StackedDataSet(self.test_features, self.test_targets)
    

class DataPackage():
    '''
    A `DataPackage` contains the information, like the name and the data size, 
    of a specific dataset (e.g. MNIST, CIFAR10, CIFAR100), and the references of
    train and test dataset.
    Therefore, the relation of DataPackage and Dataset is: 
    DataPackage includes Dataset
    TODO: the arguments `train_set` and `test_set` can be other implementation
          of dataset
    '''
    def __init__(self, name, 
                 train_set: StackedDataSet, 
                 test_set: StackedDataSet):
        self.name = name
        self.train_set = train_set
        self.test_set = test_set
        
        assert len(train_set) != 0, 'No data in train set'
        assert len(test_set) != 0, 'No data in test set'
        
        assert train_set.feature_dimension == test_set.feature_dimension
        assert train_set.feature_size == test_set.feature_size
        
        self.feature_dimension = train_set.feature_dimension
        self.feature_size = train_set.feature_size
        
        
class SVM_dataPackage(DataPackage):
    def __init__(self, set_size, dimension, name, dataset_file,
                 finding_type=0, multi_class=False):
        # try to load processed dataset from cache
        feature_dtype_str = str(FEATURE_TYPE).replace('torch.', '')
        target_dtype_str = str(TARGET_TYPE).replace('torch.', '')
        cache_file_name = f'data_cache_{name}_{feature_dtype_str}_{target_dtype_str}'
        if isfile_in_cache(cache_file_name):
            cache = load_file_in_cache(cache_file_name)
            features, targets = cache['features'], cache['targets']
        else:
            features, targets = load_SVM_dataset(dataset_file, set_size, 
                                                 dimension, finding_type,
                                                 multi_class)
            features = features.type(FEATURE_TYPE)
            targets = targets.type(TARGET_TYPE)
            cache = {
                'features': features,
                'targets': targets,
            }
            dump_file_in_cache(cache_file_name, cache)
            
        dataset = StackedDataSet(features, targets)
            
        super().__init__(name=name, train_set=dataset, test_set=dataset)
        self._finding_type = finding_type

__DATASET_PATH__ = os.path.join(
    os.getcwd(), 'dataset'
)


class ijcnn(SVM_dataPackage):
    def __init__(self):
        super().__init__(
            set_size = 49990,
            dimension = 22,
            finding_type = '1',
            name = 'ijcnn1',
            dataset_file = os.path.join(__DATASET_PATH__, 'ijcnn1')
        )


class covtype(SVM_dataPackage):
    def __init__(self):
        super().__init__(
            set_size = 581012,
            dimension = 54,
            finding_type = '1',
            name = 'covtype',
            dataset_file = os.path.join(
                __DATASET_PATH__, 
                'covtype.libsvm.binary.scale'
            )
        )
        
        
class StackedTorchDataPackage(DataPackage):
    def __init__(self, name, load_fn):
        feature_dtype_str = str(FEATURE_TYPE).replace('torch.', '')
        target_dtype_str = str(TARGET_TYPE).replace('torch.', '')
        cache_file_name = f'data_cache_{name}_{feature_dtype_str}_{target_dtype_str}'
        if isfile_in_cache(cache_file_name):
            cache = load_file_in_cache(cache_file_name)
            train_features = cache['train_features']
            train_targets = cache['train_targets']
            test_features = cache['test_features']
            test_targets = cache['test_targets']
            num_classes = cache['num_classes']
        else:
            torch_train_set, torch_test_set = load_fn()
            train_features = torch.stack(
                [feature for feature, _ in torch_train_set], axis=0
            ).type(FEATURE_TYPE)
            test_features = torch.stack(
                [feature for feature, _ in torch_test_set], axis=0
            ).type(FEATURE_TYPE)
            train_targets = torch.tensor(torch_train_set.targets).type(TARGET_TYPE)
            test_targets = torch.tensor(torch_test_set.targets).type(TARGET_TYPE)
            num_classes = len(torch_train_set.classes)
            cache = {
                'train_features': train_features,
                'train_targets': train_targets,
                'test_features': test_features,
                'test_targets': test_targets,
                'num_classes': num_classes,
            }
            dump_file_in_cache(cache_file_name, cache)
        
        self.num_classes = num_classes
        train_set = StackedDataSet(train_features, train_targets)
        test_set = StackedDataSet(test_features, test_targets)
        super().__init__(name, train_set, test_set)
        
        
def get_mnist():
    from torchvision import transforms
    from torchvision.datasets import MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=ByrdLab.FEATURE_TYPE),
        # transforms.Normalize(mean=[0.1307],std=[0.3081])
        transforms.Normalize(mean=[0.5],std=[0.5])
        # transforms.Lambda(lambda x: x / 255)
    ])
    root = 'dataset'
    torch_train_dataset = MNIST(root=root, train=True,
                                transform=transform, download=True)
    torch_test_dataset = MNIST(root=root, train=False,
                              transform=transform, download=True)
    return torch_train_dataset, torch_test_dataset


def get_mnist_sorted_by_label():
    from torchvision import transforms
    from torchvision.datasets import MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=ByrdLab.FEATURE_TYPE),
        # transforms.Normalize(mean=[0.1307],std=[0.3081])
        transforms.Normalize(mean=[0.5],std=[0.5])
        # transforms.Lambda(lambda x: x / 255)
    ])

    root = 'dataset'
    torch_train_dataset = MNIST(root=root, train=True,
                                transform=transform, download=True)
    torch_test_dataset = MNIST(root=root, train=False,
                              transform=transform, download=True)

    # 重新排列训练数据集的目标按标签排序
    sorted_indices = torch.argsort(torch.Tensor(torch_train_dataset.targets))
    torch_train_dataset.targets = torch_train_dataset.targets[sorted_indices]
    torch_train_dataset.data = torch_train_dataset.data[sorted_indices]

    return torch_train_dataset, torch_test_dataset



class mnist(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('mnist', get_mnist)


class mnist_sorted_by_labels(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('mnist_sorted_by_label', get_mnist_sorted_by_label)



def get_fashion_mnist():
    from torchvision import transforms
    from torchvision.datasets import FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=ByrdLab.FEATURE_TYPE),
        # transforms.Normalize(mean=[0.1307],std=[0.3081])
        transforms.Normalize(mean=[0.5],std=[0.5])
        # transforms.Lambda(lambda x: x / 255)
    ])
    root = 'dataset'
    torch_train_dataset = FashionMNIST(root=root, train=True,
                                transform=transform, download=True)
    torch_test_dataset = FashionMNIST(root=root, train=False,
                              transform=transform, download=True)
    return torch_train_dataset, torch_test_dataset


class fashionmnist(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('fashionmnist', get_fashion_mnist)

def get_cifar10():
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=ByrdLab.FEATURE_TYPE),
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.1307],std=[0.3081])
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#transforms.Normalize(mean=[0.5],std=[0.5])
        # transforms.Lambda(lambda x: x / 255)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=ByrdLab.FEATURE_TYPE),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # target_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     #transforms.ConvertImageDtype(dtype=ByrdLab.TARGET_TYPE)
    # ])
    root = 'dataset'
    torch_train_dataset = CIFAR10(root=root, train=True,
                                transform=transform_train,  download=True)
    torch_test_dataset = CIFAR10(root=root, train=False,
                              transform=transform_test,  download=True)
    return torch_train_dataset, torch_test_dataset


class cifar10(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('cifar10', get_cifar10)


class ClassificationToySet(StackedDataSet):
    def __init__(self, set_size, dimension=20, class_cnt=10,
                 fix_seed=False, seed=20):
        generator = torch_rng(seed) if fix_seed else None
        features = torch.randn((set_size, dimension), dtype=FEATURE_TYPE,
                        generator=generator)
        targets = torch.randint(0, class_cnt, (set_size,))
        super().__init__(features=features, targets=targets)
        
  
# TODO: need to be updated into a StackedDataSet
class LogisticRegressionToySet(DataPackage):
    def __init__(self, set_size, dimension, noise=0.1, w_star=None,
                 fix_seed=False, seed=20):
        generator = torch_rng(seed) if fix_seed else None
        X = torch.randn((set_size, dimension), dtype=FEATURE_TYPE,
                        generator=generator)
        Y = torch.zeros((set_size))
        if w_star == None:
            w_star = torch.randn(dimension + 1, dtype=FEATURE_TYPE,
                                 generator=generator)
        assert w_star.size() == torch.Size([dimension + 1])
        self.w_star = w_star

        for i, x in enumerate(X):
            if self.w_star[:-1].dot(x) + self.w_star[-1] > 0:
                Y[i] = 1
            else:
                Y[i] = 0

        # add noise
        X.add_(torch.randn_like(X), alpha=noise)
        
        super().__init__(name='ToySet', features=X, targets=Y)

    
class DistributedDataSets():
    def __init__(self, dataset, partition_cls, nodes, honest_nodes,
                 rng_pack: RngPackage=RngPackage()):
        self.dataset = dataset
        
        self.partition = partition_cls(dataset, len(honest_nodes),
                                       rng_pack=rng_pack)
        honest_subsets = self.partition.get_subsets(dataset)
        
        # allocate the partitions to all nodes
        next_pointer = 0
        self.subsets = [[] for _ in nodes]
        for node in nodes:
            if node in honest_nodes:
                self.subsets[node] = honest_subsets[next_pointer]
                next_pointer += 1
    def __getitem__(self, index):
        return self.subsets[index]
    def entire_set(self):
        return self.dataset
    
class DistributedDataSets_over_honest_and_byz_nodes():
    def __init__(self, dataset, partition_cls, nodes,
                 rng_pack: RngPackage=RngPackage()):
        self.dataset = dataset
        
        self.partition = partition_cls(dataset, len(nodes),
                                       rng_pack=rng_pack)
        # self.partition.draw_data_distribution()
        self.subsets = self.partition.get_subsets(dataset)
        pass
    def __getitem__(self, index):
        return self.subsets[index]
    def entire_set(self):
        return self.dataset
 
    
    
class EmptySet(StackedDataSet):
    def __init__(self):
        super().__init__(features=torch.zeros(1, 1), targets=torch.zeros(1, 1))
    
    
class EmptyPackage(DataPackage):
    def __init__(self):
        train_set = EmptySet()
        test_set = EmptySet()
        super().__init__(name='EmptySet', 
                         train_set=train_set, test_set=test_set)
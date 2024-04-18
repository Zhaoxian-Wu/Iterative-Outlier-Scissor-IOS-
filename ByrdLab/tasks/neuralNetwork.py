import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import ResNet18_Weights
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.dataset import DataPackage

from ByrdLab.library.initialize import RandomInitialize, ZeroInitialize
from ByrdLab.library.measurements import multi_classification_accuracy
from ByrdLab.library.tool import adapt_model_type
from ByrdLab.tasks import Task


class CNNModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # CIFAR-10图片的尺寸为32x32，经过两次池化后为8x8
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # 将张量展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x
    

# 模型:两层MLP
class MLP(nn.Module):
    """
    Inputs                                 Linear/Function                               Output
    [batch_size, 1, 28, 28]          -> Linear(28*28, hidden_size1)          -> [batch_size, hidden_size1]  # first hidden layer
                                     -> Tanh                                 -> [batch_size, hidden_size1]  # tanh activation function, may sigmoid
                                     -> Linear(hidden_size1, hidden_size2)   -> [batch_size, hidden_size2]  # second hidden layer
                                     -> Tanh                                 -> [batch_size, hidden_size2]  # tanh activation function, may sigmoid
                                     -> Linear(hidden_size2, 10)             -> [batch_size, 10]  # Classification Layer
   """

    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(784, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.classification_layer = nn.Linear(50, num_classes)
        # self.tanh1 = nn.Tanh()
        # self.tanh2 = nn.Tanh()

    def forward(self, x):
        """
        :param x: [batch_size, channel, height, width], input for network
        :return: [batch_size, n_classes], output from network
        """
        out = x.view(x.size(0), -1)  # flatten x in [batch_size, dimensions]
        out = self.hidden1(out)
        # out = self.tanh1(out)
        out = F.relu(out)
        out = self.hidden2(out)
        # out = self.tanh2(out)
        out = F.relu(out)
        out = self.classification_layer(out)
        # out = F.log_softmax(out, dim=1)
        return out
    

# VGG11/13/16/19 in Pytorch
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    
def nn_loss(predictions, targets):
    loss = torch.nn.functional.cross_entropy(
                predictions, targets.type(torch.long).view(-1))
    return loss

def random_generator(dataset, batch_size=1, rng_pack: RngPackage=RngPackage()):
    while True:
        beg = rng_pack.random.randint(0, len(dataset)-1)
        if beg+batch_size <= len(dataset):
            yield dataset[beg:beg+batch_size]
        else:
            features, targets = zip(dataset[beg:beg+batch_size],
                                    dataset[0:(beg+batch_size) % len(dataset)])
            yield torch.cat(features), torch.cat(targets)
        
def order_generator(dataset, batch_size=1, rng_pack: RngPackage=RngPackage()):
    beg = 0
    while beg < len(dataset):
        end = min(beg+batch_size, len(dataset))
        yield dataset[beg:end]
        beg += batch_size
        
def full_generator(dataset, rng_pack: RngPackage=RngPackage()):
    while True:
        yield dataset[:]

class NeuralNetworkTask(Task):
    def __init__(self, data_package: DataPackage, batch_size=32):
        weight_decay = 0.0085
        # model = VGG('VGG11', data_package.num_classes)
        model = CNNModel(data_package.num_classes)
        # model = MLP(data_package.num_classes)

        # from torchvision.models import resnet18
        # model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # model.conv1 = torch.nn.Conv2d(model.conv1.in_channels,
        #                         model.conv1.out_channels,
        #                         3, 1, 1)
        # model.maxpool = torch.nn.Identity()  # nn.Conv2d(64, 64, 1, 1, 1)
        # num_features = model.fc.in_features
        # model.fc = torch.nn.Linear(num_features, 10)

        model = adapt_model_type(model)
        loss_fn = nn_loss
        test_fn = multi_classification_accuracy

        super_params = {
            'rounds': 200,
            'display_interval': 100,
            'batch_size': batch_size,
            'test_batch_size': 10000,
            'lr': 1e-2,
            'alpha':0.1,
        }

        test_set = data_package.test_set
        get_train_iter = partial(random_generator, batch_size=super_params['batch_size'])
        get_test_iter = partial(order_generator, dataset=test_set, batch_size=super_params['test_batch_size'])
        super().__init__(weight_decay, data_package, model,
                         loss_fn=loss_fn, test_fn=test_fn,
                         initialize_fn=RandomInitialize(),
                        #  initialize_fn=ZeroInitialize(),
                         get_train_iter=get_train_iter,
                         get_test_iter=get_test_iter,
                         super_params=super_params,
                         name=f'NeuralNetwork_{data_package.name}',
                         model_name='NN')
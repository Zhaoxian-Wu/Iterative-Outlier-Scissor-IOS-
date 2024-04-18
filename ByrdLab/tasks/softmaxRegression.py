import random
from functools import partial

import torch
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.dataset import DataPackage

from ByrdLab.library.initialize import RandomInitialize, ZeroInitialize
from ByrdLab.library.measurements import multi_classification_accuracy
from ByrdLab.library.tool import adapt_model_type
from ByrdLab.tasks import Task

class softmaxRegression_model(torch.nn.Module):
    def __init__(self, feature_dimension, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=feature_dimension,
                                      out_features=num_classes, bias=True)
    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.linear(features)
    
def softmax_regression_loss(predictions, targets):
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

class softmaxRegressionTask(Task):
    def __init__(self, data_package: DataPackage, batch_size=32):
        weight_decay = 0.01
        model = softmaxRegression_model(data_package.feature_dimension,
                                        data_package.num_classes)
        model = adapt_model_type(model)
        loss_fn = softmax_regression_loss
        test_fn = multi_classification_accuracy
        
        super_params = {
            'rounds': 200,
            'display_interval': 100,
            'batch_size': batch_size,
            # 'test_batch_size': 900,
            'test_batch_size': 10000,
            'lr': 1e-2,
            'alpha': 0.1,
        }
        
        test_set = data_package.test_set
        get_train_iter = partial(random_generator,
                                 batch_size=super_params['batch_size'])
        get_test_iter = partial(order_generator, dataset=test_set,
                                 batch_size=super_params['test_batch_size'])
        super().__init__(weight_decay, data_package, model, 
                         loss_fn=loss_fn, test_fn=test_fn,
                        #  initialize_fn=RandomInitialize(),
                        initialize_fn=ZeroInitialize(),
                         get_train_iter=get_train_iter,
                         get_test_iter=get_test_iter,
                         super_params=super_params,
                         name=f'SR_{data_package.name}',
                         model_name='softmaxRegression')
        

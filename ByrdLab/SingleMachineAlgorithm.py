import torch
import random
import numpy as np
import torch.optim as optim
from ByrdLab import DEVICE
from ByrdLab.environment import Dec_Byz_Opt_Env
from ByrdLab.library.measurements import avg_loss_accuracy_dist, consensus_error, one_node_loss_accuracy_dist
from ByrdLab.library.tool import log


class SGD(Dec_Byz_Opt_Env):
    def __init__(self, graph, consensus_init=False, *args, **kw):
        super().__init__(name='SGD', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        # self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[SGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iter = self.get_train_iter(dataset=self.dist_train_set[0], rng_pack=self.rng_pack)
        dist_models.activate_model(0)
        model = dist_models.model
        grad = [torch.zeros_like(para, requires_grad=False) for para in model.parameters()]
        momentum = 0.9
        # lr = self.lr
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                test_loss, test_accuracy = avg_loss_accuracy_dist(
                    dist_models, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                

                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            
            features, targets = next(data_iter)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(features)
            loss = self.loss_fn(predictions, targets)
            model.zero_grad()
            loss.backward()
            # optimizer.step()
            
            # record loss
            # train_loss += loss.item()
            # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
            # TODO: correct prediction_cls
            # _, prediction_cls = torch.max(predictions.detach(), dim=1)
            # train_accuracy += (prediction_cls == targets).sum().item()
            # total_sample += len(targets)
            # total_sample += 1
            
            # momentum
            for index, para in enumerate(model.parameters()):
                if para.grad is not None:
                    grad[index].data.mul_(momentum)
                    grad[index].data.add_(para.grad)
            
            # gradient descend
            with torch.no_grad():
                for param, g in zip(model.parameters(), grad):
                    if g is not None:
                        param.data.mul_(1 - self.weight_decay * lr)
                        param.data.sub_(g, alpha=lr)
        
        # Compute the loss of model on each sample
        # dist_models.activate_model(0)
        # model = dist_models.model
        # data_size = len(self.data_package.train_set)
        # num_classes = self.data_package.num_classes
        # loss_model = np.zeros(num_classes * data_size)
        # for i in range(data_size):
        #     feature = self.data_package.train_set.features[i]
        #     target = self.data_package.train_set.targets[i]
        #     feature = feature.to(DEVICE)
        #     target = target.to(DEVICE)
        #     prediction = model(feature)
        #     for k in range(num_classes):
        #         target_flipped = (target + k) % num_classes
        #         loss_model[i + k * data_size] = self.loss_fn(prediction, target_flipped)
        #         # print(loss_clean_model[i + k * data_size])

        return model

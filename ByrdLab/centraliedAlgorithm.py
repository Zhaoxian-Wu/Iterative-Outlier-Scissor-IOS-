import torch
import random
import copy
from ByrdLab import DEVICE
from ByrdLab.environment import  Dist_Dataset_Opt_Env
from ByrdLab.library.dataset import EmptySet
from ByrdLab.library.partition import EmptyPartition
from ByrdLab.library.measurements import avg_loss_accuracy_dist, consensus_error, one_node_loss_accuracy_dist
from ByrdLab.library.tool import log, flatten_list, unflatten_vector, flatten_vector
from ByrdLab.library.cache_io import dump_file_in_cache, load_file_in_cache



# CSGD under model poisoning attacks
class CSGD(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes,  *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))
                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            # communication and attack
            if self.attack != None and self.byzantine_size!= 0:
                self.attack.run(worker_grad_flat)
            
            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    
   
class CSGD_compute_bound(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes,  *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'
        
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]
        # hetero_list = []
        grad_norm_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))
                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            # communication and attack
            if self.attack != None and self.byzantine_size!= 0:
                self.attack.run(worker_grad_flat)

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)


            grad_norms = torch.tensor([torch.norm(worker_grad_flat[node]) for node in self.nodes]) 
            max_grad_norm = grad_norms.max()
            print('Maximum of gradient norm:', max_grad_norm)
            grad_norm_list.append(max_grad_norm) 

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path, grad_norm_list 
    
class CSGD_compute_hetero(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes,  *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]
        hetero_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            # communication and attack
            if self.attack != None and self.byzantine_size!= 0:
                self.attack.run(worker_grad_flat)

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                distances = torch.tensor([torch.norm(worker_grad_flat[node] - aggrGrad_flat) for node in self.nodes])
                heterogeneity = distances.max()
                print('Heterogeneity:', heterogeneity.item())
                hetero_list.append(heterogeneity.item())

        return server_model, loss_path, acc_path, hetero_list
    
class CMomentum_compute_hetero(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        hetero_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                

            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            # communication and attack
            if self.attack != None and self.byzantine_size!= 0:
                self.attack.run(worker_grad_flat)

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                distances = torch.tensor([torch.norm(worker_grad_flat[node] - aggrGrad_flat) for node in self.nodes])
                heterogeneity = distances.max()
                print('Heterogeneity:', heterogeneity.item())
                hetero_list.append(heterogeneity.item())

        return server_model, loss_path, acc_path, hetero_list


# CSGD under data poisoning attacks
class CSGD_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    
class CMomentum_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)
        # alpha = 0.01

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        # worker_grad = [
        #     [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
        #     for _ in range(self.node_size)
        # ]

        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                # for index, para in enumerate(server_model.parameters()):
                #     worker_grad[node][index].data.zero_()
                #     worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                #     worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            # worker_grad_flat = flatten_list(worker_grad)  
            worker_grad_flat = flatten_list(worker_momentum)  

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    
class CMomentum_under_DPA_compute_bound(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        Bound_A_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                # grad_norms = torch.tensor([torch.norm(worker_grad_flat[node]) for node in self.nodes]) 
                # Bound_A_max = grad_norms.max()
                averaged_regular_grad = torch.mean(worker_grad_flat[self.honest_nodes], dim=0)
                grad_norms = torch.tensor([torch.norm(worker_grad_flat[node] - averaged_regular_grad) for node in self.byzantine_nodes]) 
                Bound_A_max = grad_norms.max()
                print(f'{iteration}-iteration Bound_A:', Bound_A_max.item())
                Bound_A_list.append(Bound_A_max.item()) 

        return server_model, loss_path, acc_path, Bound_A_list
    
class CSGD_under_DPA_compute_bound(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        Bound_A_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                # grad_norms = torch.tensor([torch.norm(worker_grad_flat[node]) for node in self.nodes]) 
                # Bound_A_max = grad_norms.max()
                averaged_regular_grad = torch.mean(worker_grad_flat[self.honest_nodes], dim=0)
                grad_norms = torch.tensor([torch.norm(worker_grad_flat[node] - averaged_regular_grad) for node in self.byzantine_nodes]) 
                Bound_A_max = grad_norms.max()
                print(f'{iteration}-iteration Bound_A:', Bound_A_max.item())
                Bound_A_list.append(Bound_A_max.item()) 

        return server_model, loss_path, acc_path, Bound_A_list
    

class CSGD_with_LFighter_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes,  *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            min_norm_feature = 0
            max_norm_feature = 0
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                norm_feature = torch.norm(features)
                if norm_feature > max_norm_feature:
                    max_norm_feature = norm_feature
                if min_norm_feature == 0:
                    min_norm_feature = norm_feature
                elif norm_feature < min_norm_feature:
                    min_norm_feature = norm_feature
                assert min_norm_feature <= max_norm_feature

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                

                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_grad)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            print('minimum of features norm:', min_norm_feature.item())
            print('maximum of features norm:', max_norm_feature.item())

        return server_model, loss_path, acc_path
    
class CMomentum_with_LFighter_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1,  *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)
        # alpha = 0.1

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_momentum)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    

# CSGD under data poisoning attacks
class CSGD_under_DPA_with_prob(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, prob, *args, **kw):
        super().__init__(name=f'CSGD_p={prob}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.prob = prob
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            random_number = random.uniform(0, 1)
                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    if random_number <= self.prob:
                        features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    

class CSGD_with_LFighter_under_DPA_with_prob(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, prob,  *args, **kw):
        super().__init__(name=f'CSGD_p={prob}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.prob = prob
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            # min_norm_feature = 0
            # max_norm_feature = 0
                
            random_number = random.uniform(0, 1)
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                # norm_feature = torch.norm(features)
                # if norm_feature > max_norm_feature:
                #     max_norm_feature = norm_feature
                # if min_norm_feature == 0:
                #     min_norm_feature = norm_feature
                # elif norm_feature < min_norm_feature:
                #     min_norm_feature = norm_feature
                # assert min_norm_feature <= max_norm_feature

                # data poisoning attack
                if node in self.byzantine_nodes:
                    if random_number <= self.prob:
                        features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                

                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_grad)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            # print('minimum of features norm:', min_norm_feature.item())
            # print('maximum of features norm:', max_norm_feature.item())

        return server_model, loss_path, acc_path
    
    


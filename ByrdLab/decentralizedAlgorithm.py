import torch
import random
from ByrdLab import DEVICE
from ByrdLab.environment import Dec_Byz_Iter_Env, Dec_Byz_Opt_Env
from ByrdLab.library.dataset import EmptySet
from ByrdLab.library.partition import EmptyPartition
from ByrdLab.library.measurements import avg_loss_accuracy_dist, consensus_error, one_node_loss_accuracy_dist
from ByrdLab.library.tool import log
from ByrdLab.graph import RandomGeometricGraph


# alternative: Apatation & Combination
class DSGD(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(name='DSGD', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
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
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # aggregation
                aggregation_res = self.aggregation.run(param_bf_comm, node)
                dist_models.params_vec[node].copy_(aggregation_res)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    

# DSGD with Multi-Step Aggregation
class DSGD_MSG(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, step_agg=1,  *args, **kw):
        super().__init__(name='DSGD_MSG', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation
        self.step_agg = step_agg
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        # self.graph.show(show_label=True, show_lost=True)
        
        # self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD_MSG]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)

            # if 'RGG' in self.graph.name:
            #     honest_size = self.graph.honest_size
            #     byzantine_size = self.graph.byzantine_size
            #     node_size = honest_size + byzantine_size
            #     radius = self.graph.radius
            #     self.graph = RandomGeometricGraph(node_size=node_size, byzantine_size=byzantine_size,
            #                                        radius=radius, seed=300)
                
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                test_loss, test_accuracy = avg_loss_accuracy_dist(
                    dist_models, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                # test_loss, test_accuracy = one_node_loss_accuracy_dist(
                #     dist_models, self.get_test_iter,
                #     self.loss_fn, self.test_fn,
                #     weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
            

                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                model.train()
                # model = model.to(DEVICE)
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                model.zero_grad()
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                # model.zero_grad()
                loss.backward()
                
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad.data is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad.data, alpha=lr)
            
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for _ in range(self.step_agg):
                # store the parameters before communication
                param_bf_comm.copy_(dist_models.params_vec)
                for node in self.graph.honest_nodes:
                    # Byzantine attack
                    byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                    if self.attack != None and byzantine_neighbors_size != 0:
                        self.attack.run(param_bf_comm, node, self.rng_pack)
                    # aggregation
                    aggregation_res = self.aggregation.run(param_bf_comm, node)
                    dist_models.params_vec[node].copy_(aggregation_res)


            # Debug
            # for node in self.graph.honest_nodes:
            #     dist_models.activate_model(node)
            #     model = dist_models.model
            #     for node2 in self.graph.honest_nodes:
            #         param_vec = dist_models.params_vec[node2]
            #         cumulated_param = 0
            #         for param in model.parameters():
            #             param_size = param.nelement()
            #             beg, end = cumulated_param, cumulated_param + param.nelement()
            #             cumulated_param += param_size
            #             vec = param_vec[beg: end]
            #             print(param.data.reshape(-1).equal(vec.data))

        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
 
# DSGD with multiple aggregation under label flipping attack
# TODO: split the label flipping attack into uniform case and concentrate case
#       I guess the concentrate case is what we really need in experiments  
class DSGD_MSG_under_DPA(Dec_Byz_Opt_Env):
    def __init__(self, graph, aggregation, consensus_init=False, step_agg=1,  *args, **kw):
        super().__init__(name='DSGD_MSG', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation
        self.step_agg = step_agg
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        # self.graph.show(show_label=True, show_lost=True)
        
        # self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        compromised_ratio = self.graph.byzantine_size / self.graph.node_size
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD_MSG]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                    #   if node in self.honest_nodes else None
                      for node in self.nodes]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            if 'RGG' in self.graph.name:
                honest_size = self.graph.honest_size
                byzantine_size = self.graph.byzantine_size
                node_size = honest_size + byzantine_size
                radius = self.graph.radius
                self.graph = RandomGeometricGraph(node_size=node_size, byzantine_size=byzantine_size,
                                                   radius=radius, seed=300)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample

                test_loss, test_accuracy = avg_loss_accuracy_dist(
                    dist_models, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            # for node in self.graph.honest_nodes:
            for node in self.nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                model.train()
                # model = model.to(DEVICE)
                features, targets = next(data_iters[node])
                # data poisoning attack
                if node in self.byzantine_nodes:
                # if iteration % int(1 / compromised_ratio) == 0:
                    features, targets = self.attack.run(features, targets, model=model)
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                model.zero_grad()
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                # model.zero_grad()
                loss.backward()
                grad_norm = 0

                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            # grad_norm += torch.norm(param.grad)
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for _ in range(self.step_agg):
                # store the parameters before communication
                param_bf_comm.copy_(dist_models.params_vec)
                # for node in self.graph.honest_nodes:
                for node in self.nodes:
                    # aggregation
                    aggregation_res = self.aggregation.run(param_bf_comm, node)
                    # print(f'{iteration}-th iteration norm:{torch.norm(aggregation_res - param_bf_comm) / lr}')
                    dist_models.params_vec[node].copy_(aggregation_res)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
 
    
# Adaptation While Combination
class DSGD_AWC(DSGD):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super().__init__(graph, consensus_init, *args, **kw)
        self.name = 'DSGD_AWC'
        self.aggregation = aggregation
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        grads = [
            [torch.zeros_like(param) for param in dist_models.model.parameters()] 
            for _ in self.graph.nodes
        ]
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                # TODO: training loss -> val loss
                test_loss, test_accuracy = avg_loss_accuracy_dist(
                    dist_models, self.get_test_iter, 
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for (param_idx, param) in enumerate(model.parameters()):
                        if param.grad is not None:
                            param.grad.add_(param, alpha=self.weight_decay)
                            grads[node][param_idx].copy_(param.grad)
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # aggregation
                aggregation_res = self.aggregation.run(param_bf_comm, node)
                dist_models.params_vec[node].data.copy_(aggregation_res)
                
                with torch.no_grad():
                    dist_models.activate_model(node)
                    model = dist_models.model
                    for (param_idx, param) in enumerate(model.parameters()):
                        param.sub_(grads[node][param_idx], alpha=lr)
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
      
class RSA_algorithm(Dec_Byz_Opt_Env):
    def __init__(self, graph, penalty=0.001, consensus_init=False, *args, **kw):
        super().__init__(name=f'RSA_lamb={penalty}', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.lamb = penalty
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
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
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
            # main loop
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                # aggregation
                local_model = dist_models.params_vec[node]
                for j in self.graph.neighbors[node]:
                    diff = param_bf_comm[j] - param_bf_comm[node]
                    local_model.add_(diff.sign(), alpha=lr * self.lamb)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    

class RSA_algorithm_under_DPA(Dec_Byz_Opt_Env):
    def __init__(self, graph, penalty=0.001, consensus_init=False, *args, **kw):
        super().__init__(name=f'RSA_lamb={penalty}', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.lamb = penalty
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                    #   if node in self.honest_nodes else None
                      for node in self.nodes]
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
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
            # main loop
            for node in self.nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets)
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # Byzantine attack
                # byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                # if self.attack != None and byzantine_neighbors_size != 0:
                #     self.attack.run(param_bf_comm, node, self.rng_pack)
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                # aggregation
                local_model = dist_models.params_vec[node]
                for j in self.graph.neighbors[node]:
                    diff = param_bf_comm[j] - param_bf_comm[node]
                    local_model.add_(diff.sign(), alpha=lr * self.lamb)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
    
class Decentralized_gossip(Dec_Byz_Iter_Env):
    def __init__(self, graph, aggregation, *args, **kw):
        super().__init__(name='gossip', graph=graph, *args, **kw)
        self.aggregation = aggregation
    def run(self, init_local_models = None):
        # reture_model_list: whether return the history list of all models
        self.construct_rng_pack()
        local_models = torch.zeros_like(init_local_models)
        local_models.copy_(init_local_models)
        init_ce = consensus_error(local_models, self.graph.honest_nodes)
        ce_path = [init_ce]
        
        for iteration in range(1, self.total_iterations + 1):
            old_local_models = torch.zeros_like(local_models)
            old_local_models.copy_(local_models)
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(old_local_models, node, self.rng_pack)
                # aggregation
                local_models[node] = self.aggregation.run(old_local_models, node)
            
            if iteration % self.display_interval == 0:
                ce = consensus_error(local_models, self.graph.honest_nodes)
                ce_path.append(ce)
                
        return ce_path
   
   
# Xu, Jian, and Shao-Lun Huang. 
# "Byzantine-Resilient Decentralized Collaborative Learning." 
# ICASSP 2022
class SimiliarityReweighting(Dec_Byz_Opt_Env):
    def __init__(self, graph, c=1, consensus_init=False, *args, **kw):
        super().__init__(name='SimReweight', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.c = c
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.softmax = torch.nn.Softmax(dim=0)
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[SimReweight]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before gradient descent
        param_bf_gd = torch.zeros_like(dist_models.params_vec)
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
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
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            param_bf_gd.copy_(dist_models.params_vec)
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                            
            param_bf_comm.copy_(dist_models.params_vec)
            # communication and attack
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                model_changes = param_bf_comm - param_bf_gd[node]
                model_change_norms = model_changes.norm(dim=1)
                sim_1 = self.sim_1(node, model_changes)
                sim_2 = self.sim_2(node, model_changes, model_change_norms)
                weight = sim_1 * sim_2
                # norm cliping
                # if ii >= ij, don't clip
                # else clip
                ii_gt_than_ij = model_change_norms[node] >= model_change_norms
                normed_change = (
                    model_changes.transpose(0, 1) * (ii_gt_than_ij \
                    + ~ii_gt_than_ij * model_change_norms[node] / model_change_norms)
                ).transpose(0, 1)
                
                # aggregation
                weighted_change = torch.matmul(weight, normed_change)
                aggregation_res = param_bf_gd[node] + weighted_change
                dist_models.params_vec[node].copy_(aggregation_res)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
    def sim_1(self, node, model_changes):
        world_size = model_changes.size(0)
        changes_sign = model_changes.sign()
        # distribution of -1, 0, 1
        sign_dist_torch = torch.zeros((world_size, 3), dtype=torch.double)
        # count the number of -1, 0, 1
        for n in self.graph.neighbors_and_itself[node]:
            signed_change = changes_sign[n]
            elements, cnts = torch.unique(signed_change, return_counts=True, 
                                          sorted=True)
            if elements.size(0) == 1:
                idx = int(elements[0].item())
                sign_dist_torch[n][idx] = cnts[0]
            else:
                sign_dist_dict = {
                    element.item(): cnt for element, cnt in zip(elements, cnts)
                }
                sign_dist_torch[n].copy_(torch.tensor([
                    sign_dist_dict.get(i, 0) for i in [-1, 0, 1] 
                ]))
        # nomalize
        model_size = model_changes.size(1)
        sign_dist_torch.div_(model_size)
        weight_on_neighbors = self.softmax(torch.tensor([
            -self.c * self.kl_loss(sign_dist_torch[n], sign_dist_torch[node])
            for _, n in enumerate(self.graph.neighbors_and_itself[node])
        ]))
        weight = torch.zeros(world_size).type_as(model_changes)
        for j, n in enumerate(self.graph.neighbors_and_itself[node]):
            weight[n] = weight_on_neighbors[j]
        return weight
    def sim_2(self, node, model_changes, model_change_norms):
        world_size = model_changes.size(0)
        weight = torch.zeros(world_size).type_as(model_changes)
        weight[node] = 1
        self_direction = model_changes[node] / model_change_norms[node]
        for n in self.graph.honest_neighbors[node]:
            inner_product = self_direction.dot(model_changes[n])
            if inner_product > 0:
                weight[n] = inner_product / model_change_norms[n]
            else:
                weight[n] = 0
        return weight
    
class DSGD_inner_variation(DSGD):
    def __init__(self, graph, aggregation, inner_variation=0, *args, **kw):
        super().__init__(graph=graph, *args, **kw)
        self.name = f'DSGD_sigma={inner_variation}'
        self.inner_variation = inner_variation
        self.aggregation = aggregation
    
    def initilize_models(self, dist_models, consensus=False):
        if self.initialize_fn is None:
            return
        for node in self.honest_nodes:
            dist_models.activate_model(node)
            model = dist_models.model
            seed = self.seed if consensus else node + self.seed
            self.initialize_fn(model, fix_init_model=self.fix_seed, seed=seed)
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                # TODO: training loss -> val loss
                test_loss, test_accuracy = avg_loss_accuracy_dist(
                    dist_models, self.get_test_iter, 
                    self.loss_fn, self.test_fn, weight_decay=0,
                    node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                            noise = torch.randn(param.size(), 
                                                generator=self.rng_pack.torch
                                    ) * self.inner_variation
                            param.data.add_(noise)
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # aggregation
                aggregation_res = self.aggregation.run(param_bf_comm, node)
                dist_models.params_vec[node].copy_(aggregation_res)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
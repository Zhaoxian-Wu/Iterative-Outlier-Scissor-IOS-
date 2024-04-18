import random
import torch

import matplotlib.pyplot as plt
import networkx as nx
            
def is_valid(nx_graph):
    return nx.connected.is_connected(nx_graph)

class Graph():
    def __init__(self, name, nx_graph, honest_nodes, byzantine_nodes):
        self.init(name, nx_graph, honest_nodes, byzantine_nodes)
        
    def init(self, name, nx_graph, honest_nodes, byzantine_nodes):
        self.name = name
        self.nx_graph = nx_graph
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
        # node counting
        self.node_size = nx_graph.number_of_nodes()
        self.honest_size = len(honest_nodes)
        self.byzantine_size = len(byzantine_nodes)
        # neighbor list
        self.neighbors = [
            list(nx_graph.neighbors(node)) for node in nx_graph.nodes()
        ]
        self.honest_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in honest_nodes]
            for i in nx_graph.nodes()
        ]
        self.byzantine_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in byzantine_nodes]
            for i in nx_graph.nodes()
        ]
        self.honest_neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.honest_neighbors)
        ]
        self.neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.neighbors)
        ]
        # neighbor size list
        self.honest_sizes = [
            len(node_list) for node_list in self.honest_neighbors
        ]
        self.byzantine_sizes = [
            len(node_list) for node_list in self.byzantine_neighbors
        ]
        self.neighbor_sizes = [
            self.honest_sizes[node] + self.byzantine_sizes[node] 
            for node in nx_graph.nodes()
        ]
        
        # lost node refers to the the node has more than 1/2 byzantine neighbors
        self.lost_nodes = [
            node for node in self.honest_nodes
            if self.honest_sizes[node] <= 2 * self.byzantine_sizes[node]
        ]
        
    def honest_subgraph(self, name='', relabel=True):
        nx_subgraph = self.subgraph(self.honest_nodes)
        if name == '':
            name = self.name.replace(f'n={self.node_size}', f'n={self.honest_size}').replace(f'b={self.byzantine_size}', 'b=0')
        if relabel:
            nx_subgraph = nx.convert_node_labels_to_integers(nx_subgraph)
        return Graph(name=name, nx_graph=nx_subgraph, 
                     honest_nodes=list(nx_subgraph.nodes()),
                     byzantine_nodes=[])
    
    def __getattr__(self, attr):
        '''
        inherit the properties of 'nx_graph'
        '''
        return getattr(self.nx_graph, attr)
    
    def show(self, show_label=False, show_lost=False, as_subplot=False,
             label_dict=None, node_size=400, font_size=12):
        NODE_COLOR_HONEST = '#99CCCC'
        NODE_COLOR_BYZANTINE = '#FF6666'
        NODE_COLOR_LOST = '#CCCCCC'
        EDGE_WIDTH = 2
        
        # loyout
        pos = nx.kamada_kawai_layout(self.nx_graph)

        # honest nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
            node_size = node_size, 
            nodelist = self.honest_nodes,
            node_color = NODE_COLOR_HONEST,
        )
        # Byzantine nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
            node_size = node_size,
            nodelist = self.byzantine_nodes,
            node_color = NODE_COLOR_BYZANTINE,
        )
        # Lost nodes
        if show_lost:
            nx.draw_networkx_nodes(self.nx_graph, pos, 
                node_size = node_size,
                nodelist = self.lost_nodes,
                node_color = NODE_COLOR_LOST,
            )

        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.5, width=EDGE_WIDTH)

        if show_label:
            if label_dict == None:
                label_dict = {
                    i: str(i) for i in range(self.nx_graph.number_of_nodes())
                }
            nx.draw_networkx_labels(self.nx_graph, pos, label_dict,
                                    font_size=font_size)
        
        if not as_subplot:
            plt.savefig(f'{self.name}.pdf')
            plt.show()
    
    def __getstate__(self):
        state = {
            'name': self.name,
            'nx_graph': self.nx_graph,
            'honest_nodes': self.honest_nodes,
            'byzantine_nodes': self.byzantine_nodes
        }
        return state
    
    def __setstate__(self, state):
        name = state['name']
        nx_graph = state['nx_graph']
        honest_nodes = state['honest_nodes']
        byzantine_nodes = state['byzantine_nodes']
        self.init(name, nx_graph, honest_nodes, byzantine_nodes)

class CompleteGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.complete_graph(node_size)
        
        # honest_nodes = list(range(node_size-byzantine_size))
        # byzantine_nodes = list(range(node_size-byzantine_size, node_size))

        # all_nodes = list(range(node_size))
        # byzantine_nodes = list(range(7, 7 + byzantine_size))
        # honest_nodes = [node for node in all_nodes if node not in byzantine_nodes]

        # all_nodes = list(range(node_size))
        # byzantine_nodes = [0, 1, 5]
        # # byzantine_nodes = [5]
        # honest_nodes = [node for node in all_nodes if node not in byzantine_nodes]

        all_nodes = list(range(node_size))
        # byzantine_nodes = [1, 3, 5, 7]
        byzantine_nodes = [0]
        honest_nodes = [node for node in all_nodes if node not in byzantine_nodes]

        assert byzantine_size == len(byzantine_nodes)

        name = f'Complete_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        
class LineGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.path_graph(node_size)
        
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'Line_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)

class ErdosRenyi(Graph):
    def __init__(self, node_size, byzantine_size, connected_p=0.7, seed=None):
        rng = random if seed is None else random.Random(seed)
        valid = False
        while not valid:
            graph = nx.fast_gnp_random_graph(node_size, connected_p, seed=rng)
            valid = is_valid(graph)
        
        byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
        honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        name = f'ER_n={node_size}_b={byzantine_size}_p={connected_p}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super().__init__(name = name, nx_graph = graph,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)
        
class RandomGeometricGraph(Graph):
    def __init__(self, node_size, byzantine_size, radius, seed=None):
        rng = random if seed is None else random.Random(seed)
        valid = False
        while not valid:
            graph = nx.random_geometric_graph(node_size, radius, seed=rng)
            valid = is_valid(graph)
        
        # byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
        # honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'RGG_n={node_size}_b={byzantine_size}_radius={radius}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super().__init__(name = name, nx_graph = graph,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)
        self.radius = radius
        self.P = self.Prob_matrix() 
        self.neighbors, self.honest_neighbors, self.byzantine_neighbors, self.neighbor_sizes, self.honest_sizes, self.byzantine_sizes = self.time_varying_graph()

    def Prob_matrix(self, k=0.7):
        nodes = self.nx_graph.nodes(data=True)
        P = torch.zeros((self.node_size, self.node_size))   
        for u, du in nodes:
            pu = du['pos']
            for v, dv in nodes:
                pv = dv['pos']
                d = sum((a - b) ** 2 for a, b in zip(pu, pv))
                prob = k ** ((d / self.radius ** 2) ** 2)
                P[u][v] = prob
        return P
    
    def time_varying_graph(self):
        neighbors = self.neighbors.copy()
        honest_neighbors = self.honest_neighbors.copy()
        byzantine_neighbors = self.byzantine_neighbors.copy()
        # neighbor_sizes = []
        # honest_sizes = []
        # byzantine_sizes = []
        for i, j in self.nx_graph.edges():
            if random.random() >= self.P[i][j]:
                neighbors[i].remove(j)
                neighbors[j].remove(i)
                if j in self.honest_nodes:
                    honest_neighbors[i].remove(j)
                elif j in self.byzantine_nodes:
                    byzantine_neighbors[i].remove(j)
                if i in self.honest_nodes:
                    honest_neighbors[j].remove(i)
                elif i in self.byzantine_nodes:
                    byzantine_neighbors[j].remove(i)
        
        # neighbor size list
        honest_sizes = [
            len(node_list) for node_list in honest_neighbors
        ]
        byzantine_sizes = [
            len(node_list) for node_list in byzantine_neighbors
        ]
        neighbor_sizes = [
            honest_sizes[node] + byzantine_sizes[node] 
            for node in self.nx_graph.nodes()
        ]

        return neighbors, honest_neighbors, byzantine_neighbors, neighbor_sizes, honest_sizes, byzantine_sizes      
    
class TwoCastle(Graph):
    '''
    There are 2k nodes in the netword totally
    '''
    def __init__(self, k=3, byzantine_size=1, seed=None):
        '''k >= 3, byzantine_size <= k-2'''
        assert k >= 3, 'k must be greater than or equal to 3'
        assert byzantine_size <= k - 2, 'byzantine_size must be less than or equal to k - 2'
        node_size = 2 * k
        rng = random if seed is None else random.Random(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_size))
        # inner edges
        for castle in range(2):
            edges_list = [(i, j) for i in range(k*castle, k*castle+k)
                          for j in range(i+1, k*castle+k)]
            graph.add_edges_from(edges_list)
        # outer edges
        edges_list = [(i, j) for i in range(k)
                      for j in range(k, 2*k) if i + k != j]
        graph.add_edges_from(edges_list)
        byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
        honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        name = f'TwoCastle_k={k}_b={byzantine_size}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super().__init__(name = name, nx_graph = graph,
                                        honest_nodes=honest_nodes,
                                        byzantine_nodes=byzantine_nodes)

class RingCastle(Graph):
    def __init__(self, castle_cnt, byzantine_size, seed=None):
        node_size = 4 * castle_cnt
        
        rng = random if seed is None else random.Random(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_size))
        
        # inner edges
        for castle in range(castle_cnt):
            edges_list = [(i, j) for i in range(4*castle, 4*castle+4)
                          for j in range(i+1, 4*castle+4)]
            graph.add_edges_from(edges_list)
        # outer edges
        for castle in range(castle_cnt):
            next_castle = (castle+1) % castle_cnt
            graph.add_edges_from([
                (4*castle+2, 4*next_castle+0),
                (4*castle+3, 4*next_castle+1),
            ])
            byzantine_nodes = rng.sample(graph.nodes(), byzantine_size)
            honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
            name = f'RingCastle_castle={castle_cnt}_b={byzantine_size}'
            if seed is not None:
                name = name + f'_seed={seed}'
        
        super().__init__(name, graph, honest_nodes, byzantine_nodes)
        

class OctopusGraph(Graph):
    def __init__(self, head_cnt, head_byzantine_cnt, hand_byzantine_cnt):
        assert head_cnt > head_byzantine_cnt
        assert head_cnt > hand_byzantine_cnt
        # head
        nx_graph = nx.complete_graph(head_cnt)
        # hands
        nx_graph.add_nodes_from(range(head_cnt, 2*head_cnt))
        nx_graph.add_edges_from([
            (i, i+head_cnt) for i in range(head_cnt)
        ])
        
        honest_nodes = list(range(head_byzantine_cnt, head_cnt)) \
            + list(range(head_cnt+hand_byzantine_cnt, 2*head_cnt))
        byzantine_nodes = list(range(head_byzantine_cnt)) \
            + list(range(head_cnt, head_cnt+hand_byzantine_cnt))
        
        name = f'Octopus_head={head_cnt}_headb={head_byzantine_cnt}_handb={hand_byzantine_cnt}'
        super().__init__(name, nx_graph, honest_nodes, byzantine_nodes)

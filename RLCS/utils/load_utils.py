import os
import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
from torch_sparse import spspmm, coalesce
from torch_geometric.data import Data

class TwoHopNeighbor(object):

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def hypergraph_construction(edge_index, num_nodes, k=1):
    if k == 1:
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    else:
        neighbor_augment = TwoHopNeighbor()
        hop_data = Data(edge_index=edge_index, edge_attr=None)
        hop_data.num_nodes = num_nodes
        for _ in range(k - 1):
            hop_data = neighbor_augment(hop_data)
        hop_edge_index = hop_data.edge_index
        hop_edge_attr = hop_data.edge_attr
        edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    return edge_index, edge_attr


def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
    path_train = os.path.join(root,dataset,f'{dataset}_{train_path}_{train_n}.txt')
    if not os.path.isfile(path_train):
        raise Exception("No such file: %s" % path_train)
    train_lists = []
    for line in open(path_train, encoding='utf-8'):
        q, pos, comm = line.split(",")
        q = int(q)
        pos = pos.split(" ")
        pos_ = [int(x) for x in pos if int(x)!=q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(train_lists)>=train_n:
            break
        train_lists.append((q, pos_, comm_))

    path_test = os.path.join(root,dataset,f'{dataset}_{test_path}_{test_n}.txt')
    if not os.path.isfile(path_test):
        raise Exception("No such file: %s" % path_test)
    test_lists = []
    for line in open(path_test, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(test_lists)>=test_n:
            break
        test_lists.append((q, comm_))
    path_val = os.path.join(root,dataset,f'{dataset}_{val_path}_{val_n}.txt')
    if not os.path.isfile(path_val):
        raise Exception("No such file: %s" % path_val)
    val_lists = []
    for line in open(path_val, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(val_lists)>=val_n:
            break
        val_lists.append((q, comm_))

    return train_lists, val_lists, test_lists

def load_graph(root,dataset,attack,ptb_rate=None,type=None):
    if attack == 'none':
        if dataset in ['cocs', 'photo', 'dblp', 'amazon', 'cora', 'pubmed', 'citeseer', 'facebook', 'ex-fb', 'wfb107']:
            graphx = nx.read_edgelist(f'{root}/{dataset}/{dataset}.edges', nodetype=int, data=False)
            print(f'{dataset}:', graphx)
            n_nodes = graphx.number_of_nodes()
    elif attack in ['add','random_remove','gaddm','del','random_add','gdelm','random_flip','gflipm','cdelm','cflipm','delm','flipm','meta']:
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()

    return graphx,n_nodes



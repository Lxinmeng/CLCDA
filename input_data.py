import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def my_load_data2():

    dataset = "whole_feature"
    path = "data\\whole_feature\\"
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_features = np.genfromtxt("{}{}.content".format(path, dataset),
                                         dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_features[:, :], dtype=np.float32).tolil()

    cd = np.loadtxt("data\\whole_feature\\c-d_adj.txt")
    adj = sp.csr_matrix(cd[:, :], dtype=np.float32)

    return adj, features


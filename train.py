import argparse
import os
import time
from input_data import my_load_data2
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from input_data import my_load_data2
from model import *
from preprocess import mask_test_edges,  sparse_to_tuple, preprocess_graph
# from create_preprocessing import mask_test_edges, sparse_to_tuple, preprocess_graph

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Variant Graph Auto Encoder')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--dataset', '-d', type=str, default='cora', help='Dataset string.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
parser.add_argument('--features', type=int, default=1, help='Whether to use features (1) or not (0).')
parser.add_argument('--model', default='gcn_vae', help='Model string.')
parser.add_argument('--dropout', default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()


# check device
device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

def get_roc_score(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    adj_rec = adj_rec.detach().numpy()
    np.savetxt("adj_pre.txt", adj_rec)
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def main():
    model_str = args.model
    adj, features = my_load_data2()

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    if args.features == 0:
        features = sp.identity(features.shape[0])
    
    features = sparse_to_tuple(features.tocoo())
    
    # # Create model
    graph = dgl.from_scipy(adj)
    graph.add_self_loop()

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                        torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    

    features = features.to_dense()
    in_dim = features.shape[-1]
    # Create Model
    if model_str == 'gcn_ae':
        model = GCNModelAE(in_dim, args.hidden1, args.hidden2, args.dropout)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(in_dim, args.hidden1, args.hidden2, args.dropout)
    # create training component
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('Total Parameters:', sum([p.nelement() for p in model.parameters()]))


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        model.train()

        logits = model.forward(graph, features)

        # compute loss
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        if model_str == 'gcn_vae':
            kl_divergence = 0.5 / logits.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2).sum(1).mean()
            loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    test_roc, test_ap = get_roc_score(test_edges, test_edges_false, logits)
    print('Test ROC score: ', "{:.5f}".format(test_roc))


if __name__ == '__main__':
    main()

from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import device

class GCNModelVAE(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout):
        super(GCNModelVAE, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.dropout = dropout

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(self.dropout)

    def encoder(self, g, features):
        features = self.drop(features)
        h = self.layers[0](g, features)
        h = self.drop(h)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec

class GCNModelAE(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout):
        super(GCNModelAE, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        features = self.drop(features)
        h = self.layers[0](g, features)
        h = self.drop(h)
        sampled_z = self.layers[1](g, h)
        #sampled_z = self.layers[2](g, self.embeddings)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec

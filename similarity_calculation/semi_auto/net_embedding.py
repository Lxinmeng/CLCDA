# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import time
from numpy.lib.arraysetops import intersect1d
import pandas as pd
import torch
from torch import device
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam, SGD

# from data_load import netsDataset, load_networks, netsDataset2
from data_load import netsDataset, load_networks
from utils import obtain_constraints
from model import autoEncoder, encoderLoss


def set_args():
    parser = argparse.ArgumentParser(description="Train the model to obtain features for The next process")
    parser.add_argument('--cfg_path', type=str,
                        default='config/config_embedding_integration.json',
                        nargs='+', help='The configs to train model.')
    parser.add_argument('--save_path', type=str,
                        default='data/output/integration',
                        nargs='+', help="The path to save model and myresult. ")

    args = parser.parse_args()
    return args


def train_epoch(dataloader, model, idx_layer, criterion,  optimizer, epoch):
    model.train()
    losses, losses_ml = [], []
    for step, (X, y, indx) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.float(), y.float()
        indx = indx

        y_pred, _ = model(X, idx_layer)
        loss, loss_ml = criterion(y_pred, y, indx)

        losses.append(loss.item())
        losses_ml.append(loss_ml.item())

        loss.backward()
        optimizer.step()
    
    if epoch % 200 == 0:
        logging.info("{}, Epoch: {}, Training Loss: {:.5f}, ML Loss: {:.5f}"
                    .format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, np.mean(losses), np.mean(losses_ml)))


def run(args):
    # Prepare

    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, os.path.split(args.cfg_path)[-1]), 'w') as f:
        json.dump(cfg, f, indent=1)

    # init data
    constraints_ml = [np.zeros((i, i)) for i in cfg["net_dims"]]

    logging.info("### Loading mange_net from {}".format(cfg["net_paths"]))

    #read files
    networks, symbols = [], []
    cs = np.loadtxt("D:\code\CDA\similarity_calculation\\result_data\c-seq_similarity_feature.txt")  # c-seq相似网络
    cm = np.loadtxt("D:\code\CDA\similarity_calculation\\result_data\c-m_similarity_feature.txt")  # c-m相似网络
    cr = np.loadtxt("D:\code\CDA\similarity_calculation\\result_data\c-rbp_similarity_feature.txt")  # c-rbp相似网络
    pf = pd.read_excel("D:\code\CDA\similarity_calculation\data\circ_symbol.xlsx",header=None)
    networks.append(cm)
    networks.append(cr)
    networks.append(cs)
    sy = pf.values[:, 0]
    for i in range(3):
        symbols.append(sy)

    num_layers = len(cfg['hidden_dim'])

    models = [autoEncoder(cfg["net_dims"][i], cfg["hidden_dim"]) for i in range(len(networks))]

    for idx_layer in range(num_layers):

        emb = [np.zeros((cfg["net_dims"][i], cfg["hidden_dim"][idx_layer])) for i in range(cfg["net_nums"])]
        reshape_xs = []
        for idx_net in range(cfg["net_nums"]):
            # Model choose
            model = models[idx_net]
            model = model

            optimizer = Adam(model.parameters(), lr=cfg['lr'][idx_layer])
            criterion = encoderLoss(torch.from_numpy(constraints_ml[idx_net]).float(),
                                    cfg['batch_size'], cfg["gamma"])  #上一层的约束影响下一层计算


            dataset = netsDataset(networks[idx_net])
            dataloader = DataLoader(dataset, batch_size=cfg['batch_size'],
                                    shuffle=True)

            # Train the model
            logging.info("### Train {}/{} hidden layer with {}/{} mange_net, Net {}".
                         format(idx_layer+1, num_layers, idx_net+1, cfg['net_nums'], cfg['net_paths'][idx_net]))

            for epoch in range(cfg["epoch"][idx_layer]):
                train_epoch(dataloader, model, idx_layer,
                            criterion, optimizer,epoch)

            with torch.no_grad():
                model.eval()
                reshape_x, features = model(torch.from_numpy(networks[idx_net].astype(np.float32)), idx_layer, flag='reshape')
                emb[idx_net] =features.cpu().numpy()
                reshape_xs.append(reshape_x.cpu().numpy())

        logging.info("### Extracting constraints..")
        constraints_ml = obtain_constraints(cfg['net_nums'],reshape_xs,symbols,
                                            cfg['topN'], idx_layer)
        constraints_ml = constraints_ml

        networks = emb


    ## save the features
    for i, path in enumerate(cfg["net_paths"]):
        name = os.path.split(path)[-1].rsplit('.', 1)[0]
        save_path = os.path.join(args.save_path, name+"_features.txt")
        np.savetxt(save_path, networks[i])
        print(1)

def main():
    args = set_args()
    logging.basicConfig(level=logging.INFO)

    run(args)

if __name__ == "__main__":
    main()

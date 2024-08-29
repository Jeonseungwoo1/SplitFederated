import argparse
import copy
import itertools
import math
import os
import time

import random
import torch
import torch.nn as nn
import wandb

import random
import numpy as np
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from datasets import HAM10000Dataset
import network
from utils.experiment import AverageMeter, load_config



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float()/preds.shape[0]
    return acc

def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr

    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)

    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    #Foward Propagation
    fx_server = net_server(fx_client)

    #calculate loss
    loss = criterion(fx_server, y)
    #calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    #backward propagation
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    #Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)

    #count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train= []
        count1 = 0

        print('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))

        w_server = net_server.state_dict()

        if l_epoch_count == l_epoch-1:
            l_epoch_check = True
            w_locals_server.append(copy.deepcopy(w_server))


            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            if idx not in idx_collect:
                idx_collect.append(idx)

    
        if len(idx_collect) == num_users:
            fed_check = True

            w_glob_server = FedAvg(w_locals_server)

            net_glob_server.load_state_dict(w_glob_server)

            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [net_glob_server for i in range(num_users)]

            w_local_server = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    
    return dfx_client

    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=True)
    

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                iamges, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()

                #Forward propagation
                fx = net(images)
                client_fx = fx.clone().detach.requires_grad_(True)

                #sending activation to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)


                #Backward propagation
                fx.backward(dfx)
                optimizer_client.step()

            
        return net.state_dict()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFL Script")
    parser.add_argument(
        "--config", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config["wandb"]["logging"]:
        wandb.init(project="SFL-experiment", name=config["wandv"]["run_name"])

    random.seed(config["distributed"]["random_seed"])
    np.random.seed(config["distributed"]["random_seed"])
    torch.manual_seed(config["distributed"]["random_seed"])
    torch.cuda.manual_seed(config["distributed"]["random_seed"])



    program = "SFLV1 ResNet18 on HAM10000"
    print(f"---------{program}----------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_users = 5
    epochs = 200
    frac = 1
    lr=0.0001


    net_glob_client = network.ResNet18_client_side()
    net_glob_server = network.ResNet18_server_side(network.Baseblock, [2,2,2], 7)

    net_glob_client.to(device)
    net_glob_server.to(device)

    print(net_glob_client)
    print(net_glob_server)


    loss_train_collect = []
    acc_train_collect = []
    loss_test_collect = []
    acc_test_collect = []
    batch_acc_train = []
    batch_loss_train = []
    batch_acc_test = []
    batch_loss_test = []

    criterion = nn.CrossEntropyLoss()
    count1 = 0
    count2 = 0

    # to print train - test together in each round-- these are made global
    acc_avg_all_user_train = 0
    loss_avg_all_user_train = 0
    loss_train_collect_user = []
    acc_train_collect_user = []
    loss_test_collect_user = []
    acc_test_collect_user = []

    w_glob_server = net_glob_server.state_dict()
    w_locals_server = []

    #client idx collector
    idx_collect = []
    l_epoch_check = False
    fed_check = False
    # Initialization of net_model_server and net_server (server-side model)
    net_model_server = [net_glob_server for i in range(num_users)]
    net_server = copy.deepcopy(net_model_server[0]).to(device)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

    net_glob_client.train()
    w_glob_client = net_glob_client.state_dict()

    for iter in range(epochs):
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_locals_client = []

    dataset_train, dataset_test, dict_users_train, dict_users_test = HAM10000Dataset(config["data"]["train_data"], num_users, config["data"]["test_size"])


    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dateset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])

        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))

        local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell=iter)

    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")

    w_glob_client = FedAvg(w_locals_client)

    net_glob_client.load_state_dict(w_glob_client)


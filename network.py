import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image
from glob import glob
from pandas import DataFrame

import random
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

#client-side model
class ResNet18_client_side(nn.Module):
    def __init__(self):
        super(ResNet18_client_side, self).__init__()

        #conv1 part
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        #neural network module 초기화   
        #kernel_size[0]: 커널의 높이, kernel_size[1]: 커널의 너비
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_sizep[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x):
        residual1 = F.relu(self.layer1(x))
        out1 = self.layer2(residual1)
        out1 = out1 + residual1
        residual2 = F.relu(out1)
        return residual2
    
    def load_weight(self, state_dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer.") :]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_keys)


#Server-side Model
class Baseblock(nn.Module):
    expansion =1
    def __init__(self, input_planes, planes, stride=1, dim_change = None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride = stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)
        
        output += res
        output = F.relu(output)

        return output
    

class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )


        self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AvgPool2d(kernel_size=7, stride =1)
        self.fc = nn.linear(512 * block.expansion, classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _layer(self, block, planes, num_layers, stride=2):
        dim_change= None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes*block.expansion)
                )
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)
    

    def forward(self, x):
        out2 = self.layer3(x)
        out2 = out2 + x
        x3 = F.relu(out2)


        x4 = self.layer4(x3)
        x5 = self. layer5(x4)
        x6 = self.layer6(x5)

        x7 = F.avg_pool2d(x6, 7)
        x8 = x7.view(x7.size(0), -1)
        y_hat = self.fc(x8)

        return y_hat
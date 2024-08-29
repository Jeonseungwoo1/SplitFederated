import json
import os
import random
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


class SkinData(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64,64))
        y = torch.tensor(int(self.df['target'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

def HAM10000Dataset(data_dir, num_users, test_size=0.2):
    df = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))

    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join(data_dir, '*', '*.jpg'))}
    
    df['path'] = df['image_id'].map(imageid_path.get)
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes

    train, test = train_test_split(df, test_size=test_size)
    train = train.reset_index()
    test = test.reset_index()

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Pad(3),
        transforms.RandomRotation(10),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.Pad(3),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train = SkinData(train, transform = train_transforms)
    dataset_test = SkinData(test, transforms=test_transforms)

    dict_users_train = dataset_iid(dataset_train, num_users)
    dict_users_test = dataset_iid(dataset_test, num_users)

    return dataset_train, dataset_test, dict_users_train, dict_users_test
